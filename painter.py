import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.optim.lr_scheduler import LambdaLR
from torchsummary import summary
import pydiffvg

import utils


class Painter(torch.nn.Module):
    def __init__(self,
                 args,
                 svg_path: str,
                 num_frames: int,
                 device,
                 path_to_trained_mlp=None,
                 inference=False):
        super(Painter, self).__init__()
        self.svg_path = svg_path
        self.num_frames = num_frames
        self.device = device
        self.optim_points = args.optim_points
        self.opt_points_with_mlp = args.opt_points_with_mlp
        self.render = pydiffvg.RenderFunction.apply
        self.normalize_input = args.normalize_input

        self.init_shapes()
        if self.opt_points_with_mlp:
            self.points_mlp_input_ = self.points_mlp_input.unsqueeze(0).to(device)
            self.mlp_points = PointMLP(input_dim=torch.numel(self.points_mlp_input),
                                        inter_dim=args.inter_dim,
                                        num_points_per_frame=self.points_per_frame,
                                        num_frames=num_frames,
                                        device=device,
                                        predict_global_frame_deltas=args.predict_global_frame_deltas,
                                        predict_only_global=args.predict_only_global, 
                                        inference=inference,
                                        rotation_weight=args.rotation_weight, 
                                        scale_weight=args.scale_weight, 
                                        shear_weight=args.shear_weight,
                                        translation_weight=args.translation_weight).to(device)
            
            if path_to_trained_mlp:
                print(f"Loading MLP from {path_to_trained_mlp}")
                self.mlp_points.load_state_dict(torch.load(path_to_trained_mlp))
                self.mlp_points.eval()

            # Init the weights of LayerNorm for global translation MLP if needed.
            if args.translation_layer_norm_weight:
                self.init_translation_norm(args.translation_layer_norm_weight)
        

    def init_shapes(self):
        """
        Loads the svg file from svg_path and set grads to the parameters we want to optimize
        In this case, we optimize the delta from the center and the deltas from the original points
        """
        parameters = edict()
        # a list of points (x,y) ordered by shape, len = num_frames * num_shapes_per_frame 
        # each element in the list is a (num_point_in_shape, 2) tensor
        parameters.point_delta = []

        frames_shapes, frames_shapes_group = [], []  # a list with len "num_frames" of lists of "Path" objects, each Patch has x,y points
        frames_xy_deltas_from_center = []  # a list with len "num_frames", for each frame we save a list of (x,y) ccordinates of the distance from the center
        svg_cur_path = f'{self.svg_path}.svg'
        # init the canvas_width, canvas_height
        self.canvas_width, self.canvas_height, shapes_init_, shape_groups_init_ = pydiffvg.svg_to_scene(svg_cur_path)
        self.points_per_frame = 0
        for s_ in shapes_init_:
            self.points_per_frame += s_.points.shape[0]

        print(f"A single frame contains {self.points_per_frame} points")
        # save the original center
        center_, all_points = get_center_of_mass(shapes_init_)
        self.original_center = center_.clone()
        self.original_center.requires_grad = False
        self.original_center = self.original_center.to(self.device)

        # extending the initial SVG into num_frames (default 24) frames
        for i in range(self.num_frames):
            canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg_cur_path)
            center_cur, all_points = get_center_of_mass(shapes_init)
            # init the learned (x,y) deltas from center
            deltas_from_center = get_deltas(all_points, center_, self.device)
            frames_xy_deltas_from_center.append(deltas_from_center)
            for k in range(len(shapes_init)):
                points_p = deltas_from_center[k].to(self.device)
                if self.optim_points and not self.opt_points_with_mlp:
                    points_p.requires_grad = True
                parameters.point_delta.append(points_p)

            # we add the shapes to the list after we set the grads
            frames_shapes.append(shapes_init)
            frames_shapes_group.append(shape_groups_init)

        self.frames_shapes = frames_shapes
        self.frames_shapes_group = frames_shapes_group
        self.frames_xy_deltas_from_center = frames_xy_deltas_from_center  # note that frames_xy_deltas_from_center points to parameters.point_delta so these values are being updated as well
        tensor_points_init = [torch.cat(self.frames_xy_deltas_from_center[i])
                              for i in range(len(self.frames_xy_deltas_from_center))]
        self.points_mlp_input = torch.cat(tensor_points_init)
        self.parameters_ = parameters
    

    def render_frames_to_tensor_mlp(self):
        # support only MLP for now
        frames_init, frames_svg, all_new_points = [], [], []

        prev_points = self.points_mlp_input_.clone().squeeze(0)[:self.points_per_frame] + self.original_center  # [64, 2] -> [points_per_frame, 2]
        frame_input = self.points_mlp_input_
        # normalize the frame_input to be between -1 and 1
        if self.normalize_input:
            frame_input = utils.normalize_tensor(frame_input)
        delta_prediction = self.mlp_points(frame_input)  # [1024, 2], [16*points_per_frame, 2]

        for i in range(self.num_frames):
            shapes, shapes_groups = self.frames_shapes[i], self.frames_shapes_group[i]
            new_shapes, new_shape_groups, frame_new_points = [], [], []  # for SVG frames saving

            start_frame_slice = i * self.points_per_frame
            # take all deltas for current frame
            point_delta_leanred_cur_frame = delta_prediction[
                                            start_frame_slice: start_frame_slice + self.points_per_frame,
                                            :]  # [64, 2] -> [points_per_frame, 2]
            points_cur_frame = prev_points + point_delta_leanred_cur_frame

            counter = 0
            for j in range(len(shapes)):
                # for differentiability we need to redefine and render all paths
                shape, shapes_group = shapes[j], shapes_groups[j]
                points_vars = shape.points.clone()
                points_vars[:, 0] = points_cur_frame[counter:counter + shape.points.shape[0], 0]
                points_vars[:, 1] = points_cur_frame[counter:counter + shape.points.shape[0], 1]
                counter += shape.points.shape[0]

                frame_new_points.append(points_vars.to(self.device))
                path = pydiffvg.Path(
                    num_control_points=shape.num_control_points, points=points_vars,
                    stroke_width=shape.stroke_width, is_closed=shape.is_closed)
                new_shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(new_shapes) - 1]),
                    fill_color=shapes_group.fill_color,
                    stroke_color=torch.tensor([0, 0, 0, 1]))
                new_shape_groups.append(path_group)

            scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes,
                                                                 new_shape_groups)
            cur_im = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)
            cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                     torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=self.device) * (1 - cur_im[:, :, 3:4])
            cur_im = cur_im[:, :, :3]
            frames_init.append(cur_im)
            frames_svg.append((new_shapes, new_shape_groups))
            all_new_points.append(frame_new_points)
        return torch.stack(frames_init), frames_svg, all_new_points

    def render_frames_to_tensor_direct_optim(self):
        frames_init, frames_svg, points_init_frame = [], [], []

        for i in range(self.num_frames):
            shapes = self.frames_shapes[i]
            shapes_groups = self.frames_shapes_group[i]
            new_shapes, new_shape_groups = [], []

            deltas_from_center_cur_frame = self.frames_xy_deltas_from_center[i]
            for j in range(len(shapes)):
                shape, shapes_group = shapes[j], shapes_groups[j]
                point_delta_leanred = deltas_from_center_cur_frame[j]
                points_vars = shape.points.clone()

                points_vars[:, 0] = point_delta_leanred[:, 0] + self.original_center[0]
                points_vars[:, 1] = point_delta_leanred[:, 1] + self.original_center[1]
                if i == 0: # only for a single frame
                    points_init_frame.append(points_vars)

                path = pydiffvg.Path(
                    num_control_points=shape.num_control_points, points=points_vars,
                    stroke_width=shape.stroke_width, is_closed=shape.is_closed)
                new_shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(new_shapes) - 1]),
                    fill_color=shapes_group.fill_color,
                    stroke_color=torch.tensor([0, 0, 0, 1]))
                new_shape_groups.append(path_group)

            scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes,
                                                                 new_shape_groups)
            cur_im = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)

            cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                     torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=self.device) * (1 - cur_im[:, :, 3:4])
            cur_im = cur_im[:, :, :3]
            frames_init.append(cur_im)
            frames_svg.append((new_shapes, new_shape_groups))
        return torch.stack(frames_init), frames_svg, points_init_frame

    def render_frames_to_tensor(self, mlp=True):
        if self.opt_points_with_mlp and mlp:
            return self.render_frames_to_tensor_mlp()
        else:
            return self.render_frames_to_tensor_direct_optim()

    def get_points_params(self):
        if self.opt_points_with_mlp:
            return self.mlp_points.get_points_params()
        return self.parameters_["point_delta"]

    def get_global_params(self):
        return self.mlp_points.get_global_params()

    def log_state(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        torch.save(self.mlp_points.state_dict(), f"{output_path}/model.pt")
        print(f"Model saved to {output_path}/model.pt")

    def init_translation_norm(self, translation_layer_norm_weight):
        print(f"Initializing translation layerNorm to {translation_layer_norm_weight}")
        for child in self.mlp_points.frames_rigid_translation.children():
            if isinstance(child, nn.LayerNorm):
                with torch.no_grad():
                    child.weight *= translation_layer_norm_weight



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        # x = x + 
        return self.dropout(self.pe[:x.size(0), :])


class PointModel(nn.Module):

    def __init__(self, input_dim, inter_dim, num_points_per_frame, num_frames,
                 device, predict_global_frame_deltas, predict_only_global, inference=False, 
                 rotation_weight=1e-2, scale_weight=5e-2, shear_weight=5e-2, translation_weight=1):

        super().__init__()
        self.num_points_per_frame = num_points_per_frame
        self.num_frames = num_frames
        self.inter_dim = inter_dim
        self.input_dim = input_dim
        self.embed_dim = inter_dim
        self.predict_global_frame_deltas = predict_global_frame_deltas
        self.predict_only_global = predict_only_global
        self.inference = inference

        self.project_points = nn.Sequential(nn.Linear(2, inter_dim),
                                            nn.LayerNorm(inter_dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(inter_dim, inter_dim))

        self.embedding = nn.Embedding(input_dim, self.embed_dim)
        self.pos_encoder = PositionalEncoding(d_model=self.embed_dim, max_len=input_dim)
        self.inds = torch.tensor(range(int(input_dim / 2))).to(device)

        if predict_global_frame_deltas:
            self.rotation_weight = rotation_weight
            self.scale_weight = scale_weight
            self.shear_weight = shear_weight
            self.translation_weight = translation_weight

            self.frames_rigid_shared = nn.Sequential(nn.Flatten(),
                                              nn.Linear(int(input_dim * inter_dim / 2), inter_dim),
                                              nn.LayerNorm(inter_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(inter_dim, inter_dim),
                                              nn.LayerNorm(inter_dim),
                                              nn.LeakyReLU())

            self.frames_rigid_translation = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                          nn.LayerNorm(inter_dim),
                                                          nn.LeakyReLU(),
                                                          nn.Linear(inter_dim, inter_dim),
                                                          nn.LayerNorm(inter_dim),
                                                          nn.LeakyReLU(),
                                                          nn.Linear(inter_dim, inter_dim),
                                                          nn.LayerNorm(inter_dim),
                                                          nn.LeakyReLU(),
                                                          nn.Linear(inter_dim, self.num_frames * 2))

            self.frames_rigid_rotation = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                       nn.LayerNorm(inter_dim),
                                                       nn.LeakyReLU(),
                                                       nn.Linear(inter_dim, self.num_frames * 1))
            
            self.frames_rigid_shear = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                    nn.LayerNorm(inter_dim),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(inter_dim, self.num_frames * 2))

            self.frames_rigid_scale = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                    nn.LayerNorm(inter_dim),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(inter_dim, self.num_frames * 2))

            self.global_layers = nn.ModuleList([self.frames_rigid_shared, 
                                  self.frames_rigid_translation, 
                                  self.frames_rigid_rotation, 
                                  self.frames_rigid_shear, 
                                  self.frames_rigid_scale,
                                  ])


    def get_position_encoding_representation(self, init_points):
        # input dim: init_points [num_frames * points_per_frame, 2], for ballerina [832,2] = [16*52, 2]
        # the input are the points of the given initial frame (user's drawing)
        # note that we calculate the point's distance from the object's center, and operate on this distance
        emb_xy = self.project_points(init_points)  # output shape: [1,num_frames * points_per_frame,128] -> [1,832,128]
        embed = self.embedding(self.inds) * math.sqrt(self.embed_dim)  # inds dim is N*K, embed dim is [N*K, 128]
        pos = self.pos_encoder(embed.unsqueeze(1)).permute(1, 0, 2)  # [1, N*K, 128]
        init_points_pos_enc = emb_xy + pos  # [1, N*K, 128]
        return init_points_pos_enc

    def get_frame_deltas(self, init_points, init_points_pos_enc):
        # learn global deltas per frame, via [1,N*K,128] -> [16, 2] (delta x,y per frame) -> [N*K, 2] (expend to shape)
        frame_deltas = None
        if self.predict_global_frame_deltas:
            shared_params = self.frames_rigid_shared(init_points_pos_enc)

            # calculate transform matrix parameters
            dx, dy = self.frames_rigid_translation(shared_params).reshape(self.num_frames, 2).chunk(2, axis=-1)
            dx = dx * self.translation_weight
            dy = dy * self.translation_weight

            theta = self.frames_rigid_rotation(shared_params).reshape(self.num_frames, 1) * self.rotation_weight
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            shear_x, shear_y = self.frames_rigid_shear(shared_params).reshape(self.num_frames, 2).chunk(2, axis=-1)

            shear_x = shear_x * self.shear_weight
            shear_y = shear_y * self.shear_weight

            scale_x, scale_y = self.frames_rigid_shear(shared_params).reshape(self.num_frames, 2).chunk(2, axis=-1)
            
            scale_x = torch.ones_like(dx) + scale_x * self.scale_weight
            scale_y = torch.ones_like(dy) + scale_y * self.scale_weight

            # prepare transform matrix
            l1 = torch.concat([scale_x * (cos_theta - sin_theta * shear_x), scale_y * (cos_theta * shear_y - sin_theta), dx], axis=-1)
            l2 = torch.concat([scale_x * (sin_theta + cos_theta * shear_x), scale_y * (sin_theta * shear_y + cos_theta), dy], axis=-1)
            l3 = torch.concat([torch.zeros_like(dx), torch.zeros_like(dx), torch.ones_like(dx)], axis=-1)

            transform_mat = torch.stack([l1, l2, l3], axis=1)

            transform_mat = torch.repeat_interleave(transform_mat, self.num_points_per_frame, dim=0)

            # extend points for calculation
            points_with_z = torch.concat([init_points, torch.ones_like(init_points)[:,:,0:1]], axis=-1)
            points_with_z = points_with_z.reshape(-1, 3, 1)

            # calculate new coordinates and deltas
            transformed_points = torch.matmul(transform_mat, points_with_z)[:, 0:2, :].reshape(1, -1, 2)
            frame_deltas = transformed_points - init_points
            # frame_deltas *= self.predict_global_frame_deltas

        return frame_deltas

    def forward(self, init_points):
        raise NotImplementedError("PointModel is an abstract class. Please inherit from it and implement a forward function.")

    def get_shared_params(self):
        project_points_p = list(self.project_points.parameters())
        embedding_p = list(self.embedding.parameters())
        pos_encoder_p = list(self.pos_encoder.parameters())

        return project_points_p + embedding_p + pos_encoder_p
        
    def get_points_params(self):
        shared_params = self.get_shared_params()
        project_xy_p = list(self.project_xy.parameters())
        model_p = list(self.model.parameters())
        last_lin = list(self.last_linear_layer.parameters())
        return shared_params + project_xy_p + model_p + last_lin
        
    def get_global_params(self):
        shared_params = self.get_shared_params()

        delta_p = list(self.global_layers.parameters())
        return shared_params + delta_p

class PointMLP(PointModel):
    def __init__(self, input_dim, inter_dim, num_points_per_frame, num_frames,
                 device, predict_global_frame_deltas, predict_only_global, inference,
                 rotation_weight=1e-2, scale_weight=5e-2, shear_weight=5e-2, translation_weight=1):

        super().__init__(input_dim, inter_dim, num_points_per_frame, num_frames,
                         device, predict_global_frame_deltas, predict_only_global, inference,
                         rotation_weight, scale_weight, shear_weight, translation_weight)

        self.project_xy = nn.Sequential(nn.Flatten(),
                                        nn.Linear(int(input_dim * inter_dim / 2), inter_dim),
                                        nn.LayerNorm(inter_dim),
                                        nn.LeakyReLU())

        self.model = nn.Sequential(
            nn.Linear(inter_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
        )

        self.last_linear_layer = nn.Linear(inter_dim, input_dim)

    def forward(self, init_points):
        init_points_pos_enc = self.get_position_encoding_representation(init_points)
        frame_deltas = self.get_frame_deltas(init_points, init_points_pos_enc)
        if self.predict_only_global:
            return frame_deltas.squeeze(0)

        project_xy = self.project_xy(init_points_pos_enc)  # Flatten, output is [1, 128]
        delta = self.model(project_xy)  # [1,128]
        delta_xy = self.last_linear_layer(delta).reshape(init_points.shape)  # [1,128] -> [1, N*K, 2]

        if self.predict_global_frame_deltas:
            delta_xy = delta_xy + frame_deltas

        return delta_xy.squeeze(0)

class PainterOptimizer:
    def __init__(self, args, painter):
        self.painter = painter
        self.lr_local = args.lr_local
        self.lr_base_global = args.lr_base_global
        self.lr_init = args.lr_init
        self.lr_final = args.lr_final
        self.lr_delay_mult = args.lr_delay_mult
        self.lr_delay_steps = args.lr_delay_steps
        self.max_steps = args.num_iter
        self.lr_lambda = lambda step: self.learning_rate_decay(step) / self.lr_init
        self.optim_points = args.optim_points
        self.optim_global = args.split_global_loss
        self.init_optimizers()

    def learning_rate_decay(self, step):
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp

    def init_optimizers(self):

        if self.optim_global:
            global_frame_params = self.painter.get_global_params()
            self.global_delta_optimizer = torch.optim.Adam(global_frame_params, lr=self.lr_base_global,
                                                           betas=(0.9, 0.9), eps=1e-6)
            self.scheduler_global = LambdaLR(self.global_delta_optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

        if self.optim_points:
            points_delta_params = self.painter.get_points_params()
            self.points_delta_optimizer = torch.optim.Adam(points_delta_params, lr=self.lr_local,
                                                           betas=(0.9, 0.9), eps=1e-6)
            self.scheduler_points = LambdaLR(self.points_delta_optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

    def update_lr(self):
        if self.optim_global:
            self.scheduler_global.step()
        if self.optim_points:
            self.scheduler_points.step()

    def zero_grad_(self):
        if self.optim_points:
            self.points_delta_optimizer.zero_grad()

    def step_(self, skip_global=False, skip_points=False):
        if self.optim_global and not skip_global:
            self.global_delta_optimizer.step()
        if self.optim_points and not skip_points:
            self.points_delta_optimizer.step()

    def get_lr(self, optim="points"):
        if optim == "points" and self.optim_points:
            return self.points_delta_optimizer.param_groups[0]['lr']
        else:
            return None


def get_center_of_mass(shapes):
    all_points = []
    for shape in shapes:
        all_points.append(shape.points)
    points_vars = torch.vstack(all_points)
    center = points_vars.mean(dim=0)
    return center, all_points


def get_deltas(all_points, center, device):
    deltas_from_center = []
    for points in all_points:
        deltas = (points - center).to(device)
        deltas_from_center.append(deltas)
    return deltas_from_center

