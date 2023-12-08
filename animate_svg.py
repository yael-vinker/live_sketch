from painter import Painter, PainterOptimizer
from losses import SDSVideoLoss
import utils
import os
import matplotlib.pyplot as plt
import torch
import pydiffvg
from tqdm import tqdm
from ipywidgets import Video
from pytorch_lightning import seed_everything
import argparse
import wandb
import numpy as np
from torchvision import transforms
import torchvision
import copy


def parse_arguments():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--target", type=str, default="svg_input/horse_256-01", help="file name of the svg to be animated")
    parser.add_argument("--caption", type=str, default="", help="Prompt for animation. verify first that this prompt works with the original text2vid model. If left empty will try to find prompt in utils.py")
    parser.add_argument("--output_folder", type=str, default="horse_256", help="folder name to save the results")
    parser.add_argument("--seed", type=int, default=1000)

    # Diffusion related & Losses
    parser.add_argument("--model_name", type=str, default="damo-vilab/text-to-video-ms-1.7b")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--guidance_scale", type=float, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--render_size_h", type=int, default=256, help="should fit the default settings of the chosen video model (under 'model_name')")
    parser.add_argument("--render_size_w", type=int, default=256, help="should fit the default settings of the chosen video model (under 'model_name')")
    parser.add_argument("--num_frames", type=int, default=24, help="should fit the default settings of the chosen video model (under 'model_name')")
    
    # SDS relted
    parser.add_argument("--sds_timestep_low", type=int, default=50) 
    parser.add_argument("--same_noise_for_frames", action="store_true", help="sample noise for one frame and repeat across all frames")
    parser.add_argument("-augment_frames", type=bool, default=True, help="whether to randomely augment the frames to prevent adversarial results")

    # Memory saving related
    parser.add_argument("--use_xformers", action="store_true", help="Enable xformers for unet")
    parser.add_argument("--del_text_encoders", action="store_true", help="delete text encoder and tokenizer after encoding the prompts")

    # Optimization related
    parser.add_argument("--num_iter", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--optim_points", type=bool, default=True, help="whether to optimize the points (x,y) of the object or not")
    parser.add_argument("--opt_points_with_mlp", type=bool, default=True, help="whether to optimize the points with an MLP")

    parser.add_argument("--split_global_loss", type=bool, default=True, help="whether to use a different loss for the center prediction")
    parser.add_argument("--guidance_scale_global", type=float, default=40, help="SDS guidance scale for the global path")
    parser.add_argument("--lr_base_global", type=float, default=0.0001, help="Base learning rate for the global path")

    # MLP architecture (points)
    parser.add_argument("--predict_global_frame_deltas", type=float, default=1, help="whether to predict a global delta per frame, the value is the weight of the output")
    parser.add_argument("-predict_only_global", action='store_true', help="whether to predict only global deltas")
    parser.add_argument("--inter_dim", type=int, default=128)
    parser.add_argument("--use_shared_backbone_for_global", action='store_true',
                        help="Whether to use the same backbone for the global prediction as for per point prediction")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--normalize_input", type=int, default=0)
    parser.add_argument("--translation_layer_norm_weight", type=int, default=0)

    parser.add_argument("--rotation_weight", type=float, default=0.01, help="Scale factor for global transform matrix 'rotation' terms")
    parser.add_argument("--scale_weight", type=float, default=0.05, help="Scale factor for global transform matrix 'scale' terms")
    parser.add_argument("--shear_weight", type=float, default=0.1, help="Scale factor for global transform matrix 'shear' terms")
    parser.add_argument("--translation_weight", type=float, default=1, help="Scale factor for global transform matrix 'translation' terms")

    # Learning rate related (can be simplified, taken from vectorFusion)
    parser.add_argument("--lr_local", type=float, default=0.005)
    parser.add_argument("--lr_init", type=float, default=0.002)
    parser.add_argument("--lr_final", type=float, default=0.0008)
    parser.add_argument("--lr_delay_mult", type=float, default=0.1)
    parser.add_argument("--lr_delay_steps", type=float, default=100)
    parser.add_argument("--const_lr", type=int, default=0)

    # Display related
    parser.add_argument("--display_iter", type=int, default=50)
    parser.add_argument("--save_vid_iter", type=int, default=100)

    # wandb
    parser.add_argument("-report_to_wandb", action='store_true')
    parser.add_argument("--wandb_user", type=str)
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--folder_as_wandb_run_name", action="store_true")

    args = parser.parse_args()
    seed_everything(args.seed)

    if not args.caption:
        args.caption = utils.get_caption(args.target)
        
    print("=" * 50)
    print("target:", args.target)
    print("caption:", args.caption)
    print("=" * 50)

    if args.folder_as_wandb_run_name:
        args.wandb_run_name = args.output_folder

    args.output_folder = f"./output_videos/{args.output_folder}"
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(f"{args.output_folder}/svg_logs", exist_ok=True)
    os.makedirs(f"{args.output_folder}/mp4_logs", exist_ok=True)
    
       

    if args.report_to_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                    config=args, name=args.wandb_run_name, id=wandb.util.generate_id())


    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    return args


def plot_video_seq(x_aug, orig_aug, cfg, step):
    pair_concat = torch.cat([orig_aug.squeeze(0).detach().cpu(), x_aug.squeeze(0).detach().cpu()])
    grid_img = torchvision.utils.make_grid(pair_concat, nrow=cfg.num_frames)
    plt.figure(figsize=(30,10))
    plt.imshow(grid_img.permute(1, 2, 0), vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"frames_iter{step}")
    plt.tight_layout()
    if cfg.report_to_wandb:
        wandb.log({"frames": wandb.Image(plt)}, step=step)


def get_augmentations():
    augemntations = []
    augemntations.append(transforms.RandomPerspective(
        fill=1, p=1.0, distortion_scale=0.5))
    augemntations.append(transforms.RandomResizedCrop(
        size=(256,256), scale=(0.4, 1), ratio=(1.0, 1.0)))
    augment_trans = transforms.Compose(augemntations)
    return augment_trans


if __name__ == "__main__":
    cfg = parse_arguments()

    # Everything about rasterization and curves is defined in the Painter class
    painter = Painter(cfg, cfg.target, num_frames=cfg.num_frames, device=cfg.device)
    optimizer = PainterOptimizer(cfg, painter)
    data_augs = get_augmentations()

    # Just to test that the svg and initial frames were loaded as expected
    frames_tensor, frames_svg, points_init_frame = painter.render_frames_to_tensor(mlp=False)
    output_vid_path = f"{cfg.output_folder}/init_vid.mp4"
    utils.save_mp4_from_tensor(frames_tensor, output_vid_path)

    if cfg.report_to_wandb:
        video_to_save = frames_tensor.permute(0,3,1,2).detach().cpu().numpy()
        video_to_save = ((video_to_save / video_to_save.max()) * 255).astype(np.uint8)
        wandb.log({"video_init": wandb.Video(video_to_save, fps=8)})
                       
    sds_loss = SDSVideoLoss(cfg, cfg.device)

    # If requested, set up a loss with different params for the global path.
    # Will re-use the same text-2-video diffusion pipeline
    if cfg.predict_global_frame_deltas and cfg.split_global_loss:
        global_cfg = copy.deepcopy(cfg)
        if cfg.guidance_scale_global is not None:
            global_cfg.guidance_scale = cfg.guidance_scale_global
            global_sds_loss = SDSVideoLoss(global_cfg, global_cfg.device, reuse_pipe=True)

    orig_frames = frames_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3) # (K, 256, 256, 3) -> (1, K, 3, 256, 256)
    orig_frames = orig_frames.repeat(cfg.batch_size, 1, 1, 1, 1)

    sds_losses_and_opt_kwargs = []

    if cfg.predict_global_frame_deltas:
        sds_losses_and_opt_kwargs.append((sds_loss, {"skip_global": True}))
        sds_losses_and_opt_kwargs.append((global_sds_loss, {"skip_points": True}))
    else:
        sds_losses_and_opt_kwargs.append((sds_loss, {}))

    t_range = tqdm(range(cfg.num_iter + 1))
    for step in t_range:
        for curr_sds_loss, opt_kwargs in sds_losses_and_opt_kwargs:
            loss_kwargs = {}
            logs = {}
            optimizer.zero_grad_()

            # Render the frames (inc. network forward pass)
            vid_tensor, frames_svg, new_points = painter.render_frames_to_tensor() # (K, 256, 256, 3)
            x = vid_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)  # (K, 256, 256, 3) -> (1, K, 3, 256, 256)
            x = x.repeat(cfg.batch_size, 1, 1, 1, 1)

            # Apply augmentations if needed
            if cfg.augment_frames:
                augmented_pair = data_augs(torch.cat([x.squeeze(0), orig_frames.squeeze(0)]))
                x_aug = augmented_pair[:cfg.num_frames].unsqueeze(0)
                orig_frames_aug = augmented_pair[cfg.num_frames:].unsqueeze(0)
            else:
                x_aug = x
                orig_frames_aug = orig_frames
            
            # Compute SDS loss. Note: The returned loss value is always a placeholder "1".
            # SDS is applied by changing the backprop calculation, see SpecifyGradient in losses.py 
            loss_sds = curr_sds_loss(x_aug, **loss_kwargs)
            loss = loss_sds

            t_range.set_postfix({'loss': loss.item()})
            loss.backward()

            optimizer.step_(**opt_kwargs)
            
            loss_suffix = "_global" if "skip_points" in opt_kwargs else ""
            logs.update({f"loss{loss_suffix}": loss.detach().item(), f"loss_sds{loss_suffix}": loss_sds.detach().item()})

        if not cfg.const_lr:
            optimizer.update_lr()

        logs.update({"lr_points": optimizer.get_lr("points"), "step": step})

        if cfg.report_to_wandb:
            wandb.log(logs, step=step)

        if step % cfg.save_vid_iter == 0:
            utils.save_mp4_from_tensor(vid_tensor, f"{cfg.output_folder}/mp4_logs/{step}.mp4")
            utils.save_vid_svg(frames_svg, f"{cfg.output_folder}/svg_logs", step, painter.canvas_width, painter.canvas_height)
            if cfg.report_to_wandb:
                video_to_save = vid_tensor.permute(0,3,1,2).detach().cpu().numpy()
                video_to_save = ((video_to_save / video_to_save.max()) * 255).astype(np.uint8)
                wandb.log({"video": wandb.Video(video_to_save, fps=8)}, step=step)
                plot_video_seq(x_aug, orig_frames_aug, cfg, step)
            
            if step > 0:
                painter.log_state(f"{cfg.output_folder}/models/")

    if cfg.report_to_wandb:
        wandb.finish()
    
    # Saves a high quality .gif from the final SVG frames
    utils.save_hq_video(cfg.output_folder, iter_=cfg.num_iter)

