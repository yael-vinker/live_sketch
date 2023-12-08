from diffusers import DiffusionPipeline
import torch.nn as nn
import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from model_utils import configure_lora
import einops

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import Delaunay
import numpy as np
from torch.nn import functional as nnf


# =============================================
# ===== Helper function for SDS gradients =====
# =============================================
class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


# ========================================================
# ===== Basic class to extend with SDS loss variants =====
# ========================================================
class SDSLossBase(nn.Module):

    _global_pipe = None

    def __init__(self, cfg, device, reuse_pipe=True):
        super(SDSLossBase, self).__init__()

        self.cfg = cfg
        self.device = device

        # initiate a diffusion pipeline if we don't already have one / don't want to reuse it for both paths
        self.maybe_init_pipe(reuse_pipe) 

        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        if cfg.use_xformers:
            self.pipe.enable_xformers_memory_efficient_attention()

        self.text_embeddings = self.embed_text(self.cfg.caption)

        if self.cfg.del_text_encoders:
            del self.pipe.tokenizer
            del self.pipe.text_encoder

    def maybe_init_pipe(self, reuse_pipe):
        if reuse_pipe:
            if SDSLossBase._global_pipe is None:
                SDSLossBase._global_pipe = DiffusionPipeline.from_pretrained(self.cfg.model_name, torch_dtype=torch.float16, variant="fp16")
                SDSLossBase._global_pipe = SDSLossBase._global_pipe.to(self.device)
            self.pipe = SDSLossBase._global_pipe
        else:
            self.pipe = DiffusionPipeline.from_pretrained(self.cfg.model_name, torch_dtype=torch.float16, variant="fp16")
            self.pipe = self.pipe.to(self.device)

    def embed_text(self, caption):
        # tokenizer and embed text
        text_input = self.pipe.tokenizer(caption, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                         max_length=text_input.input_ids.shape[-1],
                                         return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
            
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings = text_embeddings.repeat_interleave(self.cfg.batch_size, 0)

        return text_embeddings

        
    def prepare_latents(self, x_aug):
        x = x_aug * 2. - 1. # encode rendered image, values should be in [-1, 1]
        
        with torch.cuda.amp.autocast():
            batch_size, num_frames, channels, height, width = x.shape # [1, K, 3, 256, 256], for K frames
            x = x.reshape(batch_size * num_frames, channels, height, width) # [K, 3, 256, 256], for the VAE encoder
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample()) # [K, 4, 32, 32]
            frames, channel, h_, w_ = init_latent_z.shape
            init_latent_z = init_latent_z[None, :].reshape(batch_size, num_frames, channel, h_, w_).permute(0, 2, 1, 3, 4) # [1, 4, K, 32, 32] for the video model
            
        latent_z = self.pipe.vae.config.scaling_factor * init_latent_z  # scaling_factor * init_latents

        return latent_z

    def add_noise_to_latents(self, latent_z, timestep, return_noise=True, eps=None):
        
        # sample noise if not given some as an input
        if eps is None:
            if self.cfg.same_noise_for_frames: # This works badly. Do not use.
                eps = torch.randn_like(latent_z[:, :, 0, :, :]) # create noise for single frame
                eps = einops.repeat(eps, 'b c h w -> b c f h w', f=latent_z.shape[2])
            else:
                eps = torch.randn_like(latent_z)

        # zt = alpha_t * latent_z + sigma_t * eps
        noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

        if return_noise:
            return noised_latent_zt, eps

        return noised_latent_zt
    
    # overload this if inheriting for VSD etc.
    def get_sds_eps_to_subract(self, eps_orig, z_in, timestep_in):
        return eps_orig

    def drop_nans(self, grads):
        assert torch.isfinite(grads).all()
        return torch.nan_to_num(grads.detach().float(), 0.0, 0.0, 0.0)

    def get_grad_weights(self, timestep):
        return (1 - self.alphas[timestep])

    def sds_grads(self, latent_z, **sds_kwargs):

        with torch.no_grad():
            # sample timesteps
            timestep = torch.randint(
                low=self.cfg.sds_timestep_low,
                high=min(950, self.cfg.timesteps) - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            noised_latent_zt, eps = self.add_noise_to_latents(latent_z, timestep, return_noise=True)

            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = self.pipe.unet(z_in, timestep_in, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)
            
            eps_t = eps_t_uncond + self.cfg.guidance_scale * (eps_t - eps_t_uncond)

            eps_to_subtract = self.get_sds_eps_to_subract(eps, z_in, timestep_in, **sds_kwargs)

            w = self.get_grad_weights(timestep)
            grad_z = w * (eps_t - eps_to_subtract)

            grad_z = self.drop_nans(grad_z)

        return grad_z

# =======================================
# =========== Basic SDS loss  ===========
# =======================================
class SDSVideoLoss(SDSLossBase):
    def __init__(self, cfg, device, reuse_pipe=True):
        super(SDSVideoLoss, self).__init__(cfg, device, reuse_pipe=reuse_pipe)

    def forward(self, x_aug, grad_scale=1.0):
        latent_z = self.prepare_latents(x_aug)

        grad_z = grad_scale * self.sds_grads(latent_z)

        sds_loss = SpecifyGradient.apply(latent_z, grad_z)

        return sds_loss    

# =====================================================
# =============== VSD loss (DEPRECATED) ===============
# == Left 'as-is' in case someone wants to try again ==
# =====================================================
class VSDVideoLoss(SDSLossBase):
    def __init__(self, cfg, device):
        super(VSDVideoLoss, self).__init__(cfg, device)
        self.cfg = cfg
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(cfg.model_name, torch_dtype=torch.float16, variant="fp16")
        self.pipe = self.pipe.to(self.device)


        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        self.pipe_lora = DiffusionPipeline.from_pretrained(cfg.model_name, torch_dtype=torch.float16, variant="fp16")
        self.pipe_lora.to(device)

        self.pipe_lora.unet, self.lora_layers = configure_lora(self.pipe_lora.unet, device)

        if cfg.use_xformers:
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe_lora.enable_xformers_memory_efficient_attention()

        self.pipe.unet.requires_grad_(False)
        self.pipe_lora.unet.requires_grad_(False)
        self.pipe_lora.vae.requires_grad_(False)

        self.num_train_timesteps = len(self.pipe.scheduler.betas)
        self.pipe.scheduler.set_timesteps(self.num_train_timesteps)

        self.lora_loss_func = nn.MSELoss()

    def get_sds_eps_to_subract(self, eps_orig, z_in, timestep_in, **kwargs):

        alpha_target = kwargs["alpha_target"]

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            lora_eps_t_uncond, lora_eps_t = self.pipe_lora.unet(z_in, timestep_in, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)

        lora_eps_t = lora_eps_t_uncond + self.cfg.lora_guidance_scale * (lora_eps_t - lora_eps_t_uncond)

        target_eps = alpha_target * lora_eps_t + (1 - alpha_target) * eps_orig

        return target_eps

    def forward(self, x_aug, alpha_target, grad_scale=1.0):
        # I think that input shape of x should be (1, 16, 3, 256, 256), for 16 frames     
        latent_z = self.prepare_latents(x_aug)

        grad_z = grad_scale * self.sds_grads(latent_z, alpha_target=alpha_target)

        sds_loss = SpecifyGradient.apply(latent_z, grad_z)

        return sds_loss

    def lora_step(self, x_aug):

        x = x_aug * 2. - 1. # encode rendered image
        with torch.cuda.amp.autocast():
            batch_size, num_frames, channels, height, width = x.shape
            x = x.reshape(batch_size * num_frames, channels, height, width) # I think that x shape should be (16, 3, 256, 256), for the VAE encoder
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample()) # init_latent_z shape is now [16, 4, 32, 32]
            frames, channel, h_, w_ = init_latent_z.shape
            init_latent_z = init_latent_z[None, :].reshape(batch_size, num_frames, channel, h_, w_).permute(0, 2, 1, 3, 4) # shape should be (1, 4, 16, 32, 32)
            
        latent_z = self.pipe.vae.config.scaling_factor * init_latent_z  # scaling_factor * init_latents

        timestep = torch.randint(
            low=self.cfg.sds_timestep_low,
            high=min(950, self.cfg.timesteps) - 1,  # avoid highest timestep | diffusion.timesteps=1000
            size=(latent_z.shape[0],),
            device=self.device, dtype=torch.long)

        noise_lora = torch.randn_like(latent_z)
        
        noised_latents = self.pipe_lora.scheduler.add_noise(latent_z, noise_lora, timestep)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            noise_pred = self.pipe_lora.unet(noised_latents, timestep, encoder_hidden_states=self.text_embeddings.chunk(2)[1]).sample.float()

        lora_loss = self.lora_loss_func(noise_pred, noise_lora)

        return lora_loss