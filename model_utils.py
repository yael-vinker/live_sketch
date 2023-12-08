import torch

from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers.loaders import AttnProcsLayers

def configure_lora(unet, device):
    ### ref: https://github.com/huggingface/diffusers/blob/4f14b363297cf8deac3e88a3bf31f59880ac8a96/examples/dreambooth/train_dreambooth_lora.py#L833
    ### begin lora
    # Set correct lora layers
    unet_lora_attn_procs = {}
    unet_orig_attn_proc = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if (name.endswith("attn1.processor") or "temp_" in name) else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        elif name.startswith("transformer_in"):
            # unet_orig_attn_proc[name] = unet.attn_processors[name]
            unet_lora_attn_procs[name] = unet.attn_processors[name]
            continue

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = LoRAAttnProcessor

        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        ).to(device)


    unet.set_attn_processor(unet_lora_attn_procs)

    proc_layers = {name: processor for name, processor in unet.attn_processors.items() if not name.startswith("transformer_in")}

    unet_lora_layers = AttnProcsLayers(proc_layers)

    # unet_lora_layers._load_state_dict_pre_hooks.clear()
    # unet_lora_layers._state_dict_hooks.clear()

    unet.requires_grad_(False)
    for param in unet_lora_layers.parameters():
        param.requires_grad_(True)

    ### end lora
    return unet, unet_lora_layers
