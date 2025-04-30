import torch
import torch.nn as nn
from peft.tuners.lora import Linear as LoraLinear
from pytorch_lightning.callbacks import ModelCheckpoint

def extract_trainable_weights(model):
    """Extract trainable weights from the model."""
    trainable_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_weights[name] = param.data.clone()
    return trainable_weights

class LoRA_MappingNet(nn.Module):
    def __init__(self, in_dim, dim, genre_num, base_class, lora_cfg, new_genre_id):
        super().__init__()
        self.genre_num = genre_num
        self.new_genre_id = new_genre_id

        self.shared = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        )

        self.unshared = nn.ModuleDict()
        for gid in range(genre_num):
            if gid == new_genre_id:
                block = nn.Sequential(
                    LoraLinear(
                        base_layer=nn.Linear(dim, dim),
                        adapter_name="lora",
                        r=lora_cfg.r,
                        lora_alpha=lora_cfg.lora_alpha,
                        lora_dropout=lora_cfg.lora_dropout,
                    ),
                    nn.GELU(),
                    LoraLinear(
                        base_layer=nn.Linear(dim, dim),
                        adapter_name="lora",
                        r=lora_cfg.r,
                        lora_alpha=lora_cfg.lora_alpha,
                        lora_dropout=lora_cfg.lora_dropout,
                    ),
                )
                self.unshared[str(gid)] = block
            else:
                self.unshared[str(gid)] = base_class()

    def forward(self, x, genre):
        s = self.shared(x)
        outputs = []
        for i, gid in enumerate(genre.tolist()):
            outputs.append(self.unshared[str(gid)](s[i:i+1]))
        return torch.cat(outputs, dim=0)

# === Replace genre mappings with LoRA versions ===
def wrap_mapping_with_lora(mappingnet, in_dim, dim, new_genre_id, lora_cfg):
    base_class = lambda: nn.Sequential(*[layer for layer in mappingnet.unshared[0]])

    wrapped = LoRA_MappingNet(
        in_dim=in_dim,
        dim=dim,
        genre_num=new_genre_id + 1,
        base_class=base_class,
        lora_cfg=lora_cfg,
        new_genre_id=new_genre_id,
    )

    wrapped.shared.load_state_dict(mappingnet.shared.state_dict())

    for i in range(new_genre_id):
        wrapped.unshared[str(i)].load_state_dict(mappingnet.unshared[i].state_dict())

    return wrapped

def freeze_all_except_lora(model):
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name.lower()
        
def wrap_mapping_with_lora(mappingnet, in_dim, dim, new_genre_id, lora_cfg):
    base_class = lambda: nn.Sequential(*[layer for layer in mappingnet.unshared[0]])

    wrapped = LoRA_MappingNet(
        in_dim=in_dim,
        dim=dim,
        genre_num=new_genre_id + 1,
        base_class=base_class,
        lora_cfg=lora_cfg,
        new_genre_id=new_genre_id,
    )

    wrapped.shared.load_state_dict(mappingnet.shared.state_dict())

    for i in range(new_genre_id):
        wrapped.unshared[str(i)].load_state_dict(mappingnet.unshared[i].state_dict())

    return wrapped

def patch_attention_with_lora(attn_module, lora_cfg, name_prefix=""):
    """Replace q_proj and v_proj inside a MultiheadAttention block with LoRA versions."""
    embed_dim = attn_module.embed_dim

    # Wrap original weights
    q_proj = LoraLinear(
        base_layer=nn.Linear(embed_dim, embed_dim, bias=False),
        adapter_name=f"{name_prefix}_q_lora",
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
    )
    v_proj = LoraLinear(
        base_layer=nn.Linear(embed_dim, embed_dim, bias=False),
        adapter_name=f"{name_prefix}_v_lora",
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
    )

    # Load pretrained weights
    q_proj.base_layer.weight.data.copy_(attn_module.in_proj_weight[:embed_dim])
    v_proj.base_layer.weight.data.copy_(attn_module.in_proj_weight[2*embed_dim:])

    # Monkey patch (store the remaining proj parts manually or skip them if not needed)
    attn_module.q_proj = q_proj
    attn_module.v_proj = v_proj

    # Overwrite forward pass (if necessary), or replace the module entirely
    return attn_module


def patch_transformer_with_lora(model, lora_cfg):
    
    #number of layers to patch
    n_layers = lora_cfg.n_layers
    
    if n_layers and n_layers == 0:
        # No layers to patch
        return
        
    # Patch encoder
    if hasattr(model, "cond_encoder"):
        
        encoder_layers = model.cond_encoder
        
        if n_layers:
            layer_offset = len(encoder_layers) - n_layers
            encoder_layers = encoder_layers[-n_layers:]
        else:
            layer_offset = 0
        
        for i, encoder_layer in enumerate(encoder_layers):
            true_layer_idx = i + layer_offset
            patch_attention_with_lora(
                encoder_layer.self_attn,
                lora_cfg,
                name_prefix=f"encoder_layer_{true_layer_idx}"
            )
            
    # Patch decoder
    if hasattr(model, "seqTransDecoder"):
        
        decoder_layers = model.seqTransDecoder.stack
        
        if n_layers:        
            layer_offset = len(decoder_layers) - n_layers
            decoder_layers = decoder_layers[-n_layers:]
        else:
            layer_offset = 0

        for i, decoder_layer in enumerate(decoder_layers):
            true_layer_idx = i + layer_offset
            patch_attention_with_lora(
                decoder_layer.self_attn,
                lora_cfg,
                name_prefix=f"decoder_layer_{true_layer_idx}_self"
            )
            patch_attention_with_lora(
                decoder_layer.multihead_attn,
                lora_cfg,
                name_prefix=f"decoder_layer_{true_layer_idx}_cross"
            )

            
def patch_discriminator_tr_block_with_lora(dis_model, lora_cfg):
    
    n_layers = lora_cfg.n_layers
    
    if n_layers and n_layers == 0:
        # No layers to patch
        return
    
    dis_layers = dis_model.tr_block.layers
    
    if n_layers:
        layer_offset = len(dis_layers) - n_layers
        dis_layers = dis_layers[-n_layers:]
    else:
        layer_offset = 0
    
    for i, block in enumerate(dis_layers):
        
        true_layer_idx = i + layer_offset
        
        ln1, attn, ln2, ff = block  # unpack 4 submodules in the ModuleList

        # Patch to_q
        lora_q = LoraLinear(
            base_layer=nn.Linear(attn.to_q.in_features, attn.to_q.out_features, bias=False),
            adapter_name=f"dis_tr_block_{true_layer_idx}_q",
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
        )
        lora_q.base_layer.weight.data.copy_(attn.to_q.weight.data)
        attn.to_q = lora_q

        # Patch to_v
        lora_v = LoraLinear(
            base_layer=nn.Linear(attn.to_v.in_features, attn.to_v.out_features, bias=False),
            adapter_name=f"dis_tr_block_{true_layer_idx}_v",
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
        )
        lora_v.base_layer.weight.data.copy_(attn.to_v.weight.data)
        attn.to_v = lora_v
    

def patch_local(model, cfgs):
    """Patch the local model with LoRA layers."""

    NEW_GENRE_ID = cfgs.new_genre_id
    mapping_lora_cfg = cfgs.mapping_lora_cfg
    decoder_lora_cfg = cfgs.decoder_lora_cfg
    discriminator_lora_cfg = cfgs.discriminator_lora_cfg

    # Bypass EMA
    model.diffusion.master_model = model.diffusion.model
    model.diffusion.master_model_dis = model.dis_model

    # Patch mapping network
    
    # Decoder
    model.DanceDecoder.mapping = wrap_mapping_with_lora(
        mappingnet=model.DanceDecoder.mapping,
        in_dim=256,  # decoder
        dim=512,
        new_genre_id=NEW_GENRE_ID,
        lora_cfg=mapping_lora_cfg,
    )

    # Discriminator
    model.dis_model.mapping = wrap_mapping_with_lora(
        mappingnet=model.dis_model.mapping,
        in_dim=512,  # discriminator input
        dim=1,
        new_genre_id=16,
        lora_cfg=mapping_lora_cfg,
    )
    
    # Patch Transformer blocks (Decoder)
    patch_transformer_with_lora(model.DanceDecoder, decoder_lora_cfg)
    # Patch Transformer blocks (Discriminator)
    patch_discriminator_tr_block_with_lora(model.dis_model, discriminator_lora_cfg)
    
    # Freeze all except LoRA
    freeze_all_except_lora(model.DanceDecoder)
    freeze_all_except_lora(model.dis_model)
    
    print("LoRA for Local Module patched successfully.")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    
    return model
    
class LoRAModelCheckpoint(ModelCheckpoint):
    def _save_model(self, trainer, filepath):
        model = trainer.model

        # Extract LoRA weights only
        lora_weights = extract_lora_weights(model)

        # Save
        torch.save(lora_weights, filepath)