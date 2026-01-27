import torch
from torch import nn as nn
from timm.models.vision_transformer import Block
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from functools import partial
import numpy as np

def build_3d_sincos_position_embedding(grid_shape, embed_dim, temperature=10000.):
    h, w, d = grid_shape
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)], dim=1)[None, :, :]

    pos_embed = torch.nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


class PatchEmbed3D(nn.Module):
    """ Input shape should be [Batch_size, in_channel, minivol_width, minivol_height, minivol_depth]
    Patchify minivols, format it to fit the transformer input format, and add the positionnal encoding
    """

    def __init__(self, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x -> B, MV_C, MV_W, MV_H, MV_D
        x = self.proj(x)
        # x -> B, enc_dim, MV_W/P_W, MV_H/P_H, MV_D/P_D
        # equivalent x -> B, enc_dim, G_W, G_H, G_D
        x = x.flatten(2)
        #x -> B, enc_dim, G_W*G_H*G_D
        x = x.transpose(1,2)
        #x -> B, G_W*G_H*G_D, enc_dim
        x = self.norm(x)
        return x


class MAEViTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    Modified from timm implementation
    """
    def __init__(self, minivol_size, patch_size, in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=PatchEmbed3D, act_layer=nn.GELU):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim  # num_features for consistency with other models

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = embed_layer(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        grid_shape = [minivol_size//patch_size, minivol_size//patch_size, minivol_size//patch_size]
        self.pos_emb = build_3d_sincos_position_embedding(grid_shape, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


    def get_num_layers(self):
        return len(self.blocks)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_kept, mask, ids_restore

    def forward(self, x, mask_ratio):
        x = self.patch_embed(x)

        x = x + self.pos_emb.repeat([x.shape[0],1,1])

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x, mask, ids_restore

class MAEViTDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    Modified from timm implementation
    """
    def __init__(self, enc_embed_dim, embed_dim, patch_size, minivol_size, in_chans=1, depth=8,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 attn_drop_rate=0., drop_path_rate=0., 
                act_layer=nn.GELU):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        norm_layer=partial(nn.LayerNorm, eps=1e-6)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.embed_layer = nn.Linear(enc_embed_dim, embed_dim, bias=True)

        grid_shape = [minivol_size//patch_size, minivol_size//patch_size, minivol_size//patch_size]
        self.pos_emb = build_3d_sincos_position_embedding(grid_shape, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, patch_size*patch_size*patch_size*in_chans)

    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, x, ids_restore):
        x = self.embed_layer(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, self.embed_dim))  # unshuffle

        x = x + self.pos_emb.repeat([x.shape[0],1,1])

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x)
        return x
