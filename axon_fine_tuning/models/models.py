import torch
from torch import nn as nn
from timm.models.layers import to_3tuple
from timm.models.vision_transformer import Block
import warnings

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



class Unetr_light_finetune(nn.Module):
    def __init__(self, minivol_size, patch_size, in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=PatchEmbed3D, act_layer=nn.GELU):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.minivol_size = minivol_size

        norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.patch_embed = embed_layer(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        grid_shape = [minivol_size//patch_size, minivol_size//patch_size, minivol_size//patch_size]
        self.pos_emb = build_3d_sincos_position_embedding(grid_shape, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])


    def build_decoder(self):
        self.conv_up1 = nn.Conv3d(1, self.embed_dim//64, kernel_size=3, padding=1)
        self.relu_up1 = nn.ReLU()
        self.norm_up1 = nn.BatchNorm3d(self.embed_dim//64)

        self.conv_up2 = nn.Conv3d(self.embed_dim//64, self.embed_dim//64, kernel_size=3, padding=1)
        self.relu_up2 = nn.ReLU()

        self.norm_down1 = nn.LayerNorm(self.embed_dim)
        self.conv_down = nn.Conv3d(self.embed_dim, self.embed_dim//2, kernel_size=3, padding=1)
        self.relu_down1 = nn.ReLU()

        self.norm_down2 = nn.BatchNorm3d(self.embed_dim//2)
        self.upconv_down = nn.ConvTranspose3d(self.embed_dim//2,self.embed_dim//16, kernel_size=2, stride=2)
        self.relu_down2 = nn.ReLU()

        self.norm_mid1 = nn.LayerNorm(self.embed_dim)
        self.conv_mid = nn.Conv3d(self.embed_dim, self.embed_dim//2, kernel_size=3, padding=1)
        self.relu_mid1 = nn.ReLU()

        self.norm_mid2 = nn.BatchNorm3d(self.embed_dim//2)
        self.upconv_mid = nn.ConvTranspose3d(self.embed_dim//2,self.embed_dim//16, kernel_size=2, stride=2)
        self.relu_mid2 = nn.ReLU()

        self.norm_concat1 = nn.BatchNorm3d(self.embed_dim//8)
        self.conv_concat1 = nn.Conv3d(self.embed_dim//8, self.embed_dim//8, kernel_size=3, padding=1)
        self.relu_concat1 = nn.ReLU()

        self.norm_concat2 = nn.BatchNorm3d(self.embed_dim//8)
        self.conv_concat2 = nn.Conv3d(self.embed_dim//8, self.embed_dim//8, kernel_size=3, padding=1)
        self.relu_concat2 = nn.ReLU()

        self.norm_concat3 = nn.BatchNorm3d(self.embed_dim//8)
        self.upconv_concat = nn.ConvTranspose3d(self.embed_dim//8,self.embed_dim//64, kernel_size=2, stride=2)
        self.relu_concat3 = nn.ReLU()

        self.norm_last1 = nn.BatchNorm3d(self.embed_dim//32)
        self.conv_last1 = nn.Conv3d(self.embed_dim//32, self.embed_dim//32, kernel_size=3, padding=1)
        self.relu_last1 = nn.ReLU()

        self.norm_last2 = nn.BatchNorm3d(self.embed_dim//32)
        self.conv_last2 = nn.Conv3d(self.embed_dim//32, self.embed_dim//32, kernel_size=3, padding=1)
        self.relu_last2 = nn.ReLU()

        self.norm_last3 = nn.BatchNorm3d(self.embed_dim//32)
        self.conv_last3 = nn.Conv3d(self.embed_dim//32, 12, kernel_size=1, padding=0) # 12 is the number of neighboring affinities
        self.last_actiavation = nn.Sigmoid()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)) & (m != self.patch_embed.proj): # Don't reinitialize trained weights:
            nn.init.xavier_normal_(m.weight)
            if getattr(m, 'bias') is not None:
                nn.init.constant_(m.bias, 0)


    def get_num_layers(self):
        return len(self.blocks)


    def forward(self, x):

        x_up = self.conv_up1(x)
        x_up = self.relu_up1(x_up)
        x_up = self.norm_up1(x_up)

        x_up = self.conv_up2(x_up)
        x_up = self.relu_up2(x_up)


        x = self.patch_embed(x)

        x = x + self.pos_emb.repeat([x.shape[0],1,1])

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i+1 == len(self.blocks)//2:
                x_mid = x.clone()

        batch_size, num_patches, embeddim =  x.shape

        x = self.norm_down1(x)
        x = x.permute((0,2,1))
        x = x.reshape(batch_size, embeddim, self.minivol_size//self.patch_size, self.minivol_size//self.patch_size ,self.minivol_size//self.patch_size)
        x = self.conv_down(x)
        x = self.relu_down1(x)

        x = self.norm_down2(x)
        x = self.upconv_down(x)
        x = self.relu_down2(x)


        x_mid = self.norm_mid1(x_mid)
        x_mid = x_mid.permute((0,2,1))
        x_mid = x_mid.reshape(batch_size, embeddim, self.minivol_size//self.patch_size, self.minivol_size//self.patch_size ,self.minivol_size//self.patch_size)
        x_mid = self.conv_mid(x_mid)
        x_mid = self.relu_mid1(x_mid)

        x_mid = self.norm_mid2(x_mid)
        x_mid = self.upconv_mid(x_mid)
        x_mid = self.relu_mid2(x_mid)

        x = torch.concat([x, x_mid], 1)

        x = self.norm_concat1(x)
        x = self.conv_concat1(x)
        x = self.relu_concat1(x)

        x = self.norm_concat2(x)
        x = self.conv_concat2(x)
        x = self.relu_concat2(x)

        x = self.norm_concat3(x)
        x = self.upconv_concat(x)
        x = self.relu_concat3(x)

        x = torch.concat([x, x_up], 1)

        x = self.norm_last1(x)
        x = self.conv_last1(x)
        x = self.relu_last1(x)

        x = self.norm_last2(x)
        x = self.conv_last2(x)
        x = self.relu_last2(x)

        x = self.norm_last3(x)
        x = self.conv_last3(x)
        x = self.last_actiavation(x)

        return x





