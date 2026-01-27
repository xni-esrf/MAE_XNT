import torch
from torch import nn as nn
from timm.layers import to_3tuple
from timm.models.vision_transformer import Block
import warnings

from functools import partial
import numpy as np
import tools

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, transposed=False, use_bn=True):
        super(ConvBlock, self).__init__()

        # Choose between standard conv or transposed conv
        if transposed:
            conv_layer = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            conv_layer = nn.Conv3d(
                in_channels, out_channels, kernel_size=3, padding=1
            )

        # Define layers in sequence
        layers = []
        if use_bn:
            layers.append(nn.BatchNorm3d(in_channels))
        layers.append(conv_layer)
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



class Unetr_new(nn.Module):
    def __init__(self, minivol_size, in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=PatchEmbed3D, act_layer=nn.GELU):
        super().__init__()
        self.patch_size = 4
        self.in_chans = in_chans
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.minivol_size = minivol_size

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = embed_layer(patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim)

        grid_shape = [minivol_size//self.patch_size, minivol_size//self.patch_size, minivol_size//self.patch_size]
        self.pos_emb = build_3d_sincos_position_embedding(grid_shape, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])


    def build_decoder(self):

        self.top_conv_block1 = ConvBlock(1, self.embed_dim//64, transposed=False, use_bn=False)
        self.top_conv_block2 = ConvBlock(self.embed_dim//64, self.embed_dim//64, transposed=False, use_bn=True)
        
        self.norm_down1 = nn.LayerNorm(self.embed_dim)
        self.bottom_conv_block1 = ConvBlock(self.embed_dim, self.embed_dim//2, transposed=False, use_bn=False)
        self.bottom_conv_block2 = ConvBlock(self.embed_dim//2, self.embed_dim//16, transposed=True, use_bn=True)

        self.norm_mid1 = nn.LayerNorm(self.embed_dim)
        self.mid_conv_block1 = ConvBlock(self.embed_dim, self.embed_dim//2, transposed=False, use_bn=False)
        self.mid_conv_block2 = ConvBlock(self.embed_dim//2, self.embed_dim//16, transposed=True, use_bn=True)

        self.concat_conv_block1 = ConvBlock(self.embed_dim//8, self.embed_dim//8, transposed=False, use_bn=True)
        self.concat_conv_block2 = ConvBlock(self.embed_dim//8, self.embed_dim//8, transposed=False, use_bn=True)
        self.concat_conv_block3 = ConvBlock(self.embed_dim//8, self.embed_dim//64, transposed=True, use_bn=True)

        self.out_conv_block1 = ConvBlock(self.embed_dim//32, self.embed_dim//32, transposed=False, use_bn=True)
        self.out_conv_block2 = ConvBlock(self.embed_dim//32, self.embed_dim//32, transposed=False, use_bn=True)

        self.norm_last3 = nn.BatchNorm3d(self.embed_dim//32)
        self.conv_last3 = nn.Conv3d(self.embed_dim//32, 1, kernel_size=1, padding=0)


    def get_num_layers(self):
        return len(self.blocks)


    def forward(self, x):

        x_up = self.top_conv_block1(x)
        x_up = self.top_conv_block2(x_up)

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
        x = self.bottom_conv_block1(x)
        x = self.bottom_conv_block2(x)

        x_mid = self.norm_mid1(x_mid)
        x_mid = x_mid.permute((0,2,1))
        x_mid = x_mid.reshape(batch_size, embeddim, self.minivol_size//self.patch_size, self.minivol_size//self.patch_size ,self.minivol_size//self.patch_size)
        x_mid = self.mid_conv_block1(x_mid)
        x_mid = self.mid_conv_block2(x_mid)

        x = torch.concat([x, x_mid], 1)

        x = self.concat_conv_block1(x)
        x = self.concat_conv_block2(x)
        x = self.concat_conv_block3(x)

        x = torch.concat([x, x_up], 1)

        x = self.out_conv_block1(x)
        x = self.out_conv_block2(x)
        x = self.norm_last3(x)
        x = self.conv_last3(x)

        return x
    
class Unetr_new_ps8(nn.Module):
    def __init__(self, minivol_size, in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=PatchEmbed3D, act_layer=nn.GELU):
        super().__init__()
        self.patch_size = 8
        self.in_chans = in_chans
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.minivol_size = minivol_size

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = embed_layer(patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim)

        grid_shape = [minivol_size//self.patch_size, minivol_size//self.patch_size, minivol_size//self.patch_size]
        self.pos_emb = build_3d_sincos_position_embedding(grid_shape, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])


    def build_decoder(self):

        self.top_conv_block1 = ConvBlock(1, self.embed_dim//32, transposed=False, use_bn=False)
        self.top_conv_block2 = ConvBlock(self.embed_dim//32, self.embed_dim//32, transposed=False, use_bn=True)
        
        self.norm_down = nn.LayerNorm(self.embed_dim)
        self.bottom_conv_block1 = ConvBlock(self.embed_dim, self.embed_dim//2, transposed=False, use_bn=False)
        self.bottom_conv_block2 = ConvBlock(self.embed_dim//2, self.embed_dim//8, transposed=True, use_bn=True)

        self.norm_middeep = nn.LayerNorm(self.embed_dim)
        self.middeep_conv_block1 = ConvBlock(self.embed_dim, self.embed_dim//2, transposed=False, use_bn=False)
        self.middeep_conv_block2 = ConvBlock(self.embed_dim//2, self.embed_dim//8, transposed=True, use_bn=True)

        self.concatdeep_conv_block1 = ConvBlock(self.embed_dim//4, self.embed_dim//4, transposed=False, use_bn=True)
        self.concatdeep_conv_block2 = ConvBlock(self.embed_dim//4, self.embed_dim//4, transposed=False, use_bn=True)
        self.concatdeep_conv_block3 = ConvBlock(self.embed_dim//4, self.embed_dim//16, transposed=True, use_bn=True)

        self.norm_midshallow = nn.LayerNorm(self.embed_dim)
        self.midshallow_conv_block1 = ConvBlock(self.embed_dim, self.embed_dim, transposed=False, use_bn=False)
        self.midshallow_conv_block2 = ConvBlock(self.embed_dim, self.embed_dim//4, transposed=True, use_bn=True)
        self.midshallow_conv_block3 = ConvBlock(self.embed_dim//4, self.embed_dim//4, transposed=False, use_bn=True)
        self.midshallow_conv_block4 = ConvBlock(self.embed_dim//4, self.embed_dim//16, transposed=True, use_bn=True)

        self.concatshallow_conv_block1 = ConvBlock(self.embed_dim//8, self.embed_dim//8, transposed=False, use_bn=True)
        self.concatshallow_conv_block2 = ConvBlock(self.embed_dim//8, self.embed_dim//8, transposed=False, use_bn=True)
        self.concatshallow_conv_block3 = ConvBlock(self.embed_dim//8, self.embed_dim//32, transposed=True, use_bn=True)

        self.out_conv_block1 = ConvBlock(self.embed_dim//16, self.embed_dim//16, transposed=False, use_bn=True)
        self.out_conv_block2 = ConvBlock(self.embed_dim//16, self.embed_dim//16, transposed=False, use_bn=True)

        self.norm_last = nn.BatchNorm3d(self.embed_dim//16)
        self.conv_last = nn.Conv3d(self.embed_dim//16, 1, kernel_size=1, padding=0)


    def get_num_layers(self):
        return len(self.blocks)


    def forward(self, x):

        x_up = self.top_conv_block1(x)
        x_up = self.top_conv_block2(x_up)

        x = self.patch_embed(x)
        x = x + self.pos_emb.repeat([x.shape[0],1,1])
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i+1 == len(self.blocks)//3:
                x_midshallow = x.clone()
            elif i+1 == 2*len(self.blocks)//3:
                x_middeep = x.clone()

        batch_size, num_patches, embeddim =  x.shape

        x = self.norm_down(x)
        x = x.permute((0,2,1))
        x = x.reshape(batch_size, embeddim, self.minivol_size//self.patch_size, self.minivol_size//self.patch_size ,self.minivol_size//self.patch_size)
        x = self.bottom_conv_block1(x)
        x = self.bottom_conv_block2(x)

        x_middeep = self.norm_middeep(x_middeep)
        x_middeep = x_middeep.permute((0,2,1))
        x_middeep = x_middeep.reshape(batch_size, embeddim, self.minivol_size//self.patch_size, self.minivol_size//self.patch_size ,self.minivol_size//self.patch_size)
        x_middeep = self.middeep_conv_block1(x_middeep)
        x_middeep = self.middeep_conv_block2(x_middeep)

        x = torch.concat([x, x_middeep], 1)

        x = self.concatdeep_conv_block1(x)
        x = self.concatdeep_conv_block2(x)
        x = self.concatdeep_conv_block3(x)

        x_midshallow = self.norm_midshallow(x_midshallow)
        x_midshallow = x_midshallow.permute((0,2,1))
        x_midshallow = x_midshallow.reshape(batch_size, embeddim, self.minivol_size//self.patch_size, self.minivol_size//self.patch_size ,self.minivol_size//self.patch_size)
        x_midshallow = self.midshallow_conv_block1(x_midshallow)
        x_midshallow = self.midshallow_conv_block2(x_midshallow)
        x_midshallow = self.midshallow_conv_block3(x_midshallow)
        x_midshallow = self.midshallow_conv_block4(x_midshallow)

        x = torch.concat([x, x_midshallow], 1)

        x = self.concatshallow_conv_block1(x)
        x = self.concatshallow_conv_block2(x)
        x = self.concatshallow_conv_block3(x)

        x = torch.concat([x, x_up], 1)

        x = self.out_conv_block1(x)
        x = self.out_conv_block2(x)
        x = self.norm_last(x)
        x = self.conv_last(x)

        return x
