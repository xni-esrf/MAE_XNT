
import numpy as np
import torch
import os
import tifffile
from pathlib import Path

def minivols_to_patches(x, patch_size):
    # patchify input, [B,C,H,W,D] --> [B,C,gh,ph,gw,pw,gd,pd] --> [B,gh*gw*gd,ph*pw*pd*C]
    B, C, H, W, D = x.shape
    grid_shape = (H // patch_size, W // patch_size, D // patch_size)

    x = x.reshape(B, C, grid_shape[0], patch_size, grid_shape[1], patch_size, grid_shape[2], patch_size) # [B,C,gh,ph,gw,pw,gd,pd]
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, np.prod(grid_shape), patch_size*patch_size*patch_size * C) # [B,gh*gw*gd,ph*pw*pd*C]

    return x



def patches_to_minivols(patches, patch_size, minivol_size, nb_img_channel=1):
    B, L, C = patches.shape
    grid_shape = (minivol_size // patch_size, minivol_size // patch_size, minivol_size // patch_size)
    patch_shape = [patch_size,patch_size,patch_size]

    patches = patches.reshape(B, *grid_shape, *patch_shape, nb_img_channel)
    # restore image structure
    minivols = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, nb_img_channel,
                                                            grid_shape[0] * patch_size, 
                                                            grid_shape[1] * patch_size, 
                                                            grid_shape[2] * patch_size)
    return minivols

# Compute the dice loss assuming that given predictions and annotations are binary arrays.
def dice_loss_eval(predictions, annotations, smooth=0.0001):
    overlap = predictions*annotations
    overlap = overlap.sum()

    dice_coeff = (2.*overlap+smooth)/(predictions.sum()+annotations.sum()+smooth)
    dice_loss = 1-dice_coeff
    return dice_loss



def save_output(final_vol, print_directions, output_dir):
    output_dir = Path(output_dir)
    for i in range(print_directions):
        Path(output_dir / f"{i}").mkdir(exist_ok=True)
        for j in range(final_vol.shape[0]):
            if i == 0:
                img_np = final_vol[j, :, :]
            if i == 1:
                img_np = final_vol[:, j, :]
            if i == 2:
                img_np = final_vol[:, :, j]
            img_dir = str(output_dir / f"{i}/output_{j:05d}.tif")
            tifffile.imsave(img_dir, img_np)