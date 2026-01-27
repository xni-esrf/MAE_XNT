import json
import numpy as np
import torch
import torchvision
from random import randrange
import h5py
import math
import zarr
import itertools

class MultipleVolDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path_list, minivol_size) -> None:
        self.metadata_path_list = metadata_path_list
        self.z_file = {}
        self.crop_mean = {}
        self.crop_std = {}
        self.shape = {}
        self.weights = []

        # Load from metadata dataset volume info
        for i in range(len(self.metadata_path_list)):
            cur_metadata_path = self.metadata_path_list[i]
            with open(cur_metadata_path, 'r') as metadata_file:
                metadata_dict = json.load(metadata_file)

                zarr_path = metadata_dict["volume_path"]
                self.z_file[cur_metadata_path] =  zarr.open(zarr_path, mode='r')

                self.crop_mean[cur_metadata_path] = torch.tensor(
                    float(metadata_dict["mean_crop"]), dtype=torch.float32
                )
                self.crop_std[cur_metadata_path] = torch.tensor(
                    float(metadata_dict["std_crop"]), dtype=torch.float32
                )

                # weight proportional to volume size
                self.shape[cur_metadata_path] = self.z_file[cur_metadata_path].shape
                self.weights.append( self.shape[cur_metadata_path][0] *  self.shape[cur_metadata_path][1] *  self.shape[cur_metadata_path][2])

        self.minivol_size = minivol_size
        
        self.SYM_GROUP = []
        for perm in itertools.permutations([0,1,2]):
            for flip_pattern in itertools.product([0,1], repeat=3):
                flips = [ax for ax, f in enumerate(flip_pattern) if f == 1]
                self.SYM_GROUP.append((perm, flips))

    def apply_symmetry(self, x, perm, flips):
        """
        Apply a cube symmetry defined by permuting axes and flipping.
        - perm: tuple of (0,1,2) permutation
        - flips: list of axes to flip
        """
        x = x.permute(perm)  # reorder axes
        for ax in flips:
            x = x.flip(ax)
        return x

    def geom_transform(self, minivol):
        k = torch.randint(0, len(self.SYM_GROUP), (1,)).item()
        perm, flips = self.SYM_GROUP[k]
        return self.apply_symmetry(minivol, perm, flips)

    def __getitem__(self, idx):
        # pick dataset corresponding to the index
        picked_ds_metadata_path = self.metadata_path_list[idx]


        # pick random coords
        start_z = torch.randint(0, self.shape[picked_ds_metadata_path][0] - self.minivol_size, (1,)).item()
        start_x = torch.randint(0, self.shape[picked_ds_metadata_path][1] - self.minivol_size, (1,)).item()
        start_y = torch.randint(0, self.shape[picked_ds_metadata_path][2] - self.minivol_size, (1,)).item()

        # extract minivol
        extracted_minivol = self.z_file[picked_ds_metadata_path][
            start_z:start_z + self.minivol_size,
            start_x:start_x + self.minivol_size,
            start_y:start_y + self.minivol_size
        ]

        extracted_minivol = torch.tensor(extracted_minivol, dtype=torch.float32)
        extracted_minivol = self.geom_transform(extracted_minivol)
        extracted_minivol = torch.unsqueeze(extracted_minivol, 0)

        extracted_minivol = (
            (extracted_minivol - self.crop_mean[picked_ds_metadata_path]) /
            self.crop_std[picked_ds_metadata_path]
        )

        return extracted_minivol

    def __len__(self):
        return len(self.metadata_path_list)

