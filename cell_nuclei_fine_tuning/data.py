import json
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from random import randrange, uniform
import h5py
import math
import os
import itertools

class MultipleVolDataset():
    def __init__(self, metadata_path_list, minivol_size, contr_bright_factors=[1,0], crop_size=None) -> None:
        self.metadata_path_list = metadata_path_list
        self.raw_dict = {}
        self.annot_dict = {}

        self.weights = []

        # Build dictionnaries that associate metadata (keys) to training datasets (values)
        for i in range(len(self.metadata_path_list)):
            cur_metadata_path = self.metadata_path_list[i]
            with open(cur_metadata_path, 'r') as metadata_file :
                metadata_dict = json.load(metadata_file)

                cur_raw_file = h5py.File(metadata_dict["volume_path"], 'r', rdcc_nbytes=(2*1024*1024*1024), rdcc_w0=0.5, rdcc_nslots=196831)
                cur_raw = cur_raw_file[str(list(cur_raw_file.keys())[0])][:,:,:]

                cur_annot_file = h5py.File(metadata_dict["annot_path"], 'r', rdcc_nbytes=(2*1024*1024*1024), rdcc_w0=0.5, rdcc_nslots=196831)
                cur_annot = cur_annot_file[str(list(cur_annot_file.keys())[0])][:,:,:]
                
                # If set, get a central crop of the dataset 
                if crop_size:
                    start_z = np.random.randint(0, cur_raw.shape[0] - crop_size + 1)
                    start_x = np.random.randint(0, cur_raw.shape[1] - crop_size + 1)
                    start_y = np.random.randint(0, cur_raw.shape[2] - crop_size + 1)
                    cur_raw = cur_raw[start_z:start_z+crop_size, start_x:start_x+crop_size, start_y:start_y+crop_size]
                    cur_annot = cur_annot[start_z:start_z+crop_size, start_x:start_x+crop_size, start_y:start_y+crop_size]

                ds_mean = cur_raw.mean()
                ds_std = cur_raw.std()
                cur_raw = (cur_raw - ds_mean)/ds_std

                # Weight training dataset depending on their size, i.e., big datasets are more sampled than small ones
                self.weights.append(cur_raw.shape[0]*cur_raw.shape[1]*cur_raw.shape[2])
        
                self.raw_dict[cur_metadata_path] = torch.tensor(cur_raw)
                self.annot_dict[cur_metadata_path] = torch.tensor(cur_annot)

        self.minivol_size = minivol_size
        self.contr_fact = contr_bright_factors[0]
        self.bright_fact = contr_bright_factors[1]

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

    def geom_transform(self, raw_minivol, annot_minivol):
        k = torch.randint(0, len(self.SYM_GROUP), (1,)).item()
        perm, flips = self.SYM_GROUP[k]
        return self.apply_symmetry(raw_minivol, perm, flips), self.apply_symmetry(annot_minivol, perm, flips)

    def __getitem__(self, idx):
        # pick dataset from indice and attributes of this dataset(from metadata file)
        picked_ds_metadata_path = self.metadata_path_list[idx]
        picked_raw_array = self.raw_dict[picked_ds_metadata_path]
        picked_annot_array = self.annot_dict[picked_ds_metadata_path]

        # get independant random coordinates of the minivol to be extracted
        start_z = randrange(picked_raw_array.shape[0]-self.minivol_size)
        start_x = randrange(picked_raw_array.shape[1]-self.minivol_size)
        start_y = randrange(picked_raw_array.shape[2]-self.minivol_size)

        # Extract a minivol from the picked dataset respecting the corrdinate defined above
        extracted_minivol = picked_raw_array[start_z:start_z+self.minivol_size, start_x:start_x + self.minivol_size, start_y:start_y + self.minivol_size]
        extracted_annot = picked_annot_array[start_z:start_z+self.minivol_size, start_x:start_x + self.minivol_size, start_y:start_y + self.minivol_size]
        extracted_minivol, extracted_annot = self.geom_transform(extracted_minivol, extracted_annot)
        extracted_minivol, extracted_annot = torch.unsqueeze(extracted_minivol,0), torch.unsqueeze(extracted_annot,0)
        
        if self.contr_fact != 1 or self.bright_fact !=0:
            cur_contr = uniform(self.contr_fact,1)
            cur_bright = uniform(-self.bright_fact,self.bright_fact)
            extracted_minivol = extracted_minivol*cur_contr+cur_bright

        return extracted_minivol, extracted_annot



class EvalDatasetOverlap():
    def __init__(self, ds_metadata_dict, minivol_size, normalize_data, reflect_padding=True, normalize_crop=None) -> None:
        ds_file = h5py.File(ds_metadata_dict["volume_path"], 'r', rdcc_nbytes=(2*1024*1024*1024), rdcc_w0=0.5, rdcc_nslots=196831)
        mem_array = ds_file[str(list(ds_file.keys())[0])]

        self.vol = torch.tensor(mem_array[:,:,:])

        #If set, pad volumes to handle edges
        if reflect_padding:
            self.z_pad_size = ((minivol_size//2-(mem_array.shape[0]%(minivol_size//2)))%(minivol_size//2))//2 + minivol_size//2
            self.x_pad_size = ((minivol_size//2-(mem_array.shape[1]%(minivol_size//2)))%(minivol_size//2))//2 + minivol_size//2
            self.y_pad_size = ((minivol_size//2-(mem_array.shape[2]%(minivol_size//2)))%(minivol_size//2))//2 + minivol_size//2
            self.pad_size = (self.z_pad_size, self.x_pad_size, self.y_pad_size)
            self.vol = torch.unsqueeze(self.vol,0)
            self.vol = torch.nn.functional.pad(self.vol, (self.y_pad_size, self.y_pad_size, self.x_pad_size, self.x_pad_size, self.z_pad_size , self.z_pad_size), mode="reflect")
            self.vol = torch.squeeze(self.vol)
        else :
            self.pad_size = (0,0,0)

        # Get index ranges for every dim
        self.minivol_size = minivol_size
        self.volume_shape = self.vol.shape
        self.nb_minivol_z, self.nb_minivol_x, self.nb_minivol_y= (self.volume_shape[0]//(minivol_size//2))-1, (self.volume_shape[1]//(minivol_size//2))-1, (self.volume_shape[2]//(minivol_size//2))-1
        self.total_nb_minivol = self.nb_minivol_z*self.nb_minivol_x*self.nb_minivol_y

        if normalize_data:
            # Normalize according to the volume statistics
            if normalize_crop==None:
                ds_mean = self.vol.mean()
                ds_std = self.vol.std()
                self.vol = (self.vol-ds_mean)/ds_std
            # Normalize based on a crop statistics
            else:
                offset = [(self.vol.shape[0] - normalize_crop[0])//2, (self.vol.shape[1] - normalize_crop[1])//2, (self.vol.shape[2] - normalize_crop[2])//2]
                ds_mean = self.vol[offset[0]:offset[0]+normalize_crop[0],offset[1]:offset[1]+normalize_crop[1],offset[2]:offset[2]+normalize_crop[2]].mean()
                ds_std = self.vol[offset[0]:offset[0]+normalize_crop[0],offset[1]:offset[1]+normalize_crop[1],offset[2]:offset[2]+normalize_crop[2]].std()
                self.vol = (self.vol-ds_mean)/ds_std

    def __len__(self):
        return self.total_nb_minivol

    def __getitem__(self, idx):

        if idx == self.total_nb_minivol:
            raise StopIteration

        # get minivol index for next minivol for every dim
        idx_z, idx_x, idx_y = idx%self.nb_minivol_z, (idx//self.nb_minivol_z)%self.nb_minivol_x, (idx//self.nb_minivol_z)//self.nb_minivol_x

        start_z, start_x, start_y =  idx_z*(self.minivol_size//2), idx_x*(self.minivol_size//2), idx_y*(self.minivol_size//2)

        # Extract a minivol from the picked dataset respecting the corrdinates defined above
        extracted_minivol = self.vol[start_z:start_z+self.minivol_size, start_x:start_x + self.minivol_size, start_y:start_y + self.minivol_size]
        extracted_minivol = torch.unsqueeze(extracted_minivol,0)

        return extracted_minivol, (start_z, start_x, start_y)

# Buffer where processed minivolumes are aggregated
class DestVolBufferOverlap():
    def __init__(self, volume_shape, minivol_size):
        self.minivol_size = minivol_size
        self.buffer_vol = torch.squeeze(torch.zeros(volume_shape))
        self.volume_shape = volume_shape

        # Build a Hann Window for overlapping weighting
        hann_z = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.minivol_size) / (self.minivol_size - 1)))
        hann_x = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.minivol_size) / (self.minivol_size - 1)))
        hann_y = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.minivol_size) / (self.minivol_size - 1)))
        self.hann_window = torch.tensor(np.outer(hann_x, hann_y)[:, :, np.newaxis] * hann_z[np.newaxis, np.newaxis, :])


    def add_batch(self, batch, batch_coordinates):
        # include minivol into the volume to fill
        for i in range(batch.shape[0]):
            minivol = batch[i,:,:,:,:]
            minivol = torch.squeeze(minivol)

            minivol_coordinates = [batch_coordinates[0][i],batch_coordinates[1][i], batch_coordinates[2][i]]

            minivol = minivol*self.hann_window

            self.buffer_vol[minivol_coordinates[0]:minivol_coordinates[0]+self.minivol_size, minivol_coordinates[1]:minivol_coordinates[1]+self.minivol_size, minivol_coordinates[2]:minivol_coordinates[2]+self.minivol_size] = minivol + self.buffer_vol[minivol_coordinates[0]:minivol_coordinates[0]+self.minivol_size, minivol_coordinates[1]:minivol_coordinates[1]+self.minivol_size, minivol_coordinates[2]:minivol_coordinates[2]+self.minivol_size]

    def get_vol(self, pad_size):
        # remove the pad added in the EvalDatasetOverlap class
        return self.buffer_vol[pad_size[0]:self.buffer_vol.shape[0]-pad_size[0], pad_size[1]:self.buffer_vol.shape[1]-pad_size[1], pad_size[2]:self.buffer_vol.shape[2]-pad_size[2]]
        


# Get the annotation volume associated to a metadata file
def get_annotation_array(metadata_dict):
    ds_file = h5py.File(metadata_dict["annot_path"], 'r', rdcc_nbytes=(2*1024*1024*1024), rdcc_w0=0.5, rdcc_nslots=196831)
    mem_array = ds_file[str(list(ds_file.keys())[0])]

    annotation_vol = torch.tensor(mem_array[:,:,:])

    return annotation_vol