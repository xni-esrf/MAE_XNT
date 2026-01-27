import numpy as np
from torch.utils.data import Dataset
import torch
from scipy import ndimage as ndi
import random
import os
import h5py
import torchvision

def Affinity(seg, neighb):
    # Conversion of instance segmentation labels into affinities
    n_edges = neighb.shape[0]
    n_vox = seg.shape[2]
    aff_matrix = torch.zeros((n_edges, n_vox, n_vox, n_vox), dtype=seg.dtype)
    for e in range(n_edges):
        # Affinity is 1 if the two neighboring voxels are not background and belong to the same axon
        aff_matrix[e, max(0, -neighb[e,0]):n_vox, max(0, -neighb[e,1]):n_vox, max(0, -neighb[e,2]):n_vox] = \
        (seg[max(0, -neighb[e,0]):n_vox, max(0, -neighb[e,1]):n_vox, max(0, -neighb[e,2]):n_vox] == seg[0:min(n_vox, n_vox+neighb[e,0]), 0:min(n_vox, n_vox+neighb[e,1]), 0:min(n_vox, n_vox+neighb[e,2])]) * \
        (seg[max(0, -neighb[e,0]):n_vox, max(0, -neighb[e,1]):n_vox, max(0, -neighb[e,2]):n_vox] > 0) * (seg[0:min(n_vox, n_vox+neighb[e,0]), 0:min(n_vox, n_vox+neighb[e,1]), 0:min(n_vox, n_vox+neighb[e,2])] > 0)
    return aff_matrix


def WeightingAffinity(aff, mask):
    # Class balancing function, weights are computed for each affinity "class" 
    counter = aff*mask
    fracs_1 = torch.clip((torch.sum(counter, (0,2,3,4))/torch.sum(mask, (0,2,3,4))), 0.05, 0.95).view(1,aff.shape[1],1,1,1)
    fracs_0 = 1 - fracs_1
    return 1/(2*fracs_0), 1/(2*fracs_1)

class RandomCrop3d(torch.nn.Module):
    # Utility class to crop a random 3D volume in the original volume
    @staticmethod
    def get_loc(img, vol_size):
        d, h, w = img.shape
        # To avoid artifacts caused by elastic deformation on the volume borders
        td, th, tw = [vol_size]*3
        if w == tw and h == th and d == td:
            return 0, 0, 0, h, w, d
        
        x = torch.randint(0, w - tw + 1, size=(1,)).item()
        y = torch.randint(0, h - th + 1, size=(1,)).item()
        z = torch.randint(0, d - td + 1, size=(1,)).item()
        return z, y, x, td, th, tw
    
    def __init__(self, vol_size):
        super().__init__()   
        self.size = vol_size

    def forward(self, img, label, mask, label_roi):
        z, y, x, td, th, tw = self.get_loc(img, self.size)
        while torch.sum(mask[z:z+td, y:y+th, x:x+tw])/torch.numel(mask[z:z+td, y:y+th, x:x+tw]) < 0.05: # Resample if labeled proportion of the cropped volume is inferior to 5%
            z, y, x, td, th, tw = self.get_loc(img, self.size)

        return img[z:z+td, y:y+th, x:x+tw], label[z:z+td, y:y+th, x:x+tw], mask[z:z+td, y:y+th, x:x+tw], label_roi[z:z+td, y:y+th, x:x+tw]


class RandomRot(torch.nn.Module):
    # Randomly sample one the 24 possible rotations of the volume
    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
     
    def forward(self, img, label, mask, label_roi):
        face, rot, flip = self.get_params()

        img = self.h_flip(self.rotate_cube(img, face, rot), flip)
        label = self.h_flip(self.rotate_cube(label, face, rot), flip)
        mask = self.h_flip(self.rotate_cube(mask, face, rot), flip)
        label_roi = self.h_flip(self.rotate_cube(label_roi, face, rot), flip)

        return img, label, mask, label_roi

    @staticmethod
    def rotate_cube(m, face, rot):
        # Bring the chosen face to the front 
        if face == 1:  
            m = torch.permute(m, (1, 0, 2))  
        elif face == 2:  
            m = torch.permute(m, (2, 1, 0))  
        elif face == 3:  
            m = torch.permute(m, (1, 0, 2))
            m = torch.flip(m, [1])  
        elif face == 4:  
            m = torch.permute(m, (2, 1, 0))
            m = torch.flip(m, [2])  
        elif face == 5:  
            m = torch.flip(m, [0])  

        # Rotate around the new front-facing axis
        m = torch.rot90(m, rot, (1, 2))
        return m
    
    @staticmethod
    def h_flip(m, flip):  # Horizontal flip
        if flip == 1:
            m = torchvision.transforms.functional.hflip(m)
        return m

    @staticmethod
    def get_params():
        face = torch.randint(0, 6, (1,)).item()  # 6 face choices
        rot = torch.randint(0, 4, (1,)).item()  # 4 in-plane rotations
        flip = torch.randint(0,2, (1,)).item() # Horizontal flip
        return face, rot, flip


class RandomNoise(torch.nn.Module):
    # Random gaussian noise. Gaussian standard deviation is uniformly sampled between 0 and std
    def __init__(self, std):
        super().__init__()   
        self.std = std

    def forward(self, img):
        shape = img.shape
        noise = self.get_params(shape, self.std)
        return img + noise
    
    @staticmethod
    def get_params(shape, std):
        std_sampled = torch.rand((1,))*std
        return torch.randn(*shape)*std_sampled


class RandomShift(torch.nn.Module):
    # Randomly shift and scale the input volume gray intensity
    def __init__(self, shift_low, shift_high, scale_low, scale_high):
        super().__init__()   
        self.shift_low = shift_low
        self.shift_high = shift_high
        self.scale_low = scale_low
        self.scale_high = scale_high

    def forward(self, img):
        shift, scale = self.get_params(self.shift_low, self.shift_high, self.scale_low, self.scale_high)
        return img*scale + shift
    
    @staticmethod
    def get_params(shift_low, shift_high, scale_low, scale_high):
        shift = (torch.rand((1,))*(shift_high-shift_low) - shift_high)
        scale = (torch.rand((1,))*(scale_high-scale_low) + scale_low)
        return shift, scale


class PatchedData(Dataset):

    def __init__(self, label_file, img_file, pad_size, volume_size, minivol_size, neighb, transform):
        self.neighb = neighb
        self.crop = transform[0]
        self.rot = transform[1]
        self.noise = transform[2]
        self.shift = transform[3]
        self.minivol_size = minivol_size

        offset = (200 - volume_size)//2
        start_center_crop = 500 + offset

        img = torch.tensor(h5py.File(img_file)['volumes']['raw'][start_center_crop-pad_size:start_center_crop+volume_size+pad_size,start_center_crop-pad_size:start_center_crop+volume_size+pad_size,start_center_crop-pad_size:start_center_crop+volume_size+pad_size]).type(torch.float32)
        self.img = (img - img.mean())/img.std()

        label = torch.tensor(h5py.File(label_file)['volumes']['labels'][offset:volume_size+offset, offset:volume_size+offset, offset:volume_size+offset]).type(torch.float32)
        mask = torch.ones_like(label)
        mask[label == 0] = 0
        # Erosion
        label = label.numpy()
        foreground = np.zeros(shape=label.shape, dtype=bool)
        for id in np.unique(label):
            if id == 0:
                continue
            label_mask = label == id
            eroded_label_mask = ndi.binary_erosion(
                label_mask, iterations=1, border_value=1
            )
            foreground = np.logical_or(eroded_label_mask, foreground)

        background = np.logical_not(foreground)
        label[background] = 0
        label = torch.tensor(label)

        # Padding 0 to align with raw input
        self.label = torch.nn.functional.pad(label, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).type(torch.float32)
        self.mask = torch.nn.functional.pad(mask, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)

        # Label ROI to mask padded data
        self.label_roi = torch.zeros_like(self.mask)
        self.label_roi[pad_size:volume_size + pad_size, pad_size:volume_size + pad_size, pad_size:volume_size + pad_size] = 1


    def __getitem__(self, idx):
        minivol, minivol_label, minivol_mask, minivol_roi = self.rot(*self.crop(self.img, self.label, self.mask, self.label_roi))

        minivol = self.noise(self.shift(minivol))

        minivol_aff = Affinity(minivol_label, self.neighb)

        # Mask affinities between unlabelled voxels only
        minivol_mask = 1 - Affinity(1 - minivol_mask, self.neighb)

        minivol_roi = Affinity(minivol_roi,  self.neighb)
        # Mask affinities between voxel outside the labelled area and the ones inside the labelled area
        minivol_mask *= minivol_roi

        return minivol.unsqueeze(0), minivol_aff, minivol_mask
    


def set_seeds(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(random_seed)