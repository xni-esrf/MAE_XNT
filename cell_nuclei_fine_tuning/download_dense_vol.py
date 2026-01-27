import webknossos as wk
import shutil
import os
import tifffile
import json

import webknossos as wk
from pathlib import Path
import numpy as np
import os
import tifffile
import h5py
import sys

from scipy.ndimage import zoom
from scipy.ndimage import convolve
from scipy.ndimage import median_filter


def gaussian_kernel_3d(size=3, sigma=1.0):
    """Generates a 3D Gaussian kernel."""
    # Create a grid of (x, y, z) coordinates
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    # Calculate the 3D Gaussian function
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))

    # Normalize the kernel so that the sum of all elements is 1
    kernel = kernel / np.sum(kernel)

    return kernel

# C432 665f23ca010000b8107bfd50
URL = "https://webknossos.crick.ac.uk/annotations/665f23ca010000b8107bfd50/"
annotation = wk.Annotation.download(URL)
data = wk.Annotation.open_as_remote_dataset(URL)
output_path = sys.argv[1]

bboxes = annotation.user_bounding_boxes

for bbox in bboxes:
    vol_pos = list(bbox.topleft)

    vol_dir = os.path.join(output_path, bbox.name)
    os.mkdir(vol_dir)

    raw_data = data.get_layer("color").get_mag(2).read(absolute_bounding_box=bbox)
    raw_data = np.squeeze(raw_data)
    raw_data = raw_data.astype(np.float32)

    # Interpolate to go back to the 100nm resolution
    raw_data = zoom(raw_data, zoom=(2, 2, 2), order=3)

    with h5py.File(os.path.join(vol_dir,"raw.h5"), 'w') as hf:
        hf.create_dataset(bbox.name, data=raw_data, chunks=(50,50,50))
    tifffile.imwrite(os.path.join(vol_dir, "raw.tiff"), raw_data)


    seg_data = data.get_layer("segmentation").get_mag(2).read(absolute_bounding_box=bbox)
    seg_data = np.squeeze(seg_data)
    seg_data = seg_data.astype(np.float32)

    # From intance segmentation to semantic segmentation
    seg_data[seg_data>=1]=1

    # Make annotations smoother
    seg_data = median_filter(seg_data, size=5)
    gauss_filer = gaussian_kernel_3d(size=3, sigma=1.0)
    seg_data = convolve(seg_data, gauss_filer, mode='reflect')
    seg_data = zoom(seg_data, zoom=(2, 2, 2), order=3)

    with h5py.File(os.path.join(vol_dir,"annotations.h5"), 'w') as hf:
        hf.create_dataset(bbox.name, data=seg_data, chunks=(50,50,50))
    tifffile.imwrite(os.path.join(vol_dir, "annotations.tiff"), seg_data)

    # Constitute volume metadata
    cur_metadata_dict = {}
    cur_metadata_dict["volume_path"] = os.path.join(output_path,bbox.name,"raw.h5")
    cur_metadata_dict["annot_path"] = os.path.join(output_path,bbox.name,"annotations.h5")
    cur_metadata_dict["volume.shape"] = raw_data.shape


    metadata_dir_path = "metadata/C432_dense"
    with open(os.path.join(metadata_dir_path,bbox.name+".json"), 'w') as json_file:
        json.dump(cur_metadata_dict, json_file)
    
