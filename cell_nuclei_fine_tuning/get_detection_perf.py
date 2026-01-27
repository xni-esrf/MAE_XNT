import torch
import numpy as np
import os
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import json
import tifffile
from torch.utils.data import  DataLoader
import warnings
from random import shuffle
import tifffile
import h5py
import random
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import binary_dilation
from skimage.measure import marching_cubes


import models
import data
import tools


# Parse input arguments
parse = ArgumentParser(description="")
parse.add_argument('dataset_metadata_path', help='Path to the json file, which is used to load information about the processed dataset')
parse.add_argument('checkpoint_path', help='Path of the checkpoint to be loaded')
parse.add_argument('eval_set', help="Use test, validation_early_stop or validation_sphericity to evaluate performances")
parse.add_argument('--batch_size', default=None, type=int, help="The number of minivol per batch, if unspecified, us the same as training batch size")
parse.add_argument('--no_normalization', action='store_false', dest="normalize_data", default=True, help="Normalize training data")
parse.add_argument('--num_workers', default=64, type=int, help="Number of workers used in the Dataloader")
parse.add_argument('--normalize_crop', default=[500,500,500], nargs=3, type=int, help="Used when test set border are not filled with biological features")
parse.add_argument('--volume_threshold', default=None, type=int, help="Filter cell nuclei that have too few voxels")
parse.add_argument('--mean_dist_seed_watershed', default=20, type=int, help="Minimal distance between seed used for watershed")
parse.add_argument('--sphericity_threshold', default=0.6, type=float, help="Filter cell nuclei that are not spherical enough")
parse.add_argument('--random_seed', default=0, type=int, help='Random seed value')
params=parse.parse_args()

random_seed = params.random_seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

assert params.eval_set=="test" or params.eval_set=="validation_early_stop" or params.eval_set=="validation_sphericity"
with open(params.dataset_metadata_path, 'r') as js_file :
    vol_id = json.load(js_file)[params.eval_set]
metadata_path_list = [os.path.join("metadata/C432_sparse", elem+".json") for elem in vol_id]


# Load the hyper-parameters used to train the loaded model
with open(str(Path(params.checkpoint_path).parent/ "params.json"), 'r') as network_param_file :
    network_params = json.load(network_param_file)
train_batch_size = network_params['batch_size']
minivol_size = network_params["minivol_size"]
patch_size = network_params["patch_size"]
enc_emb_dim = network_params["enc_emb_dim"]
enc_depth = network_params["enc_depth"]
num_heads = network_params["num_heads"]

# Set Hyper-parameters
if params.batch_size == None:
    batch_size = train_batch_size
else:
    batch_size = params.batch_size
normalize_data = params.normalize_data

base_device = torch.device("cuda:0")

# Build models
model = models.Unetr_new(minivol_size, in_chans=1, embed_dim=enc_emb_dim, depth=enc_depth, num_heads=num_heads)
model.build_decoder()

#Load Models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    state = torch.load(params.checkpoint_path)
model.to(base_device)
cuda_devices = [i for i in range(torch.cuda.device_count())]
model = torch.nn.DataParallel(model,cuda_devices)
model.load_state_dict(state["model_weights"])
model.eval()

precision_list = []
recal_list = []

for path in metadata_path_list :

    # Build data loading pipeline
    with open(path, 'r') as metadata_file :
        metadata_dict = json.load(metadata_file)
    eval_ds = data.EvalDatasetOverlap(metadata_dict, minivol_size, normalize_data, reflect_padding=False, normalize_crop=params.normalize_crop)
    loader = DataLoader(eval_ds, batch_size=batch_size, num_workers=params.num_workers, pin_memory=True, drop_last=False)
    vol_buffer = data.DestVolBufferOverlap(eval_ds.vol.shape, minivol_size)
    vol_name = path.split("/")[-1][:-5]

    # Run the training loop
    mixed_precision = True
    with torch.no_grad():
        for minivols, coordinates in loader:
            minivols = minivols.to(base_device)
            
            with torch.autocast(device_type='cuda', enabled=mixed_precision, dtype=torch.bfloat16):
                predictions = torch.sigmoid(model(minivols))

            vol_buffer.add_batch(predictions.detach().cpu(), coordinates)
        seg_map = vol_buffer.get_vol(eval_ds.pad_size)
    # Get the semantic segmentation vol
    seg_map = torch.clip(seg_map,0,1)
    seg_map = np.asarray(seg_map, dtype=np.float32)


    # Get instance seg map
    distance = ndi.distance_transform_edt(np.rint(seg_map)).astype(np.float32)
    coords = peak_local_max(distance, min_distance=params.mean_dist_seed_watershed)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    mask = mask.astype(np.int32)
    markers, _ = ndi.label(mask)
    instance_seg_map = watershed(-distance, markers, mask=distance.astype(np.int32))

    # Remove too small segmented cells
    cell_ids = np.unique(instance_seg_map)
    voxel_spacing = (1,1,1)
    voxel_volume = np.prod(voxel_spacing)
    for cur_cell_id in cell_ids:
        if cur_cell_id !=0: # do not consider background
            nuclei_vol = np.sum(instance_seg_map == cur_cell_id) * voxel_volume
            if nuclei_vol< params.volume_threshold: # filter nuclei that are too small
                instance_seg_map[instance_seg_map == cur_cell_id] = 0
            elif params.sphericity_threshold!=0:  # compute the sphericity of the remaining nuclei
                mask = (instance_seg_map == cur_cell_id).astype(np.uint8)
                verts, faces, _, _ = marching_cubes(np.pad(mask, pad_width=1, mode='constant', constant_values=0), level=0.5, spacing=voxel_spacing)
                nuclei_surface = 0.0
                for tri in faces:
                    tri_pts = verts[tri]
                    a = np.linalg.norm(tri_pts[0] - tri_pts[1])
                    b = np.linalg.norm(tri_pts[1] - tri_pts[2])
                    c = np.linalg.norm(tri_pts[2] - tri_pts[0])
                    s = 0.5 * (a + b + c)
                    nuclei_surface += np.sqrt(s * (s - a) * (s - b) * (s - c))  # Heron’s formula
                nuclei_sphericity = (np.pi ** (1/3)) * ((6 * nuclei_vol) ** (2/3)) / nuclei_surface
                if nuclei_sphericity < params.sphericity_threshold:
                    instance_seg_map[instance_seg_map == cur_cell_id] = 0


    # Get coordinates inside the central cube
    central_cube_size = (500,500,500)
    margins = [(s - cs) // 2 for s, cs in zip(distance.shape, central_cube_size)]
    coords_center = np.array([
        c for c in coords
        if all(m <= c[i] < s - m for i, (m, s) in enumerate(zip(margins, distance.shape)))
    ])

    # Coordinates that were filtered out (outside the central cube)
    coords_margin = np.array(list(set(map(tuple, coords)) - set(map(tuple, coords_center))))

    ds_file = h5py.File(metadata_dict["annot_path"], 'r', rdcc_nbytes=(1024*1024*1024), rdcc_w0=0.5, rdcc_nslots=196831)
    gt_points = ds_file[str(list(ds_file.keys())[0])]
    gt_points=np.array(gt_points)
    nb_nuclei = gt_points.sum()

    instance_seg_map_gt = instance_seg_map*gt_points
    detected_gt_nuclei = np.unique(instance_seg_map_gt) 
    nb_detected_gt_nuclei = len(detected_gt_nuclei)-1 # minus 1 because less backgroung

    # As the code from the dataset paper,remove cell nuclei where center is out margin
    for cur_coord in coords_margin:
        id_to_remove = instance_seg_map[cur_coord[0],cur_coord[1],cur_coord[2]]
        instance_seg_map[instance_seg_map==id_to_remove]=0

    # remove gt cell nuclei to get only false positives
    for id_cell in detected_gt_nuclei.tolist():
        instance_seg_map[instance_seg_map==id_cell]=0

    false_positive = len(np.unique(instance_seg_map))-1 # minus 1 because less backgroung

    eps=0
    precision = (eps+nb_detected_gt_nuclei) / (eps+nb_detected_gt_nuclei+false_positive)
    recal = (eps+nb_detected_gt_nuclei) / (eps+nb_nuclei)
    precision_list.append(precision)
    recal_list.append(float(recal))

mean_precision = np.asarray(precision_list).mean()
mean_recal = np.asarray(recal_list).mean()
print("Mean Precision / Mean Recal")
print(mean_precision,mean_recal)
