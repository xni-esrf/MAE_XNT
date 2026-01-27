
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from tqdm import tqdm
import torch
import h5py 
import json
import os
import multiprocessing
import time
import waterz

# Monkey patch waterz __init__.py
_original_expanduser = os.path.expanduser
project_root = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(project_root, '.cython_cache')
def _patched_expanduser(path):
    if path == '~/.cython/inline':
        return cache_dir
    return _original_expanduser(path)

os.path.expanduser = _patched_expanduser

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_hann_window(minivol_size):
    ## torchio hann window function
    hann_window_3d = torch.as_tensor([1])
    for spatial_dim, size in enumerate(minivol_size):
        window_shape = np.ones_like(minivol_size)
        window_shape[spatial_dim] = size
        hann_window_1d = torch.hann_window(
            size + 2,
            periodic=False,
        )
        hann_window_1d = hann_window_1d[1:-1].view(*window_shape)
        hann_window_3d = hann_window_3d * hann_window_1d
    return hann_window_3d

def pred_aff_from_sub_volumes(img, model, batch_size, minivol_size, stride, agreg='hann'):
    # Model inference on subvolumes and agglomeration 
    device = torch.device("cuda:0")
    # Modify the batch size to make it fit on your hardware, the original configuration requires approximately 160 GB GPU RAM
    mixed_precision = True
    d_dim_in, h_dim_in, w_dim_in = img.shape

    # (D, H, W) -> (d_dim_out, h_dim_out, w_dim_out, minivol_size, minivol_size, minivol_size)
    minivols = img.unfold(0, minivol_size, stride).unfold(1, minivol_size, stride).unfold(2, minivol_size, stride)
    d_dim_out, h_dim_out, w_dim_out = minivols.shape[:3]

    # (d_dim_out, h_dim_out, w_dim_out, minivol_size, minivol_size, minivol_size) -> (d_dim_out * h_dim_out * w_dim_out, 1, minivol_size, minivol_size, minivol_size)
    minivols = minivols.flatten(0,2).unsqueeze(0).transpose(1,0)
    total_minivols = minivols.shape[0]  

    final_pred = torch.zeros((total_minivols, 3, minivol_size, minivol_size, minivol_size), dtype=torch.float16)

    with torch.no_grad():
        for indices in tqdm(batch(np.arange(total_minivols), batch_size)):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=mixed_precision):
                pred = model(minivols[indices].to(device))   
                # Keep only three direct neighbor affinities           
                final_pred[indices] = pred[:,:3].cpu()

    final_pred = final_pred.to(torch.float32)

    # (total_minivols, 3, minivol_size, minivol_size, minivol_size) -> (3, d_dim_out, h_dim_out, w_dim_out, minivol_size, minivol_size, minivol_size)
    affs = final_pred.permute(1, 0, 2, 3, 4).reshape(3, d_dim_out, h_dim_out, w_dim_out, minivol_size, minivol_size, minivol_size)

    if agreg == 'hann':
        norm = get_hann_window((minivol_size, minivol_size, minivol_size)).tile((3, total_minivols, 1, 1, 1)).reshape(3, d_dim_out, h_dim_out, w_dim_out, minivol_size, minivol_size, minivol_size)
    elif agreg == 'avg':
        norm = torch.ones_like(affs)
    
    affs = affs * norm

    # (3, d_dim_out, h_dim_out, w_dim_out, minivol_size, minivol_size, minivol_size) -> (3, minivol_size, d_dim_out, minivol_size, minivol_size, h_dim_out, w_dim_out)
    affs = affs.permute(0, 4, 1, 5, 6, 2, 3)
    norm = norm.permute(0, 4, 1, 5, 6, 2, 3)

    # Folding along height and width dimensions
    affs = affs.reshape(3, minivol_size * d_dim_out * minivol_size * minivol_size, h_dim_out * w_dim_out)
    norm = norm.reshape(3, minivol_size * d_dim_out * minivol_size * minivol_size, h_dim_out * w_dim_out)
    affs = torch.nn.functional.fold(affs, output_size=(h_dim_in, w_dim_in), kernel_size=(minivol_size, minivol_size), stride=(stride, stride))
    norm = torch.nn.functional.fold(norm, output_size=(h_dim_in, w_dim_in), kernel_size=(minivol_size, minivol_size), stride=(stride, stride))

    # Folding along depth dimension
    affs = affs.reshape(3, minivol_size, d_dim_out * h_dim_in * w_dim_in)
    norm = norm.reshape(3, minivol_size, d_dim_out * h_dim_in * w_dim_in)
    affs = torch.nn.functional.fold(affs, output_size=(d_dim_in, h_dim_in * w_dim_in), kernel_size=(minivol_size, 1), stride=(stride, 1))
    norm = torch.nn.functional.fold(norm, output_size=(d_dim_in, h_dim_in * w_dim_in), kernel_size=(minivol_size, 1), stride=(stride, 1))

    # Normalize affinities
    affs = affs / norm
    # (3, D, H * W) -> (3, D, H, W) and clipping because we don't use activation function
    affs = torch.clip(affs.reshape((3, d_dim_in, h_dim_in, w_dim_in)), 0, 1)
    
    return affs

def filter_fragments(aff, segmentation, threshold):
    # Filter fragments with a total mean affinity less than threshold
    mean_aff = np.mean(aff, axis=0)
    fragment_ids = np.unique(segmentation)
    for fragment in fragment_ids:
        seg_frag = segmentation == fragment
        mean = np.mean(mean_aff[seg_frag])
        if mean < threshold:
            segmentation[seg_frag] = 0
    return segmentation


def dilate_fragments(seg):
    # Binary dilation on fragments
    for i, frame in enumerate(seg):
        for j in range(3):
            mask = frame == 0
            modif = ndi.grey_dilation(frame, size=3)
            frame[mask] = modif[mask]
            seg[i] = frame
    return seg

def watershed_from_aff(aff):
    # Going from affinity map to fragments
    boundary = np.mean(aff, axis=0)
    max_aff = np.max(aff)
    # Binary boundary map
    boundary[boundary<0.5*max_aff] = 0.0
    boundary[boundary>=0.5*max_aff] = 1.0

    # Distance transform on the binary boundary map
    distance = ndi.distance_transform_edt(boundary)

    # Seeds for watershed based on local maxima
    coordinates = peak_local_max(distance, min_distance=5, labels=boundary.astype(np.int64))
    mask = np.zeros(distance.shape)
    mask[tuple(coordinates.T)] = True

    markers, _ = ndi.label(mask)
    # Watershed
    labels = watershed(-distance, markers, mask=boundary.astype(np.int64))
    labels = filter_fragments(aff, labels, threshold=0.3)
    labels = dilate_fragments(labels)

    return labels


def generate_subvolume_indices(volume_size, stride):
    # Creation of subvolume coordinates for parallel processing 
    indices = []
    
    # Creates the list of indices for each subvolume
    for z in range(0, volume_size[0], stride):
        for y in range(0, volume_size[1], stride):
            for x in range(0, volume_size[2], stride):
                indices.append((z,y,x))
                
    return indices

def save_prediction(segmentation, file,  model_folder, agreg, volume_size, offset, threshold):
    # Utility function to save segmentation output in h5 file
    voxel_size=33
    save_folder = f'result_{file}_{offset}_{volume_size}_{agreg}_{threshold}/'
    offset = offset[0]
    # Write the result in a h5 file
    os.makedirs(os.path.join(model_folder,save_folder), exist_ok=True)
    f = h5py.File(os.path.join(model_folder,save_folder,'submission.h5'), 'w')
    if file == 'test':
        segmentation = segmentation[::3,::3,::3]
        voxel_size = voxel_size*3
    dset = f.create_dataset("submission", data=segmentation)
    dset.attrs['offset'] = np.array([offset*33]*3)
    dset.attrs['resolution'] = np.array([voxel_size]*3)
    f.close()

def run_inference(file, model, model_folder, rootdata, batch_size, agreg, volume_size, offset, threshold):
    # Inference pipeline from affinity prediction to final segmentation
    start = time.time()

    if file == 'train':
        img_file = os.path.join(rootdata,'xpress-training-raw.h5')
    elif file == 'val':
        img_file = os.path.join(rootdata,'xpress-validation-raw.h5')
    elif file == 'test':
        img_file = os.path.join(rootdata,'xpress-test-raw.h5')
    else:
        img_file = file


    with open(os.path.join(model_folder,'params.json'), 'r') as network_param_file:
        model_params = json.load(network_param_file)
    
    if isinstance(model_params['minivol_size'], list):
        minivol_size = model_params['minivol_size'][0]
    else: 
        minivol_size = model_params['minivol_size']
        

    stride = minivol_size//2

    # Handling non-cubic volumes: volume_size and offset as tuples (depth, height, width)
    if isinstance(volume_size, int):
        volume_size = (volume_size, volume_size, volume_size)
    if isinstance(offset, int):
        offset = (offset, offset, offset)
    
    num_mini_vox = [int(np.ceil((volume_size[i] - minivol_size) / stride)) for i in range(3)]
    size_output = [num_mini_vox[i] * stride + minivol_size for i in range(3)]
    oversize = [size_output[i] - volume_size[i] for i in range(3)]
    start_roi = [int(offset[i] - np.ceil(oversize[i] / 2)) for i in range(3)]
    end_roi = [start_roi[i] + size_output[i] for i in range(3)]

    for i in range(3):
        if end_roi[i] > 1200 or start_roi[i] < 0:
            raise RuntimeError(f'Fail: ROI specified out of range in dimension {i}')

    img = torch.tensor(h5py.File(img_file)['volumes']['raw'][start_roi[0]:end_roi[0],
                                                            start_roi[1]:end_roi[1],
                                                            start_roi[2]:end_roi[2]]).type(torch.float32)

    # Standardization
    img = (img - img.mean()) / img.std()

    file_aff = f'result_{file}_{offset}_{volume_size}/'

    if threshold == 0:
        affs = pred_aff_from_sub_volumes(img, model, batch_size, minivol_size, stride, agreg)
        affs = affs.numpy()
        
        # Adjusting for non-cubic shape
        affs = affs[:,
                    int(np.ceil(oversize[0] / 2)):size_output[0] - oversize[0] // 2,
                    int(np.ceil(oversize[1] / 2)):size_output[1] - oversize[1] // 2,
                    int(np.ceil(oversize[2] / 2)):size_output[2] - oversize[2] // 2]
        return affs
    
    try:
        affs = h5py.File(os.path.join(model_folder,file_aff,'affs.h5'))['volumes'][:,:,:,:]
    except FileNotFoundError:
        affs = run_inference(file, model, model_folder, rootdata, batch_size, agreg, volume_size, offset, threshold=0)

    seg_merge = np.zeros_like(affs[0], dtype=np.uint64)
    # Partitioning of the volume 
    minivol_par_size = 128
    context = 24
    batch_size = multiprocessing.cpu_count()
    idcs = generate_subvolume_indices(volume_size, minivol_par_size)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    segs = []
    for i in tqdm(range(0, len(idcs), batch_size)):
        batch_idcs = idcs[i:i+batch_size]
        batch_subvolumes = [affs[:, max(z-context,0) : min(z + minivol_par_size + context, volume_size[0]), max(y-context,0) : min(y + minivol_par_size + context, volume_size[1]), max(x-context,0) : min(x + minivol_par_size + context, volume_size[2])] for z, y, x in batch_idcs]
        segs.extend(pool.map(watershed_from_aff, batch_subvolumes))
    id_offset = 0
    # Merging segmentations and relabeling for unique labels per mini volume
    for i, idc in enumerate(idcs):
        z, y, x = idc
        add = np.zeros_like(segs[i])
        # Don't add offset to 0 labels
        add[segs[i] != 0] += (segs[i][segs[i] != 0] + id_offset)

        seg_merge[z : min(z + minivol_par_size, volume_size[0]), y : min(y + minivol_par_size, volume_size[1]), x : min(x + minivol_par_size, volume_size[2])] = add[min(z,context):min(min(z,context)+minivol_par_size, volume_size[0]),min(y,context):min(min(y,context)+minivol_par_size, volume_size[1]),min(x,context):min(min(x,context)+minivol_par_size, volume_size[2])]
        id_offset +=  np.max(segs[i])
    # Hierarchical agglomeration on the whole fragment map, could be done on subvolumes aswell
    seg_final = next(waterz.agglomerate(affs, thresholds=[threshold] ,fragments=seg_merge))
    print('execution time : ', time.time() - start)

    return seg_final







