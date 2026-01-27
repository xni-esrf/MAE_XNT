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
import random

import models
import data
import tools

# Parse input arguments
parse = ArgumentParser(description="")
parse.add_argument('dataset_metadata_path' ,help='Name of a json file, which is used to load information about the processed dataset')
parse.add_argument('checkpoint_path', help='Path of the checkpoint to be loaded')
parse.add_argument('eval_set', help="Use test, validation or train set to evaluate performances")
parse.add_argument('--batch_size', default=None, type=int, help="The number of minivol per batch, if unspecified, us the same as training batch size")
parse.add_argument('--no_normalization', action='store_false', dest="normalize_data", default=True, help="Normalize training data")
parse.add_argument('--num_workers', default=16, type=int, help="Number of workers used in the Dataloader")
parse.add_argument('--mask_border_size', default=50, type=int, help="Margin size not considered in the final results")
parse.add_argument('--no_reflect_padding', action='store_false', dest="reflect_padding", default=True, help="Use reflect padding to improve edge results")
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

# Get volumes metadata
assert params.eval_set=="train" or params.eval_set=="test" or params.eval_set=="validation"
with open(params.dataset_metadata_path, 'r') as js_file :
    vol_id = json.load(js_file)[params.eval_set]
metadata_path_list = [os.path.join("metadata/C432_dense", elem+".json") for elem in vol_id]

# Load the hyper-parameters used to train the loaded model
with open(str(Path(params.checkpoint_path).parent/ "params.json"), 'r') as network_param_file :
    network_params = json.load(network_param_file)
train_batch_size = network_params['batch_size']
minivol_size = network_params["minivol_size"]
patch_size = network_params["patch_size"]
enc_emb_dim = network_params["enc_emb_dim"]
enc_depth = network_params["enc_depth"]
num_heads = network_params["num_heads"]

if params.batch_size == None:
    batch_size = train_batch_size
else:
    batch_size = params.batch_size
normalize_data = params.normalize_data
mask_border_size = params.mask_border_size

base_device = torch.device("cuda:0")

# Build models
if patch_size == 4:
    model = models.Unetr_new(minivol_size, in_chans=1, embed_dim=enc_emb_dim, depth=enc_depth, num_heads=num_heads)
elif patch_size==8:
    model = models.Unetr_new_ps8(minivol_size, in_chans=1, embed_dim=enc_emb_dim, depth=enc_depth, num_heads=num_heads)
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


mean_acc = 0
mean_dice = 0
mean_precision = 0
mean_recall = 0
mixed_precision = True
for path in metadata_path_list :

    # Build data loading pipeline
    with open(path, 'r') as metadata_file :
        metadata_dict = json.load(metadata_file)
    eval_ds = data.EvalDatasetOverlap(metadata_dict, minivol_size, normalize_data, reflect_padding=params.reflect_padding)
    loader = DataLoader(eval_ds, batch_size=batch_size, num_workers=params.num_workers, pin_memory=True, drop_last=False)
    vol_buffer = data.DestVolBufferOverlap(eval_ds.vol.shape, minivol_size)

    # Run the training loop
    with torch.no_grad():
        for minivols, coordinates in loader:
            minivols = minivols.to(base_device)
            
            with torch.autocast(device_type='cuda', enabled=mixed_precision, dtype=torch.bfloat16):
                pred_minivols = model(minivols)

            vol_buffer.add_batch(pred_minivols.detach().cpu(), coordinates)

        pred_vol = vol_buffer.get_vol(eval_ds.pad_size)

    segmented_vol = torch.minimum(torch.maximum(torch.round(pred_vol), torch.zeros(pred_vol.shape)), torch.ones(pred_vol.shape))
    annotated_vol = data.get_annotation_array(metadata_dict)
    annotated_vol = torch.round(torch.squeeze(annotated_vol))

    if mask_border_size!=0:
        segmented_vol = segmented_vol[mask_border_size:segmented_vol.shape[0]-mask_border_size,mask_border_size:segmented_vol.shape[1]-mask_border_size,mask_border_size:segmented_vol.shape[2]-mask_border_size]
        annotated_vol = annotated_vol[mask_border_size:annotated_vol.shape[0]-mask_border_size,mask_border_size:annotated_vol.shape[1]-mask_border_size,mask_border_size:annotated_vol.shape[2]-mask_border_size]


    tp = torch.sum(segmented_vol*annotated_vol)
    fp = torch.sum(segmented_vol*(1-annotated_vol))
    fn = torch.sum((1 - segmented_vol) * annotated_vol)
    tn = torch.sum((1 - segmented_vol) * (1-annotated_vol))

    eps=0.0001
    accuracy = (tp+tn+eps)/(tp+tn+fn+fp+eps)
    dice = (2*tp+eps)/(2*tp+fp+fn+eps)
    precision = (tp+eps)/(tp+fp+eps)
    recall = (tp+eps)/(tp+fn+eps)

    mean_acc += accuracy
    mean_dice += dice
    mean_precision += precision
    mean_recall += recall

mean_acc = mean_acc/len(metadata_path_list)
mean_dice = mean_dice/len(metadata_path_list)
mean_precision = mean_precision/len(metadata_path_list)
mean_recall = mean_recall/len(metadata_path_list)

mean_acc = round(mean_acc.item(),3)
mean_dice = round(mean_dice.item(),3)
mean_precision = round(mean_precision.item(),3)
mean_recall = round(mean_recall.item(),3)
print(mean_dice)
