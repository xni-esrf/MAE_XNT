import torch
import numpy as np
import os
import json
import sys
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import WeightedRandomSampler, DataLoader
import math
import random

import tools
import data
import models

# Parse input arguments
parse = ArgumentParser(description="Training from scratch a U-Netr to segment volumes")
parse.set_defaults(lr_cosine_decay=True)
parse.add_argument('dataset_metadata_path', help='Name of a json file, which is used to load information about the processed dataset')
parse.add_argument('checkpoint_dir', help='Path of the directory where checkpoints are stored')
parse.add_argument('--loaded_checkpoint_path', default=None, help='If specified, path to the checkpoint to be loaded from')
parse.add_argument('--loss_func', default="L2", help="Loss function used")
parse.add_argument('--lr', default=0.0002, type=float, help="The learning rate")
parse.add_argument('--no_lr_cosine_decay', dest='lr_cosine_decay', action='store_false', help="Do not train with lr cosine decay")
parse.add_argument('--weight_decay', default=0.0001, type=float, help="The weight decay coefficient")
parse.add_argument('--enc_depth', default=12, type=int, help="Number of blocks in the U-Netr")
parse.add_argument('--enc_emb_dim', default=768, type=int, help="Embedding dimension in the blocks")

parse.add_argument('--num_heads', default=12, type=int, help="Number of heads in the attention layer")
parse.add_argument('--minivol_size', default=64, type=int, help='Training minivol shape')
parse.add_argument('--patch_size', default=4,  type=int, help='VIT patch size')
parse.add_argument('--batch_size', default=96, type=int, help="The number of minivol per batch")
parse.add_argument('--nb_train_epoch', default=100, type=int, help="The number of training epochs")
parse.add_argument('--nb_minivol_per_epoch', default=24000, type=int, help="Define the number of minivol to process in one epoch. It basically sets the duration of one epoch") 
parse.add_argument('--saving_period', default=5, type=int, help='The number of epochs to run between checkpoint savings')
parse.add_argument('--contr_bright_factors', default=[0.5,0.5], nargs=2, type=float, help='Contrast and brightness augmentation factors')
parse.add_argument('--ds_crop_size', default=None, type=int, help='Dimension of the central crop extracted from the training volume')
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

base_device = torch.device("cuda:0")

# Create the checkpoint directory and its parent if they do not exist
checkpoint_dir = Path(params.checkpoint_dir)
checkpoint_dir.parent.mkdir(exist_ok=True)
checkpoint_dir.mkdir(exist_ok=True)

# Save training parameters
with open(checkpoint_dir / "params.json", 'w') as par_file:
    json.dump(dict(vars(params)), par_file)

# Build data loader
with open(params.dataset_metadata_path, 'r') as js_file :
    vol_id = json.load(js_file)["train"]
metadata_path_list = [os.path.join("metadata/C432_dense", elem+".json") for elem in vol_id]
datasets = data.MultipleVolDataset(metadata_path_list, params.minivol_size, contr_bright_factors=params.contr_bright_factors, crop_size=params.ds_crop_size)
weight_sampler = WeightedRandomSampler(datasets.weights, params.nb_minivol_per_epoch)
loader = DataLoader(datasets, batch_size=params.batch_size, sampler=weight_sampler, num_workers=64, pin_memory=True, drop_last=True)

# Initialize model
if params.patch_size==4:
    model = models.Unetr_new(params.minivol_size, in_chans=1, embed_dim=params.enc_emb_dim, depth=params.enc_depth, num_heads=params.num_heads)
elif params.patch_size==8:
    model = models.Unetr_new_ps8(params.minivol_size, in_chans=1, embed_dim=params.enc_emb_dim, depth=params.enc_depth, num_heads=params.num_heads)


model.build_decoder()
model.train()
model.to(base_device)
cuda_devices = [i for i in range(torch.cuda.device_count())]
model = torch.nn.DataParallel(model,cuda_devices)

# Set loss and optimizer
if params.loss_func == "L1":
    loss_func = torch.nn.L1Loss()
elif params.loss_func == "L2":
    loss_func = torch.nn.MSELoss()
elif params.loss_func == "BCE":
    pos_weight = torch.ones([1], device=base_device)*6
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

lr = params.lr
initial_lr = params.lr
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=params.weight_decay)

start_epoch_nb = 0
if params.loaded_checkpoint_path:
    print("Loading weights...")
    state = torch.load(params.loaded_checkpoint_path, map_location=torch.device(base_device))
    model.load_state_dict(state['model_weights'])
    optimizer.load_state_dict(state['optimizer_state'])
    start_epoch_nb = state['epoch']+1

mixed_precision=True
scaler = torch.amp.GradScaler(enabled=mixed_precision)

# Trainining loop
for cur_epoch in range(start_epoch_nb, params.nb_train_epoch+1):
    epoch_loss = 0

    if params.lr_cosine_decay:
        lr = initial_lr/10 + (initial_lr - initial_lr/10 ) * 0.5 * (1. + math.cos(math.pi * (cur_epoch) / (params.nb_train_epoch+1)))
        print("lr decayed to {}".format(lr))
        for group in optimizer.param_groups:
            group['lr'] = lr


    for minivols, annotations in tqdm(loader, bar_format='{percentage:.0f}% | {elapsed}<{remaining}'):
        minivols, annotations = minivols.to(base_device), annotations.to(base_device)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', enabled=mixed_precision, dtype=torch.bfloat16):
            dec_output = model(minivols)
            cur_loss = loss_func(dec_output, annotations)

        scaler.scale(cur_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += cur_loss


    print("Average loss value of epoch {} is {}".format(cur_epoch, epoch_loss/len(loader)))

    # save model at regular interval
    if cur_epoch % params.saving_period == 0 :
        print("saving interval reached (epoch n°{}), saving checkpoint...".format(cur_epoch))
        state = {
            "epoch": int(cur_epoch),
            "model_weights": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(state, checkpoint_dir  / f"weights_epoch_{cur_epoch}.cp")
