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
parse = ArgumentParser(description="Training a UNETR model to learn the id function")
parse.set_defaults(lr_cosine_decay=True)
parse.add_argument('dataset_metadata_path', help='Name of a json file, which is used to load information about the processed dataset')
parse.add_argument('checkpoint_dir', help='Path of the directory where checkpoints are stored')
parse.add_argument('found_model_checkpoint_path', help='')
parse.add_argument('--loss_func', default="L2", help="Loss function used")
parse.add_argument('--lr', default=0.0001, type=float, help="The learning rate")
parse.add_argument('--lwise_lr_decay', default=0.85, type=float, help="The learning rate")
parse.add_argument('--no_lr_cosine_decay', dest='lr_cosine_decay', action='store_false', help="Do not train with lr cosine decay")
parse.add_argument('--weight_decay', default=0.0001, type=float, help="The weight decay coefficient")
parse.add_argument('--batch_size', default=96, type=int, help="The number of minivol per batch")
parse.add_argument('--nb_train_epoch', default=100, type=int, help="The number of training epochs")
parse.add_argument('--nb_minivol_per_epoch', default=24000, type=int, help="Define the number of minivol to process in one epoch. It basically sets the duration of one epoch") 
parse.add_argument('--saving_period', default=5, type=int, help='The number of epochs to run between every checkpoint saving')
parse.add_argument('--contr_bright_factors', default=[0.5,0.5], nargs=2, type=float, help='Contrast and brightness augmentation factor')
parse.add_argument('--ds_crop_size', default=None, type=int, help='Dimentions of the central crop extrated from the training volume')
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

# Create the checkpoint directory and its parent if it does not exist.
checkpoint_dir = Path(params.checkpoint_dir)
checkpoint_dir.parent.mkdir(exist_ok=True)
checkpoint_dir.mkdir(exist_ok=True)

# Load the foundation model parameters
enc_depth = 12
enc_emb_dim = 768
minivol_size = 64
patch_size = 4
num_heads = 12

# Save training parameters
training_params = dict(vars(params))
training_params["enc_depth"] = enc_depth
training_params["enc_emb_dim"] = enc_emb_dim
training_params["minivol_size"] = minivol_size
training_params["patch_size"] = patch_size
training_params["num_heads"] = num_heads
with open(checkpoint_dir / "params.json", 'w') as par_file:
    json.dump(training_params, par_file)

# Build data loader
with open(params.dataset_metadata_path, 'r') as js_file :
    vol_id = json.load(js_file)["train"]
metadata_path_list = [os.path.join("metadata/C432_dense", elem+".json") for elem in vol_id]
datasets = data.MultipleVolDataset(metadata_path_list, minivol_size, contr_bright_factors=params.contr_bright_factors, crop_size=params.ds_crop_size)
weight_sampler = WeightedRandomSampler(datasets.weights, params.nb_minivol_per_epoch)
loader = DataLoader(datasets, batch_size=params.batch_size, sampler=weight_sampler, num_workers=16, pin_memory=True, drop_last=True)

# Initialize model and load the found model weights 
if patch_size==4:
    model = models.Unetr_new(minivol_size, in_chans=1, embed_dim=enc_emb_dim, depth=enc_depth, num_heads=num_heads)
elif patch_size==8:
    model = models.Unetr_new_ps8(minivol_size, in_chans=1, embed_dim=enc_emb_dim, depth=enc_depth, num_heads=num_heads)
state = torch.load(params.found_model_checkpoint_path, map_location=torch.device(base_device))
state['encoder_weights'].pop('module.norm.weight')
state['encoder_weights'].pop('module.norm.bias')
state['encoder_weights'] = dict((k[7:],v) for (k,v) in state['encoder_weights'].items())
model.load_state_dict(state['encoder_weights'])
encoder_nb_parameters = 0
for idx, (_, _) in enumerate(model.named_parameters()):
    encoder_nb_parameters += 1
model.build_decoder()
model.train()
model.to(base_device)
cuda_devices = [i for i in range(torch.cuda.device_count())]
model = torch.nn.DataParallel(model,cuda_devices)
decoder_number_parameters = int(encoder_nb_parameters)
for idx, (_, _) in enumerate(model.named_parameters()):
        decoder_number_parameters -= 1
decoder_number_parameters = decoder_number_parameters*(-1)

# Set layer wise lr decay and optimiser
if params.lwise_lr_decay != 1 :
    cur_lwise_lr_decay_rate = 1
    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        layer_names.append(name)
    layer_names.reverse()
    model_parameters = []
    cur_block_num = len(model.module.blocks)
    for idx, name in enumerate(layer_names):
        if idx < decoder_number_parameters:
            model_parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad], 'lr':params.lr, "weight_decay":params.weight_decay, "lwise_lr_decay_rate":1}]
        else:
            model_parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad], 'lr':params.lr*cur_lwise_lr_decay_rate, "weight_decay":params.weight_decay, "lwise_lr_decay_rate":cur_lwise_lr_decay_rate}]
            assert "emb" in name or "block" in name
            if "block" in name :
                if name.split(".")[2] != cur_block_num:
                    cur_block_num = name.split(".")[2]
                    cur_lwise_lr_decay_rate = cur_lwise_lr_decay_rate*params.lwise_lr_decay
    optimizer = torch.optim.AdamW(model_parameters)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
initial_lr = params.lr


# Set loss and optimizer
if params.loss_func == "L1":
    loss_func = torch.nn.L1Loss()
elif params.loss_func == "L2":
    loss_func = torch.nn.MSELoss()
elif params.loss_func == "BCE":
    pos_weight = torch.ones([1], device=base_device)*6
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

mixed_precision=True
scaler = torch.amp.GradScaler(enabled=mixed_precision)

start_epoch_nb = 0
# Trainining loop
for cur_epoch in range(start_epoch_nb, params.nb_train_epoch+1):
    epoch_loss = 0

    if params.lr_cosine_decay:
        lr = initial_lr/10 + (initial_lr - initial_lr/10 ) * 0.5 * (1. + math.cos(math.pi * (cur_epoch) / (params.nb_train_epoch+1)))
        for group in optimizer.param_groups:
            if params.lwise_lr_decay != 1:
                group['lr'] = lr*group["lwise_lr_decay_rate"]
            else:
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
