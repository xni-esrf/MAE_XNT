import torch
import numpy as np
import os
import json
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import math
from torch.utils.data import WeightedRandomSampler, DataLoader
import random

import models
import data
import tools

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Parse input arguments
parse = ArgumentParser(description="")
parse.add_argument('metadata_dir_path', help='Path of the dataset directory metadata')
parse.add_argument('checkpoint_dir', help='Path of the directory where checkpoints are stored')
parse.add_argument('--loaded_checkpoint', default=None, help='Path of a checkpoints to get a training back')
parse.add_argument('--minivol_size', default=64, type=int, help='Training minivolume shape')
parse.add_argument('--patch_size', default=4,  type=int, help='VIT patch size')
parse.add_argument('--batch_size', default=320, type=int, help="The number of minivolumes per batch")
parse.add_argument('--mask_ratio', default=0.9, type=float, help="Masking ratio used")
parse.add_argument('--loss_func', default="L2", type=str, help="Used loss function, can be L1, L2 or BCE")
parse.add_argument('--weight_decay', default=0.05, type=float, help="The weight decay coefficient")
parse.add_argument('--nb_train_epoch', default=300, type=int, help="The number of training epochs")
parse.add_argument('--nb_minivol_per_epoch', default=960000, type=int, help="Define the number of minivol to process in one epoch. It basically sets the duration of one epoch") 
parse.add_argument('--saving_period', default=5, type=int, help='The number of epochs to run between every checkpoint saving')
parse.add_argument('--optim_betas', default=(0.9,0.95), type=float, nargs=2, help="Betas values of the Adam optimizer")
parse.add_argument('--enc_depth', default=12, type=int, help="Number of transformer blocks in the encoder")
parse.add_argument('--dec_depth', default=2, type=int, help="Number of transformer blocks in the decoder")
parse.add_argument('--enc_emb_dim', default=768, type=int, help="Dimenstion of the encoder embeddings")
parse.add_argument('--dec_emb_dim', default=768, type=int, help="Dimenstion of the decoder embeddings")
parse.add_argument('--enc_num_heads', default=12, type=int, help="Number of heads in the encoder attention layers")
parse.add_argument('--dec_num_heads', default=12, type=int, help="Number of heads in the decoder attention layers")
parse.add_argument('--warmup_epoch_nb', default=10, type=int, help="Number of epochs of the warmup stage")
params=parse.parse_args()


"""print("Sharing Strategy : file_system")
torch.multiprocessing.set_sharing_strategy('file_system')"""

# Create output directories if they do not exist
checkpoint_dir = Path(params.checkpoint_dir)
checkpoint_dir.mkdir(exist_ok=True)

# Save training hyperparameters
with open(checkpoint_dir / "params.json", 'w') as par_file:
    json.dump(dict(vars(params)), par_file)

# Get Hyper-parameters
minivol_size = params.minivol_size
batch_size = params.batch_size
weight_decay = params.weight_decay
patch_size = params.patch_size
mask_ratio = params.mask_ratio
warmup_epoch_nb = params.warmup_epoch_nb
nb_train_epoch = params.nb_train_epoch
loss_func = params.loss_func

peak_lr = batch_size*0.00015/256
print("lr is fixed based on the batch size, here it is : {}".format(peak_lr))

base_device = torch.device("cuda:0")

# Get dataset metadata
metadata_dir_path =params.metadata_dir_path
metadata_path_list = []
for root, dirs, files in os.walk(metadata_dir_path):
    for file in files:
        metadata_path_list.append(os.path.join(root, file))
print(len(metadata_path_list))

# Build dataloader from metadata
datasets = data.MultipleVolDataset(metadata_path_list, minivol_size)
weight_sampler = WeightedRandomSampler(datasets.weights, params.nb_minivol_per_epoch)
loader = DataLoader(datasets, batch_size=batch_size, sampler=weight_sampler, num_workers=64, pin_memory=True, drop_last=True, prefetch_factor=2, persistent_workers=True)

# Build models
encoder = models.MAEViTEncoder(minivol_size, patch_size, in_chans=1, embed_dim=params.enc_emb_dim, depth=params.enc_depth, num_heads=params.enc_num_heads)
decoder = models.MAEViTDecoder(params.enc_emb_dim, params.dec_emb_dim ,patch_size, minivol_size, in_chans=1, depth=params.dec_depth, num_heads=params.dec_num_heads)
encoder.train()
decoder.train()
encoder.to(base_device)
decoder.to(base_device)
cuda_devices = [i for i in range(torch.cuda.device_count())]
encoder = torch.nn.DataParallel(encoder,cuda_devices)
decoder = torch.nn.DataParallel(decoder,cuda_devices)

# Sets the optimiser and scheduler
optimizer = torch.optim.AdamW(list(encoder.parameters())+list(decoder.parameters()), lr=peak_lr/warmup_epoch_nb, weight_decay=params.weight_decay, betas=params.optim_betas)

mixed_precision=True
scaler = torch.amp.GradScaler(enabled=mixed_precision)
start_epoch = 0

if params.loaded_checkpoint:
    print("loading {}".format(params.loaded_checkpoint))
    state = torch.load(params.loaded_checkpoint, map_location=base_device)

    encoder.load_state_dict(state["encoder_weights"])
    decoder.load_state_dict(state["decoder_weights"])
    optimizer.load_state_dict(state["optimizer_state"])
    scaler.load_state_dict(state["scaler_state"]) 

    start_epoch = state["epoch"] + 1


# Run the training loop
loss_values = []
for cur_epoch in range(start_epoch, nb_train_epoch):
    epoch_loss = 0
    
    # set epoch learning rate
    if cur_epoch < warmup_epoch_nb:
        new_lr = peak_lr*(cur_epoch+1)/warmup_epoch_nb
    else:
        new_lr = peak_lr/10 + (peak_lr - peak_lr/10 ) * 0.5 * (1. + math.cos(math.pi * (cur_epoch - warmup_epoch_nb) / (nb_train_epoch - warmup_epoch_nb)))
    for group in optimizer.param_groups:
        group['lr'] = new_lr

    for minivols in tqdm(loader, bar_format='{percentage:.0f}% | {elapsed}<{remaining}'):
        minivols = minivols.to(base_device)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', enabled=mixed_precision, dtype=torch.bfloat16):
            enc_output, mask, ids_restore = encoder(minivols, mask_ratio)
            dec_output = decoder(enc_output, ids_restore)

            # Format and normalize target patches
            target_patches = tools.minivols_to_patches(minivols, patch_size)
            
            # Compute loss only for masked patches
            if loss_func=="L2":
                cur_loss = torch.square(dec_output - target_patches)
            elif loss_func=="L1":
                cur_loss = torch.abs(dec_output - target_patches)
            cur_loss = cur_loss.mean(dim=-1)
            cur_loss = (cur_loss * mask).sum() / mask.sum()
        scaler.scale(cur_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += cur_loss
    
    epoch_loss = epoch_loss/len(loader)
    print("Average loss value of epoch {} is {}".format(cur_epoch, epoch_loss))
    loss_values.append(epoch_loss.detach().cpu())
    

    # save model at regular interval
    if cur_epoch % params.saving_period == 0:
        state = {
            "epoch": int(cur_epoch),
            "encoder_weights": encoder.state_dict(),
            "decoder_weights": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict()
        }
        torch.save(state, checkpoint_dir  / f"weights_epoch_{cur_epoch}.cp")
        print("Checkpoint saved" )

        # also save train loss at frequent interval, removing the warmup period
        if cur_epoch >= warmup_epoch_nb and params.loaded_checkpoint == None:
            tools.save_train_loss(loss_values[warmup_epoch_nb-1:], checkpoint_dir)
        
