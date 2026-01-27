import torch
import numpy as np
import json
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import models.models as models
from util.data_prep import * 
from util.util_eval import validate_model
import csv
import os


# Parse input arguments
parse = ArgumentParser(description="Training a UNETR model")
parse.add_argument('rootdata', type=str, help='The root directory')
parse.add_argument('model_dir', type=str, help='The model saving directory')
parse.add_argument('--found_model_checkpoint_path', type=str, help='The pretrained model directory')
parse.add_argument('--mode_train', default='finetune', type=str, help='scratch or finetune')
parse.add_argument('--lr', default=5e-4, type=float, help="The learning rate")
parse.add_argument('--lr_cosine_decay', default=True, help="Train with lr cosine decay")
parse.add_argument('--lwise_lr_decay', default=0.75, type=float, help="Layer wise lr decay parameter")
parse.add_argument('--minivol_size',  type=int, default=64, help='The minivol size that the model will use as input')
parse.add_argument('--num_steps', default=400, type=int, help='The total number of steps to train the network')
parse.add_argument('--saving_period', default=25, type=int, help='Number of steps between each save and test on validation')
parse.add_argument('--num_workers', default=32, type=int, help='The number of workers that work in parallel (cpu). Must be inferior to number of cpu cores.')
parse.add_argument('--volume_size', default=200, type=int, help='The volume size on which we will train the model') # In general not useful to modify, was used for incremental tests
parse.add_argument('--weight_decay', default=0.05, type=float, help="The weight decay coefficient")
parse.add_argument('--seed', default=42, type=int, help="The seed")
parse.add_argument('--batch_size', default=152, type=int, help="The batch size")
parse.add_argument('--whole', default=True, type=str, help="val on whole validation set")
parse.add_argument('--tune_threshold', default=False, type=bool, help="whether to tune the agglomeration threshold or not")
parse.add_argument("--batch_size_inference", type=int, default=250, help='batch size for inference')

params=parse.parse_args()
params.lr_min = params.lr/10

params.pad_size = 48 # Context around labeled region
print(params.model_dir)
params.model_dir = os.path.join(params.model_dir, f'{params.lr}_{params.lwise_lr_decay}_{params.weight_decay}_{params.num_steps}')
print(params.model_dir)

os.makedirs(params.model_dir, exist_ok=False)
base_device = torch.device("cuda:0")
set_seeds(params.seed) # Use if you want to fix seed

# Neighborhood definition for affinity prediction
neighb = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-3, 0, 0],
    [0, -3, 0],
    [0, 0, -3],
    [-9, 0, 0],
    [0, -9, 0],
    [0, 0, -9],
    [-27, 0, 0],
    [0, -27, 0],
    [0, 0, -27]
])

# Image and label files
img_file = os.path.join(params.rootdata, 'xpress-training-raw.h5')
label_file = os.path.join(params.rootdata, 'xpress-training-voxel-labels.h5')

transform = [RandomCrop3d(params.minivol_size), RandomRot(), RandomNoise(std=0.5), RandomShift(shift_low=-0.2, shift_high=0.2, scale_low=0.8, scale_high=1)]

dataset = PatchedData(label_file, img_file, pad_size=params.pad_size, volume_size=params.volume_size, minivol_size=params.minivol_size, neighb=neighb, transform=transform)

Sampler = torch.utils.data.WeightedRandomSampler([1], 1000000)
train_dataloader = DataLoader(dataset, batch_size=params.batch_size, sampler=Sampler, num_workers=params.num_workers, prefetch_factor=1, pin_memory=True)

# Load the foundation model parameters

enc_depth = 12
enc_emb_dim = 768
minivol_size = 64
patch_size = 4
normalize_data = True

# Save training parameters
training_params = dict(vars(params))
training_params["enc_depth"] = enc_depth
training_params["enc_emb_dim"] = enc_emb_dim
training_params["minivol_size"] = minivol_size
training_params["patch_size"] = patch_size
training_params["normalize_data"] = normalize_data

with open(os.path.join(params.model_dir,"params.json"), 'w') as par_file:
    json.dump(training_params, par_file)


# Initialize model and load the found model weights 
model = models.Unetr_light_finetune(minivol_size, patch_size, in_chans=1, embed_dim=enc_emb_dim, depth=enc_depth, num_heads=12)
if params.mode_train == 'finetune':
    state = torch.load(params.found_model_checkpoint_path, map_location=torch.device(base_device), weights_only=True)
    state['encoder_weights'] = dict((k[7:],v) for (k,v) in state['encoder_weights'].items())
    model.load_state_dict(state['encoder_weights'], strict=False)

model.build_decoder()
model.train(True)
model.to(base_device)

cuda_devices = [i for i in range(torch.cuda.device_count())]
model = torch.nn.DataParallel(model,cuda_devices)

# Set layer wise lr decay and optimiser
param_groups = {}
num_layers = enc_depth + 1
layer_scales = list(params.lwise_lr_decay ** (num_layers - i) for i in range(num_layers + 1))
for name, param in model.named_parameters():
    name = name.split('.')
    if (name[1] == 'pos_emb') or (name[1] == 'patch_embed'):
        n_layer = 0
    elif name[1] == 'blocks':
        n_layer = int(name[2]) + 1
    else:
        n_layer = num_layers
    group_name = f"layer_{n_layer}"
    if group_name not in param_groups :
        param_groups[group_name] = {"rate" : layer_scales[n_layer], "params" : []}
    param_groups[group_name]["params"].append(param)
optimizer = torch.optim.AdamW(list(param_groups.values()), lr=params.lr, weight_decay=params.weight_decay)
initial_lr = params.lr
lr = initial_lr

# Set loss 
class weighted_MSELoss(torch.nn.Module):  
    def __init__(self):
        super().__init__()
    def forward(self,preds,targets,weights):
        squared_error = ((preds - targets)**2) * weights
        return torch.mean(torch.masked_select(squared_error, weights>0))
    
loss_func = weighted_MSELoss()

mixed_precision=True
scaler = torch.amp.GradScaler(enabled=mixed_precision)

# Model training
# Agglomeration thresholds for validation
if params.tune_threshold :
    thresholds = [0.5,0.6,0.7]
else:
    thresholds = [0.7]

#Initialize evalutation file

with open(os.path.join(params.model_dir,'res.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['thresholds :'] + thresholds)
    writer.writerow(['epoch'])

with open(os.path.join(params.model_dir, 'log.txt'), 'w') as f:
    f.write('start\n')

# Training loop
max_val = 0
pbar = tqdm(total=params.num_steps)
for step, batch in enumerate(train_dataloader):
    optimizer.zero_grad()
    inputs, targets, masks = batch
    inputs, targets, masks = inputs.to(base_device), targets.to(base_device), masks.to(base_device)
    weights_zeros, weights_ones = WeightingAffinity(targets, masks)
    weights = masks * ((targets * weights_ones) - (targets - 1)*weights_zeros)

    if params.lr_cosine_decay:
        # Cosine annealing
        lr = params.lr_min + 1/2 * (initial_lr - params.lr_min) * (1 + np.cos(step / (params.num_steps) * np.pi))
    for group in optimizer.param_groups:
        group['lr'] = lr * group['rate']

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=mixed_precision):
        pred = model(inputs)
        loss_value = loss_func(pred, targets, weights)

    scaler.scale(loss_value).backward()
    scaler.step(optimizer)
    scaler.update()
    pbar.write(str(loss_value.item()))
    pbar.update(1)

    with open(os.path.join(params.model_dir,'log.txt'), 'a') as f:
        f.write(f'loss : {loss_value.item()}\n')


    if (step % params.saving_period == 0) & (step > 24):

        model.eval()
        # Model monitoring on validation
        validate_model(model, params.model_dir, params.rootdata, params.batch_size_inference, step, thresholds, params.whole)
        model.train()


        with open(os.path.join(params.model_dir,'res.csv'), 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            rows = [row for row in reader]
            last_val = max([float(el) for el in rows[-1][1:]])
            #Keep best
            if last_val > max_val:
                max_val = last_val
                torch.save({    
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'validation_score': last_val,
                    }, os.path.join(params.model_dir,'model_final.pt'))
    if step > params.num_steps:
        break

