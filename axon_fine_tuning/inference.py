#Inference

import torch
from util.util_inference import *
import argparse
import models.models as models
import os
import h5py
import json

parse = argparse.ArgumentParser()
parse.add_argument('rootdata', type=str, help='The root directory')
parse.add_argument("model_folder", type=str, default='model_aff', help='model folder')
parse.add_argument("--file", type=str, default='val', help='train val or test volume')
parse.add_argument("--batch_size", type=int, default=100, help='batch size for inference')
parse.add_argument("--agreg", type=str, default='hann', help='aggregation method for overlapping patches')
parse.add_argument("--volume_size", type=int, default=699, help='size of the total volume to be processed')
parse.add_argument("--offset", type=int, default=252, help='offset')
parse.add_argument("--threshold", type=float, default=0, help='threshold for agglomeration, put 0 if you want the affinity map')
params = parse.parse_args()

device = 'cuda:0'

with open(os.path.join(params.model_folder,'params.json'), 'r') as network_param_file:
    model_params = json.load(network_param_file)
model = models.Unetr_light_finetune(model_params['minivol_size'], model_params['patch_size'], in_chans=1, embed_dim=model_params['enc_emb_dim'], depth=model_params['enc_depth'], num_heads=12)
model.build_decoder()

model.to(device)
cuda_devices = [i for i in range(torch.cuda.device_count())]
model = torch.nn.DataParallel(model,cuda_devices)
checkpoint = torch.load(os.path.join(params.model_folder, 'model_final.pt'), weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

if isinstance(params.volume_size, int):
    params.volume_size = (params.volume_size, params.volume_size, params.volume_size)
if isinstance(params.offset, int):
    params.offset = (params.offset, params.offset, params.offset)
    
if __name__ == '__main__':
    # In case we only want to visualize affinities
    if params.threshold == 0:
        file_aff = f'result_{params.file}_{params.offset}_{params.volume_size}/'
        affs = run_inference(params.file, model, params.model_folder, params.rootdata, params.batch_size, params.agreg, params.volume_size, params.offset, params.threshold)
        os.makedirs(os.path.join(params.model_folder + file_aff), exist_ok=True)
        # Save affinity map prediction
        with h5py.File(os.path.join(params.model_folder,file_aff,'affs.h5'), 'w') as f:
            f.create_dataset("volumes", data=affs)
        f.close()
    else:
        seg_final = run_inference(params.file, model, params.model_folder, params.rootdata, params.batch_size, params.agreg, params.volume_size, params.offset,  params.threshold)
        save_prediction(seg_final, params.file, params.model_folder, params.agreg, params.volume_size, params.offset, params.threshold)
        

