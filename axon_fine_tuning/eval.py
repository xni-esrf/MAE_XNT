
from util.util_eval import  run_eval, test_model
import argparse
import json
import models.models as models
import torch
import os

parse = argparse.ArgumentParser()
parse.add_argument('rootdata', type=str, help='The root directory')
parse.add_argument("model_folder", type=str, help='model folder')
parse.add_argument("--file", default='val', type=str, help='train val or test volume')
parse.add_argument("--agreg", type=str, default='hann', help='aggregation method for overlapping patches')
parse.add_argument("--batch_size", type=int, default=100, help='batch size for inference')
parse.add_argument("--volume_size", type=int, default=699, help='size of the total volume to be processed, put 0 if you want to evaluate on a pre defined partition of the validation set')
parse.add_argument("--offset", type=int, default=252, help='offset')
parse.add_argument("--threshold", type=float, default=0.7, help='threshold for agglomeration')

params = parse.parse_args()

device = 'cuda:0'

with open(os.path.join(params.model_folder,'params.json'), 'r') as network_param_file:
    model_params = json.load(network_param_file)
model = models.Unetr_light_finetune(model_params['minivol_size'], model_params['patch_size'], in_chans=1, embed_dim=model_params['enc_emb_dim'], depth=model_params['enc_depth'], num_heads=12)
model.build_decoder()

model.to(device)
cuda_devices = [i for i in range(torch.cuda.device_count())]
model = torch.nn.DataParallel(model,cuda_devices)
checkpoint = torch.load(os.path.join(params.model_folder,'model_final.pt'), weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

if  isinstance(params.offset, int):
    params.offset = (params.offset, params.offset, params.offset)

if isinstance(params.volume_size, int):
    params.volume_size = (params.volume_size, params.volume_size, params.volume_size)

if params.volume_size == (0, 0, 0):
    test_model(model, params.model_folder, params.rootdata, params.batch_size, params.threshold)
else:
    run_eval(params.offset, params.volume_size, params.agreg, params.file, model, params.model_folder, params.rootdata, params.batch_size, params.threshold)