import glob
import os, sys
import re
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
sys.path.append(os.getcwd())
from dld.data.utils.preprocess import Normalizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="finedance",
    help="dataset name",
)

args = parser.parse_args()
dataset_name = args.dataset
modir = f"data/{dataset_name}/mofea319"

data_li = []
for file in tqdm(os.listdir(modir)):
    if not file.split('.')[-1] == 'npy':
        continue
    filepath = os.path.join(modir, file)
    data = np.load(filepath)[:,:139]
    data = torch.from_numpy(data)
    for idx in range(data.shape[0]):
        data_li.append(data[idx].unsqueeze(0))
data_li = torch.cat(data_li, dim=0)
data_li_ori = data_li.clone()
Normalizer_ = Normalizer(data_li)
torch.save(Normalizer_, f'data/Normalizertest_{dataset_name}.pth')

reNorm = torch.load(f'data/Normalizertest_{dataset_name}.pth')
data_newnormed = reNorm.normalize(data_li)
data_newunnormed = reNorm.unnormalize(data_newnormed)
print(data_newnormed[0,:20])
print(data_newunnormed[0,:20])
print(data_li_ori[0,:20])