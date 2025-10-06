import argparse
import copy
import gc
import numpy as np
import h5py
import os
import random
from tqdm import tqdm, trange
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import timeit
from PIL import Image
import sys
from datetime import datetime 
import matlab.engine
import scipy.io as sio
import cv2
import tempfile
import concurrent.futures

from dataset import LoadDataSet

## settings
parser = argparse.ArgumentParser()

parser.add_argument('--data_name', type=str, default='RC-49', choices=["RC-49", "RC-49_imb", "UTKFace", "SteeringAngle", "Cell200"]) #"Cell200_imb" "Cell200"
parser.add_argument('--imb_type', type=str, default='unimodal', choices=['unimodal', 'dualmodal', 'trimodal', 'standard', 'none']) #none means using all data
parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--seed', type=int, default=2025, metavar='S', help='random seed')
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--min_label', type=float, default=0.0)
parser.add_argument('--max_label', type=float, default=90.0)
parser.add_argument('--num_channels', type=int, default=3, metavar='N')
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--max_num_img_per_label', type=int, default=2**20, metavar='N')
parser.add_argument('--num_img_per_label_after_replica', type=int, default=0, metavar='N')

args = parser.parse_args()


# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

## Setup NIQE parameters
if args.data_name in ["RC-49", "RC-49_imb"]:
    if args.img_size==64:
        block_size = 8
    elif args.img_size==128:
        block_size = 16
    elif args.img_size==256:
        block_size = 32
elif args.data_name == "UTKFace":
    if args.img_size==64:
        block_size = 8
    elif args.img_size==128:
        block_size = 16
    elif args.img_size==192:
        block_size = 24
elif args.data_name == "SteeringAngle":
    if args.img_size==64:
        block_size = 8
    elif args.img_size==128:
        block_size = 32
elif args.data_name == "Cell200":
    if args.img_size==64:
        block_size = 8
sharpness_threshold = 0.1

## output path
path_to_output = os.path.join(args.root_path, 'niqe_models/{}'.format(args.data_name))
os.makedirs(path_to_output, exist_ok=True)

## dataset
dataset = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label, max_label=args.max_label, img_size=args.img_size, max_num_img_per_label=args.max_num_img_per_label, num_img_per_label_after_replica=args.num_img_per_label_after_replica, imbalance_type=args.imb_type)
train_images, train_labels, train_labels_norm = dataset.load_train_data()
unique_train_labels, counts_train_elements = np.unique(train_labels, return_counts=True) 


## training niqe model on the whole training set
if args.data_name == "RC-49_imb":
    path_to_model = os.path.join(path_to_output, "niqe_model_{}_{}_{}.mat".format(args.data_name, args.imb_type, args.img_size))
else:
    path_to_model = os.path.join(path_to_output, "niqe_model_{}_{}.mat".format(args.data_name, args.img_size))

def train_niqe_model(X, block_size, sharpness_threshold, output_mat):
    eng = matlab.engine.start_matlab()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the image to a temporary directory
        file_paths = []
        for i in range(X.shape[0]):
            img = np.transpose(X[i], (1, 2, 0))  # 3×64×64 → 64×64×3
            filename = os.path.join(temp_dir, f"train_{i}.png")
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            file_paths.append(filename)
        # Call MATLAB's cellstr function via the engine
        mat_paths = eng.cellstr(file_paths)  
        # create ImageDatastore
        eng.workspace['filePaths'] = mat_paths
        eng.eval("imds = imageDatastore(filePaths);", nargout=0)
        # train model
        eng.eval(f"model = fitniqe(imds, 'BlockSize', {block_size}, 'SharpnessThreshold', {sharpness_threshold});", nargout=0)
        eng.save(output_mat, 'model', nargout=0)
    eng.quit()

# if args.train_with_label:
#     path_to_model_dir = os.path.join(path_to_output, "niqe_model_{}_{}".format(args.data_name, args.img_size))
#     os.makedirs(path_to_model_dir, exist_ok=True)
    

train_niqe_model(train_images, block_size=[block_size, block_size], sharpness_threshold=sharpness_threshold, output_mat = path_to_model)











