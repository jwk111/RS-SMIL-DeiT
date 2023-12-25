import sys
sys.path.append('/home/jwk/Project/remoteSensing/RS-SMIL-DeiT')

import os
import random
from config import config
from tqdm import tqdm

random.seed(config.seed)



ucm_root = '/home/jwk/Project/remoteSensing/datasets/UCMerced_LandUse/Images'
nwpu_root = '/home/jwk/Project/remoteSensing/datasets/NWPU-RESISC45'
aid_root = '/home/jwk/Project/remoteSensing/datasets/AID'
# 修改以下3行
data_path = 'dataset/AID_2_8'
config.data_root = aid_root
train_ratio = 0.2

if not os.path.exists(data_path):
    os.makedirs(data_path)

train_txt = open(os.path.join(data_path,'train.txt') , 'w')
val_txt = open(os.path.join(data_path,'valid.txt') , 'w')
label_txt = open(os.path.join(data_path,'label_list.txt') , 'w')

label_list = []

for dir in tqdm(os.listdir(config.data_root)):
    if dir not in label_list:
        label_list.append(dir)
        label_txt.write('{} {}\n'.format(dir, str(len(label_list)-1)))
        data_path = os.path.join(config.data_root, dir)
        train_list = random.sample(os.listdir(data_path), 
                                   int(len(os.listdir(data_path))*train_ratio))
        for im in train_list:
            train_txt.write('{}/{} {}\n'.format(dir, im, str(len(label_list)-1)))
        for im in os.listdir(data_path):
            if im in train_list:
                continue
            else:
                val_txt.write('{}/{} {}\n'.format(dir, im, str(len(label_list)-1)))



