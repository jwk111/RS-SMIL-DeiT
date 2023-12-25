# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/jwk/Project/remoteSensing/RS-SMIL-DeiT')
import os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.nn.parallel.data_parallel import data_parallel
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset.dataset import *
from networks.network import *
from networks.lr_schedule import *
from metrics.metric import *
from utils.plot import *
from config import config

import argparse

def get_args():
    parser = argparse.ArgumentParser('inference', add_help=False)
    parser.add_argument('--model', type=str, default='DeiT_small', help='model name')
    parser.add_argument('--dataset', type=str, default='UCM', help='dataset name',choices=['UCM','NWPU','AID'])
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=12, help='num workers')
    parser.add_argument('--num_classes', type=int, default=21, help='num classes')
    parser.add_argument('--num_epochs', type=int, default=80, help='num epochs')
    parser.add_argument('--width', type=int, default=256, help='width')
    parser.add_argument('--height', type=int, default=256, help='height')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lrf', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--resume', type=str, default=False, help='resume path')
    # parser.add_argument('--pretrain', type=str, default='', help='pretrain path')
    parser.add_argument('--freeze', type=bool, default=True, help='freeze backbone')
    parser.add_argument('--save_dir', type=str, default='DeiT_small', help='save path')
    parser.add_argument('--save_model', type=str, default='DeiT_small', help='save model name')
    parser.add_argument('--seed', type=int, default=1000, help='seed')
    parser.add_argument('--iter_smooth', type=int, default=10, help='print log frequency')
    parser.add_argument('--ssp_path', type=str, default=None, help='ssp path')
    return parser

ucm_root = '/home/jwk/Project/remoteSensing/datasets/UCMerced_LandUse/Images'
nwpu_root = '/home/jwk/Project/remoteSensing/datasets/NWPU-RESISC45'
aid_root = '/home/jwk/Project/remoteSensing/datasets/AID'

def inference(args):
    #num of classes
    if args.dataset == 'UCM':
        args.num_classes = 21
        args.data_root = ucm_root
    elif args.dataset == 'NWPU':
        args.num_classes = 45
        args.data_root = nwpu_root
    elif args.dataset == 'AID':
        args.num_classes = 30
        args.data_root = aid_root
    # model
    # load checkpoint
    pth_path = "/home/jwk/Project/remoteSensing/RS-SMIL-DeiT/log/MIL_DeiT/2023-06-09-17-55-46/checkpoints/best_model.pth"
    proj_path = '/home/jwk/Project/remoteSensing/RS-SMIL-DeiT/log/DeiT_small/2023-06-09-17-55-46'
    # model = torch.load(os.path.join('./checkpoints', args.checkpoint))
    model = torch.load(pth_path)
    print("model:",model)
    # model = torch.nn.DataParallel(model)
    model.cuda()
    
    # validation data
    transform = transforms.Compose([transforms.Resize((args.width, args.height)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_valid = RSDataset('./UCM_6_4/valid.txt', width=args.width, 
                          height=args.height, transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    sum = 0
    val_top1_sum = 0
    labels = []
    preds = []
    model.eval()
    for ims, label in dataloader_valid:
        labels += label.numpy().tolist()

        input = Variable(ims).cuda()
        target = Variable(label).cuda()
        output = model(input)

        _, pred = output.topk(1, 1, True, True)
        preds += pred.t().cpu().numpy().tolist()[0]

        top1_val = accuracy(output.data, target.data, topk=(1,))
        
        sum += 1
        val_top1_sum += top1_val[0]
    avg_top1 = val_top1_sum / sum
    print('acc: {}'.format(avg_top1.data)) 

    labels_=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]   
    plot_confusion_matrix(labels, preds, labels_,save_path=proj_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser("inference",parents=[get_args()])
    args = parse.parse_args()
    inference(args)
