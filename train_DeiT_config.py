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
from networks.transformers import *

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet18', help='model name')
    parser.add_argument('--dataset', type=str, default='UCM', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=12, help='num workers')
    parser.add_argument('--width', type=int, default=256, help='width')
    parser.add_argument('--height', type=int, default=256, help='height')
    parser.add_argument('--epochs', type=int, default=80, help='epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')

    parser.add_argument('--resume', type=str, default=False, help='resume path')
    # parser.add_argument('--pretrain', type=str, default='', help='pretrain path')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='save path')
    parser.add_argument('--log_path', type=str, default='logs', help='log path')
    parser.add_argument('--seed', type=int, default=1000, help='seed')

    return parser


def train(args):
    # model
    if config.model == 'ResNet18':
        backbone = models.resnet18(pretrained=True)
        model = ResNet18(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet34':
        backbone = models.resnet34(pretrained=True)
        model = ResNet34(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet50':
        backbone = models.resnet50(pretrained=True)
        model = ResNet50(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet101':
        backbone = models.resnet101(pretrained=True)
        model = ResNet101(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet152':
        backbone = models.resnet152(pretrained=True)
        model = ResNet152(backbone, num_classes=config.num_classes)
    elif config.model == 'DeiT_small':
        backbone = deit_small(pretrained=True,num_classes=config.num_classes)
        model = backbone
    elif config.model == 'MIL_DeiT':
        backbone = MIL_deit_small(pretrained=True,num_classes=config.num_classes)
        model = backbone
    else:
        print('ERROR: No model {}!!!'.format(config.model))
    print(model)
    # model = torch.nn.DataParallel(model)
    model.cuda()
    
    # freeze layers
    if "ResNet" in config.model:
        if config.freeze:
            for p in model.backbone.layer1.parameters(): p.requires_grad = False
            for p in model.backbone.layer2.parameters(): p.requires_grad = False
            for p in model.backbone.layer3.parameters(): p.requires_grad = False
            for p in model.backbone.layer4.parameters(): p.requires_grad = False
    else:
        if config.freeze:
            for name, para in model.named_parameters():
                # 除head, pre_logits外，其他权重全部冻结
                if "head" not in name and "pre_logits" not in name and "MIL" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))



    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    # train data
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(0.05, 0.05, 0.05),
                                    transforms.RandomRotation(10),
                                    # transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_train = RSDataset('./UCM_6_4/train.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)

    # validation data
    transform = transforms.Compose([transforms.Resize((config.width, config.height)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_valid = RSDataset('./UCM_6_4/valid.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers,pin_memory=True)

    # log
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log = open('./log/log.txt', 'a')

    log.write('-'*30+'\n')
    log.write('model:{}\nnum_classes:{}\nnum_epoch:{}\nlearning_rate:{}\nim_width:{}\nim_height:{}\niter_smooth:{}\n'.format(
               config.model, config.num_classes, config.num_epochs, config.lr, 
               config.width, config.height, config.iter_smooth))

    # load checkpoint
    if config.resume:
        model = torch.load(os.path.join('./checkpoints', config.checkpoint))

    # train
    sum = 0
    train_loss_sum = 0
    train_top1_sum = 0
    max_val_acc = 0
    train_draw_acc = []
    val_draw_acc = []

    lr = config.lr

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=config.lr, momentum=0.9, weight_decay=5E-5)
    optimizer = optim.SGD(pg, lr=config.lr, momentum=0.9, weight_decay=1E-6)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / config.num_epochs)) / 2) * (1 - config.lrf) + config.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=config.lrf)

    for epoch in range(config.num_epochs):
        ep_start = time.time()

        # # adjust lr
        # lr = step_lr(epoch)

        # # optimizer
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
        #                              lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)



        model.train()
        top1_sum = 0
        for i, (ims, label) in enumerate(dataloader_train):
            input = Variable(ims).cuda()
            target = Variable(label).cuda().long()

            if 'MIL' in config.model:
                outputs_class, outputs_MIL = model(input)
                loss1 = criterion(outputs_class, target)
                loss2 = criterion(outputs_MIL, target)
                loss = 0.5*loss1 + 0.5*loss2
                outputs = outputs_class
            else:
                outputs = model(input)
                loss = criterion(outputs, target)

            # output = model(input)
            
            # loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            top1 = accuracy(outputs.data, target.data, topk=(1,))
            train_loss_sum += loss.data.cpu().numpy()
            train_top1_sum += top1[0]
            sum += 1
            top1_sum += top1[0]
            lr=optimizer.param_groups[0]["lr"]
            if (i+1) % config.iter_smooth == 0:
                print('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f'
                       %(epoch+1, config.num_epochs, i+1, len(dst_train)//config.batch_size, 
                       lr, train_loss_sum/sum, train_top1_sum/sum))
                log.write('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f\n'
                           %(epoch+1, config.num_epochs, i+1, len(dst_train)//config.batch_size, 
                           lr, train_loss_sum/sum, train_top1_sum/sum))
                sum = 0
                train_loss_sum = 0
                train_top1_sum = 0

        train_draw_acc.append(top1_sum/len(dataloader_train))
        
        epoch_time = (time.time() - ep_start) / 60.
        if epoch % 1 == 0 and epoch < config.num_epochs:
            # eval
            val_time_start = time.time()
            val_loss, val_top1 = eval(model, dataloader_valid, criterion)
            val_draw_acc.append(val_top1)
            val_time = (time.time() - val_time_start) / 60.

            print('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s'
                   %(epoch+1, config.num_epochs, val_loss, val_top1, val_time*60))
            print('epoch time: {}s'.format(epoch_time*60))
            if val_top1[0].data > max_val_acc:
                max_val_acc = val_top1[0].data
                print('Taking snapshot...')
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save(model, '{}/{}.pth'.format('checkpoints', config.save_model))

            log.write('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s\n'
                       %(epoch+1, config.num_epochs, val_loss, val_top1, val_time*60))
        train_draw_acc_np = [tensor.cpu().numpy() for tensor in train_draw_acc]
        val_draw_acc_np = [tensor.cpu().numpy() for tensor in val_draw_acc]
        draw_curve(train_draw_acc_np, val_draw_acc_np)
        # draw_curve(train_draw_acc, val_draw_acc)
    log.write('-'*30+'\n')
    log.close()

# validation
def eval(model, dataloader_valid, criterion):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    model.eval()
    for ims, label in dataloader_valid:
        input_val = Variable(ims).cuda()
        target_val = Variable(label).cuda()
        output_val = model(input_val)
        loss = criterion(output_val, target_val)
        top1_val = accuracy(output_val.data, target_val.data, topk=(1,))
        
        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1_val[0]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    return avg_loss, avg_top1

if __name__ == '__main__':
    parse = argparse.ArgumentParser("train",parents=[get_args])
    args = parse.parse_args()
    train(args)
