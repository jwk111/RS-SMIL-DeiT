#! /bin/bash
# python train_DeiT.py --model DeiT_small --dataset UCM --num_epochs 100 --save_dir DeiT_small 

# python train_DeiT.py --model MIL_DeiT --dataset UCM --num_epochs 100 --save_dir MIL_DeiT_DINO --ssp_path '/home/jwk/Project/remoteSensing/RS-SMIL-DeiT/checkpoints/dino_deitsmall16_pretrain.pth'

# python train_DeiT.py --model DeiT_small --dataset UCM --num_epochs 100 --save_dir DeiT_small_DINO --ssp_path '/home/jwk/Project/remoteSensing/RS-SMIL-DeiT/checkpoints/dino_deitsmall16_pretrain.pth'


###############################################################
#####             AID
###############################################################

## resnet50 baseline
# python train.py --model ResNet50 --dataset AID --num_epochs 100 --save_dir ResNet50

## DeiT_small baseline
# python train.py --model DeiT_small --dataset AID --num_epochs 100 --save_dir DeiT_small

## Deit_small + MIL
# python train.py --model MIL_DeiT --dataset AID --num_epochs 100 --save_dir MIL_DeiT

## Deit_small + DINO
# python train.py --model DeiT_small --dataset AID --num_epochs 100 --save_dir DeiT_small_DINO --ssp_path '/home/jwk/Project/remoteSensing/RS-SMIL-DeiT/checkpoints/dino_deitsmall16_pretrain.pth'

## Deit_small + MIL + DINO
# python train.py --model MIL_DeiT --dataset AID --num_epochs 100 --save_dir MIL_DeiT_DINO --ssp_path '/home/jwk/Project/remoteSensing/RS-SMIL-DeiT/checkpoints/dino_deitsmall16_pretrain.pth'

###############################################################
#####             NWPU 2_8
###############################################################

## resnet50 baseline
# python train.py --model ResNet50 --dataset NWPU28 --num_epochs 100 --save_dir ResNet50

## DeiT_small baseline
# python train.py --model DeiT_small --dataset NWPU28 --num_epochs 100 --save_dir DeiT_small

## Deit_small + MIL
# python train.py --model MIL_DeiT --dataset NWPU28 --num_epochs 100 --save_dir MIL_DeiT

## Deit_small + DINO
# python train.py --model DeiT_small --dataset NWPU28 --num_epochs 100 --save_dir DeiT_small_DINO --ssp_path '/home/jwk/Project/remoteSensing/RS-SMIL-DeiT/checkpoints/dino_deitsmall16_pretrain.pth'

# ## Deit_small + MIL + DINO
# python train.py --model MIL_DeiT --dataset NWPU28 --num_epochs 100 --save_dir MIL_DeiT_DINO --ssp_path '/home/jwk/Project/remoteSensing/RS-SMIL-DeiT/checkpoints/dino_deitsmall16_pretrain.pth'

###############################################################
#####             NWPU 8_2
###############################################################

# ## resnet50 baseline
# python train.py --model ResNet50 --dataset NWPU82 --num_epochs 100 --save_dir ResNet50

# ## DeiT_small baseline
# python train.py --model DeiT_small --dataset NWPU82 --num_epochs 100 --save_dir DeiT_small

# ## Deit_small + MIL
# python train.py --model MIL_DeiT --dataset NWPU82 --num_epochs 100 --save_dir MIL_DeiT

# ## Deit_small + DINO
# python train.py --model DeiT_small --dataset NWPU82 --num_epochs 100 --save_dir DeiT_small_DINO --ssp_path '/home/jwk/Project/remoteSensing/RS-SMIL-DeiT/checkpoints/dino_deitsmall16_pretrain.pth'

# ## Deit_small + MIL + DINO
# python train.py --model MIL_DeiT --dataset NWPU82 --num_epochs 100 --save_dir MIL_DeiT_DINO --ssp_path '/home/jwk/Project/remoteSensing/RS-SMIL-DeiT/checkpoints/dino_deitsmall16_pretrain.pth'


###########################################
###  DEIT-MIL-DINO (pretrain)300epochs

# python train.py --model MIL_DeiT --dataset UCM --num_epochs 100 --save_dir MIL_DeiT_DINO --ssp_path '/home/jwk/Project/remoteSensing/dino/train_log/UCM_deit_dino/checkpoint0280.pth'

# python train.py --model MIL_DeiT --dataset AID --num_epochs 100 --save_dir MIL_DeiT_DINO --ssp_path '/home/jwk/Project/remoteSensing/dino/train_log/AID_deit_dino/checkpoint0220.pth'

python train.py --model MIL_DeiT --dataset NWPU28 --num_epochs 100 --save_dir MIL_DeiT_DINO --ssp_path '/home/jwk/Project/remoteSensing/dino/train_log/NWPU_deit_dino/checkpoint.pth'

python train.py --model MIL_DeiT --dataset NWPU82 --num_epochs 100 --save_dir MIL_DeiT_DINO --ssp_path '/home/jwk/Project/remoteSensing/dino/train_log/NWPU_deit_dino/checkpoint.pth'
