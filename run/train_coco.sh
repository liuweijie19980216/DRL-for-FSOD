#!/usr/bin/env bash

base_dir="save_models/COCO"

# base class training
CUDA_VISIBLE_DEVICES=0 python train.py --dataset coco \
--epochs 21 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--meta_type 0 --meta_train True --meta_loss True
