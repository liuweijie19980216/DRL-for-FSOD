#!/usr/bin/env bash

base_dir="save_models/VOC_first"
# base class training
CUDA_VISIBLE_DEVICES=0 python train.py --dataset pascal_voc_0712 \
--epochs 21 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--meta_type 1 --meta_train True --meta_loss True
