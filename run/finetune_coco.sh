#!/usr/bin/env bash

base_dir="save_models/COCO"


CUDA_VISIBLE_DEVICES=0 python train.py --dataset coco \
--epochs 30 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--r True --checksession 200 --checkepoch 20 \
--meta_type 0 --shots 10 --phase 2 --meta_train True --meta_loss True --temperture 1

CUDA_VISIBLE_DEVICES=0 python train.py --dataset coco \
--epochs 30 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--r True --checksession 200 --checkepoch 20 \
--meta_type 0 --shots 30 --phase 2 --meta_train True --meta_loss True --temperture 5




