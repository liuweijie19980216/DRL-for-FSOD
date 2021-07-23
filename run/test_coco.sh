#!/usr/bin/env bash

base_dir="save_models/COCO"

# testing on base and novel class
CUDA_VISIBLE_DEVICES=0 python test.py --dataset coco \
--load_dir $base_dir  --meta_type 0 \
--checksession 10 --checkepoch 29 --shots 10 \
--phase 2 --meta_test True --meta_loss True --temperture 1

CUDA_VISIBLE_DEVICES=0 python test.py --dataset coco \
--load_dir $base_dir  --meta_type 0 \
--checksession 30 --checkepoch 29 --shots 30 \
--phase 2 --meta_test True --meta_loss True --temperture 5