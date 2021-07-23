#!/usr/bin/env bash

base_dir="save_models/VOC_second"

# number of shots
for j in 1 2 3 5 10
do
# few-shot fine-tuning
CUDA_VISIBLE_DEVICES=0 python train.py --dataset pascal_voc_0712 \
--epochs 30 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--r True --checksession 200 --checkepoch 20 \
--meta_type 2 --shots $j --phase 2 --meta_train True --meta_loss True --temperture 52

# testing on base and novel class
CUDA_VISIBLE_DEVICES=0 python test.py --dataset pascal_voc_0712 \
--load_dir $base_dir  --meta_type 2 \
--checksession $j --checkepoch 29 --shots $j \
--phase 2 --meta_test True --meta_loss True  --temperture 52
done


