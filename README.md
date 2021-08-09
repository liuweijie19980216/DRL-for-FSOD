
﻿# Dynamic Relevance Learning for Few-Shot Object Detection

(arXiv) PyTorch implementation of paper "Dynamic Relevance Learning for Few-Shot Object Detection"
[\[PDF\]](https://arxiv.org/abs/2108.02235)

<p align="center">
<img src="https://github.com/liuweijie19980216/DRL-for-FSOD/blob/master/imgs/figure2.png" width="800px" alt="teaser">
</p>




## Tabel of Contents
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Getting Started](#getting-started)


## Installation

Code built on top of [Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild](https://github.com/YoungXIAO13/FewShotDetection).
 
**Requirements**

* CUDA 9.0
* Python=3.6
* PyTorch=0.4.0
* torchvision=0.2.1
* gcc >= 4.9 

**Build**

Create conda env:
```sh
conda create --name FSdetection --file spec-file.txt
conda activate FSdetection
```

Compile the CUDA dependencies:
```sh
cd {repo_root}/lib
sh make.sh
```

## Data Preparation

We evaluate our method on two commonly-used benchmarks. Detailed data preparation commands can be found in [data/README.md](https://github.com/liuweijie19980216/DRL-for-FSOD/tree/master/data/README.md)

### PASCAL VOC
 
We use the train/val sets of PASCAL VOC 2007+2012 for training and the test set of PASCAL VOC 2007 for evaluation. 
We split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 splits proposed in [FSRW](https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/data/datasets/builtin_meta.py). 

Download [PASCAL VOC 2007+2012](http://host.robots.ox.ac.uk/pascal/VOC/), create softlink named ``VOCdevkit`` in the folder ``data/``.


### COCO

We use COCO 2014 and keep the 5k images from minival set for evaluation and use the rest for training. 
We use the 20 object classes that are the same with PASCAL VOC as novel classes and use the rest as base classes.

Download [COCO 2014](https://cocodataset.org/#home), create softlink named ``coco`` in the folder ``data/``.


## Getting Started

We provide pre-trained models of 10-shot setting on COCO.
```bash
wget https://www.dropbox.com/s/l04vfuaf3ir6410/save_models.zip?dl=0 && mv save_models.zip?dl=0 save_models.zip

unzip save_models.zip && rm save_models.zip
```
You will get a dir like:
```
save_models/
    COCO/
```
### Base-Class Training
**Pre-trained ResNet**:
we used [ResNet101](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0) pretrained model on ImageNet in our experiments. 
Download it and put it into the ``data/pretrained_model/``.

Base training on base classes with sufficient samples:
```bash
# the first split on VOC
bash run/train_voc_first.sh

# the second split on VOC
bash run/train_voc_second.sh

# the third split on VOC
bash run/train_voc_third.sh

# NonVOC / VOC split on COCO
bash run/train_coco.sh
```

### Few-Shot Fine-tuning

Fine-tune the base-training models on a balanced training data including both base and novel classes:
```bash
bash run/finetune_voc_first.sh

bash run/finetune_voc_second.sh

bash run/finetune_voc_third.sh

bash run/finetune_coco.sh
```


### Testing

Evaluation is conducted on the test set of PASCAL VOC 2007 or minival set of COCO 2014:
```bash
bash run/test_voc_first.sh

bash run/test_voc_second.sh

bash run/test_voc_third.sh

bash run/test_coco.sh
```



## Quantitative Results

### Multiple Runs

By running multiple times (~10) the few-shot fine-tuning experiments and averaging the results, we got the performance below:

**Pascal-VOC Novel Set 1 (AP@50)**

|           | shot=1| shot=2  | shot=3 | shot=5 | shot=10 |
| :------: | :------:       | :------:        | :------:       | :------:        | :------:        |
| [Meta RCNN](https://github.com/yanxp/MetaR-CNN)       |  19.9     |   25.5     |   35.0         |   45.7| 51.5|
| [FSDetView](https://github.com/YoungXIAO13/FewShotDetection)      |  24.2         |   35.3         |   42.2         | 49.1|  57.4|
| DRL(normal)      |  30.3        |   40.8          |   49.1        |    48.0|58.6|
|  DRL(residual)       |  28.0         |   40.5         |   49.4        |    49.9|59.4|


The detailed experimental results seen the [Paper](https://arxiv.org/abs/2108.02235)

If our project is helpful for your research, please consider citing:
```
@INPROCEEDINGS{Liu2021DRLFsod,
    author    = {Weijie Liu, Chong Wang*, Haohe Li, Shenghao Yu, Song Chen, Xulun Ye and Jiafei Wu},
    title     = {Dynamic Relevance Learning for Few-Shot Object Detection},
    booktitle = {arXiv Preprint},
    year      = {2021}}
```
