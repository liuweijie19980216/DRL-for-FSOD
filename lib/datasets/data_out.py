# --------------------------------------------------------
# Pytorch Meta R-CNN
# Written by Anny Xu, Xiaopeng Yan, based on the code from Jianwei Yang
# --------------------------------------------------------
import os
import os.path
import sys
import torch.utils.data as data
import cv2
import torch
import random
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from model.utils.config import cfg
import collections
import pickle


if __name__ == '__main__':
    bboxes = []
    labels = []
    gt_bboxes = {'bboxes':bboxes,
                  'labels':labels}

    root = '/home/liuwj/Repository/FewShotDetection/data/VOCdevkit'
    image_set = [('2007', 'trainval'), ('2012', 'trainval')]
    img_size = 32
    metaclass = ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair', 'diningtable', 'dog', 'horse',
                      'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']

    shuffle = True
    _annopath = os.path.join('%s', 'Annotations', '%s.xml')
    _imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')

    ids = list()
    for (year, name) in image_set:
        _year = year
        rootpath = os.path.join(root, 'VOC' + year)
        for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
            ids.append((rootpath, line.strip()))

    class_to_idx = dict(zip(metaclass, range(len(metaclass))))  # class to index mapping
    if shuffle:
        random.shuffle(ids)


    for img_id in ids:
        target = ET.parse(_annopath % img_id).getroot()
        img = cv2.imread(_imgpath % img_id, cv2.IMREAD_COLOR)
        cv2.imwrite('origin.jpg', img)
        img = img.astype(np.float32, copy=False)
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if difficult:
                continue
            name = obj.find('name').text.strip()
            if name not in metaclass:
                continue

            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                bndbox.append(cur_pt)
            x1 = bndbox[0]
            y1 = bndbox[1]
            x2 = bndbox[2]
            y2 = bndbox[3]
            bbox = img[y1:y2, x1:x2, :]  # h,w,c
            cv2.imwrite('crop.jpg', bbox)
            bbox_resize = cv2.resize(bbox, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite('resize.jpg', bbox_resize)
            bbox_resize -= cfg.PIXEL_MEANS
            bbox_out = bbox_resize.transpose(2,0,1)
            label = class_to_idx[name]
            #cv2.imwrite('resize.jpg', bbox_resize)
            break

        gt_bboxes['bboxes'].append(bbox_out)
        gt_bboxes['labels'].append(label)

    with open('base_class_gtbbox32.pkl', 'wb') as f:
        pickle.dump(gt_bboxes, f, pickle.HIGHEST_PROTOCOL)
    print('save ' + str(len(gt_bboxes['bboxes'])) + ' base class gtbbox done!')

