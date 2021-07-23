# --------------------------------------------------------
# Pytorch Meta R-CNN
# Written by Anny Xu, Xiaopeng Yan, based on the code from Jianwei Yang
# --------------------------------------------------------
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import model.faster_rcnn.dygcn as dygcn


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, n_classes, class_agnostic, meta_train, meta_test=None, meta_loss=None):
        super(_fasterRCNN, self).__init__()

        self.n_classes = n_classes

        self.class_agnostic = class_agnostic
        self.meta_train = meta_train
        self.meta_test = meta_test
        self.meta_loss = False

        self.DRL = True
        if self.DRL:
            self.dynamic_gcn = dygcn.DynamicGCN(self.n_classes)
            self.criterion = nn.NLLLoss()
            self.tempe = cfg.temperture
            print(f"temperture=={self.tempe}")
        self.roi_save = False

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data_list, im_info_list, gt_boxes_list, num_boxes_list,
                average_shot=None, mean_class_attentions=None, mean_features=None):
        # return attentions for testing
        if average_shot:
            prn_data = im_data_list[0]  # len(metaclass)*4*224*224
            attentions, prototypes = self.prn_network(prn_data)
            return attentions, prototypes
        # extract attentions for training
        if self.meta_train and self.training:
            prn_data = im_data_list[0]  # len(metaclass)*4*224*224
            # feed prn data to prn_network
            attentions, prototypes = self.prn_network(prn_data)
            prn_cls = im_info_list[0]  # len(metaclass)

        im_data = im_data_list[-1]
        im_info = im_info_list[-1]
        gt_boxes = gt_boxes_list[-1]
        num_boxes = num_boxes_list[-1]

        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(self.rcnn_conv1(im_data))

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        # roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
        # rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
        # rois_label = Variable(rois_label.view(-1).long())
        # rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        # rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        # rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

        rois = Variable(rois)

        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'crop':
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))  # (b*128)*1024*7*7
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)  # (b*128)*2048
        # 提取特征
        if self.roi_save:
            return pooled_feat, rois_label

        # meta training phase
        if self.meta_train:
            rcnn_loss_cls = []
            rcnn_loss_bbox = []
            rcnn_loss_gcn = []

            # pooled feature maps need to operate channel-wise multiplication with
            # the corresponding class's attentions of every roi of image
            for b in range(batch_size):
                zero = Variable(torch.FloatTensor([0]).cuda())
                proposal_labels = rois_label[b * 128:(b + 1) * 128].data.cpu().numpy()[0]
                unique_labels = list(np.unique(proposal_labels))  # the unique rois labels of the input image

                for i in range(attentions.size(0)):  # attentions len(attentions)*2048
                    # 类别0表示背景
                    if prn_cls[i].numpy()[0] + 1 not in unique_labels:
                        rcnn_loss_cls.append(zero)
                        rcnn_loss_bbox.append(zero)
                        continue

                    roi_feat = pooled_feat[b * cfg.TRAIN.BATCH_SIZE:(b + 1) * cfg.TRAIN.BATCH_SIZE, :]  # 128*2048
                    cls_feat = attentions[i].view(1, -1, 1, 1)  # 1*2048*1*1

                    diff_feat = roi_feat - cls_feat.squeeze()
                    corr_feat = F.conv2d(roi_feat.unsqueeze(-1).unsqueeze(-1),
                                         cls_feat.permute(1, 0, 2, 3),
                                         groups=2048).squeeze()

                    # # subtraction + correlation: [bs, 2048]
                    channel_wise_feat = torch.cat((self.corr_fc(corr_feat), self.diff_fc(diff_feat)), dim=1)

                    #combined with the roi feature: [bs, 2048 * 2]
                    channel_wise_feat = torch.cat((channel_wise_feat, roi_feat), dim=1)

                    #compute object bounding box regression
                    bbox_pred = self.RCNN_bbox_pred(channel_wise_feat)  # 128*4
                    #bbox_pred = self.RCNN_bbox_pred(corr_feat)
                    if self.training and not self.class_agnostic:
                        # select the corresponding columns according to roi labels
                        bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                        batch_rois_label = rois_label[b * cfg.TRAIN.BATCH_SIZE:(b + 1) * cfg.TRAIN.BATCH_SIZE]
                        bbox_pred_select = torch.gather(
                            bbox_pred_view, 1, batch_rois_label.view(
                                batch_rois_label.size(0), 1, 1).expand(batch_rois_label.size(0), 1, 4))
                        bbox_pred = bbox_pred_select.squeeze(1)

                    # compute object classification probability
                    cls_score = self.RCNN_cls_score(channel_wise_feat)
                    #cls_score = self.RCNN_cls_score(corr_feat)
                    ###############################################################3
                    # pred_logit = F.softmax(cls_score)
                    # pred = torch.argmax(pred_logit, dim=1)
                    # print(pred)
                    ########################################################
                    if self.training:
                        # classification loss
                        RCNN_loss_cls = F.cross_entropy(cls_score, rois_label[b * 128:(b + 1) * 128])
                        if self.DRL:
                            # support set中所有的类别
                            all_support_labels = [cls[0] + 1 for cls in prn_cls]
                            # 先取出与roi类别一致的类，在随机取N类
                            #support_labels, support_ids = self.sample_support_labels(all_support_labels, unique_labels)
                            support_labels = all_support_labels
                            support_ids = range(len(all_support_labels))
                            # 合并成tensor
                            support_labels = torch.tensor(support_labels).cuda()
                            #support_labels = torch.cat([label for label in support_labels], dim=0).cuda()
                            query_labels = rois_label[b * 128:(b + 1) * 128]
                            cls_ids = self.get_cls_id(labels=query_labels, all_rois=False)
                            group_labels = torch.cat([support_labels, query_labels[cls_ids]], dim=0)
                            gcn_input = torch.cat([prototypes[support_ids], roi_feat[cls_ids]], dim=0)
                            atten_prob = torch.zeros(len(support_ids), self.n_classes).cuda()

                            rois_prob = F.softmax(cls_score / self.tempe).squeeze(1)[cls_ids]
                            prob_for_gcn = torch.cat([atten_prob, rois_prob], dim=0)

                            out_for_gcn = self.dynamic_gcn(gcn_input, len(support_ids), support_labels, prob_for_gcn)
                            probs_log = torch.log(out_for_gcn + 1e-12)
                            #probs_for_gtg = probs_for_gtg[len(attentions):]
                            gcn_loss = self.criterion(probs_log, group_labels)
                            assert not torch.isnan(gcn_loss), 'gcn_loss is nan!'
                            rcnn_loss_gcn.append(gcn_loss)

                        rcnn_loss_cls.append(RCNN_loss_cls)
                        # bounding box regression L1 loss
                        RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target[b * 128:(b + 1) * 128],
                                                         rois_inside_ws[b * 128:(b + 1) * 128],
                                                         rois_outside_ws[b * 128:(b + 1) * 128])

                        rcnn_loss_bbox.append(RCNN_loss_bbox)

            # meta attentions loss
            if self.meta_loss:
                attentions_score = self.Meta_cls_score(attentions)
                meta_loss = F.cross_entropy(attentions_score, Variable(torch.cat(prn_cls, dim=0).cuda()))
            else:
                meta_loss = 0

            return rois, rpn_loss_cls, rpn_loss_bbox, rcnn_loss_cls, rcnn_loss_bbox, rois_label, 0, 0, meta_loss, rcnn_loss_gcn

        # meta testing phase
        elif self.meta_test:
            cls_prob_list = []
            bbox_pred_list = []
            cls_score_list = []
            probs_gcn_list = []
            for i in range(len(mean_class_attentions)):
                mean_attentions = mean_class_attentions[i]
                cls_feat = mean_attentions.view(1, -1, 1, 1)  # 1*2048*1*1
                diff_feat = pooled_feat - cls_feat.squeeze()
                corr_feat = F.conv2d(pooled_feat.unsqueeze(-1).unsqueeze(-1),
                                     cls_feat.permute(1, 0, 2, 3),
                                     groups=2048).squeeze()
                # # subtraction + correlation: [bs, 2048]
                channel_wise_feat = torch.cat((self.corr_fc(corr_feat), self.diff_fc(diff_feat)), dim=1)
                # # combined with the roi feature: [bs, 2048 * 2]
                channel_wise_feat = torch.cat((channel_wise_feat, pooled_feat), dim=1)
                # compute bbox offset
                bbox_pred = self.RCNN_bbox_pred(channel_wise_feat)
                if self.training and not self.class_agnostic:
                    # select the corresponding columns according to roi labels
                    bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                    bbox_pred_select = torch.gather(
                        bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                    bbox_pred = bbox_pred_select.squeeze(1)

                # compute object classification probability
                cls_score = self.RCNN_cls_score(channel_wise_feat)

                #cls_score[:, -1] = cls_score[:, -1] * 1.5

                cls_prob = F.softmax(cls_score)


                RCNN_loss_cls = 0
                RCNN_loss_bbox = 0

                cls_prob = cls_prob.view(batch_size, rois.size(1), -1)

                cls_score = cls_score.view(batch_size, rois.size(1), -1)
                bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

                cls_prob_list.append(cls_prob)
                bbox_pred_list.append(bbox_pred)
                # probs_gcn_list.append(probs_gcn)
                cls_score_list.append(cls_score)


            # pred_cls = torch.argmax(cls_prob.squeeze(), dim=1)
            # sofa_index = []
            # error_cls = []
            # for label_index, label in enumerate(rois_label):
            #     if label == 20:
            #         sofa_index.append(label_index)
            # if len(sofa_index) != 0:
            #     for index in sofa_index:
            #         if pred_cls[index] != 20:
            #             error_cls.append(pred_cls[index].item())
            #     print('sofas are error classified:', error_cls)

            # if rois_label[0] == 20:
            #     print('label:', rois_label[:20])
            #     print('pred:', pred_cls[:20])

            return rois, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, cls_prob_list, bbox_pred_list, 0

        # original faster-rcnn implementation
        else:
            bbox_pred = self.RCNN_bbox_pred(pooled_feat)
            if self.training and not self.class_agnostic:
                # select the corresponding columns according to roi labels
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(
                    bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                bbox_pred = bbox_pred_select.squeeze(1)

            # compute object classification probability
            cls_score = self.RCNN_cls_score(pooled_feat)  # 128 * 1001
            cls_prob = F.softmax(cls_score)

            RCNN_loss_cls = 0
            RCNN_loss_bbox = 0

            if self.training:
                # classification loss
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

                # bounding box regression L1 loss
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
            bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, cls_prob, bbox_pred, 0

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        def weights_normal_init(model, dev=0.001):
            if isinstance(model, list):
                for m in model:
                    weights_normal_init(m, dev)
            else:
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        m.weight.data.normal_(0.0, dev)
                    elif isinstance(m, nn.Linear):
                        m.weight.data.normal_(0.0, dev)
                    elif isinstance(m, torch.nn.BatchNorm1d):
                        m.weight.data.normal_(1.0, 0.02)
                        m.bias.data.fill_(0)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

        weights_normal_init(self.corr_fc)
        weights_normal_init(self.diff_fc)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    # proposal选择一类对象作为前景，和与其相同数量的背景proposals参与训练
    def get_cls_id(self, labels, all_rois=False):
        pg_id, bg_id = [], []
        for i in range(len(labels)):
            if labels[i] == 0:
                bg_id.append(i)
            else:
                pg_id.append(i)
        if not all_rois:
            bg_id = bg_id[:len(pg_id)]
        cls_ids = pg_id + bg_id
        return cls_ids

    def sample_support_labels(self, all_support_labels, unique_labels):
        support_labels = []
        support_labels_id = []
        rest_support_labels = []
        for i in all_support_labels:
            if i.numpy() in unique_labels:
                support_labels.append(i)
            else:
                rest_support_labels.append(i)
        random_labels = random.sample(rest_support_labels, 20 - len(support_labels))
        support_labels = support_labels + random_labels
        for i in support_labels:
            support_labels_id.append(all_support_labels.index(i))
        return support_labels, support_labels_id

    # proposal不作为anchor 取56个用于分类
    # def get_labeled_and_unlabeled_points(self, labels, num_points_per_class, num_classes=21):
    #     anchor_labs, anchor_ids, cls_ids = [], [], []
    #     num_cls = 0
    #     labs_buffer = np.zeros(num_classes)
    #     labs_buffer[0] = 1
    #     num_points = labels.shape[0]
    #     for i in range(num_points):
    #         if labs_buffer[labels[i]] == num_points_per_class:
    #             if num_cls < 56:
    #                 cls_ids.append(i)
    #                 num_cls += 1
    #             else:
    #                 break
    #         else:
    #             anchor_ids.append(i)
    #             anchor_labs.append(labels[i])
    #             labs_buffer[labels[i]] += 1
    #     return anchor_labs, anchor_ids, cls_ids



