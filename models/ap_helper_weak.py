# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import os
import sys
import numpy as np
import torch
import scipy.stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils.eval_det_weak import eval_det_cls, eval_det_multiprocessing
from utils.eval_det_weak import get_iou_obb, get_iou_obb_seg
from utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from utils.box_util import get_3d_box

sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from sunrgbd.sunrgbd_utils import extract_pc_in_box3d


def flip_axis_to_camera(pc):
    """ Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2


def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2


def softmax(x):
    """ Numpy function for softmax"""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def parse_predictions(end_points, config_dict):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}
    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    batch_pc = end_points['point_clouds'].cpu().numpy()[:, :, 0:3]  # B,N,3
    N = batch_pc.shape[1]

    pred_center = end_points['center']  # B,num_proposal,3
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points['size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(end_points['size_residuals'], 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points['sem_cls_scores'], -1)  # B,num_proposal
    sem_cls_probs = softmax(end_points['sem_cls_scores'].detach().cpu().numpy())  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal

    # objectness_label = end_points['objectness_label']
    # object_assignment = end_points['object_assignment']
    # sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    obj_logits = end_points['objectness_scores'].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    pred_sem_score_map = end_points['semantic_score_map']

    bsize = pred_center.shape[0]
    num_proposal = pred_center.shape[1]
    nonempty_box_mask = np.ones((bsize, num_proposal))
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())

    ins_seg_ind = np.zeros((bsize, num_proposal, N))
    # pred_mask = np.zeros((bsize, num_proposal))
    point_in_box = []
    pred_mask = np.zeros((bsize, num_proposal))  # ***************
    for i in range(bsize):
        pc = batch_pc[i, :, :]  # (N,3)
        point_in_box.append([])
        # ins_3d_with_prob = np.zeros((num_proposal,8))#******************
        for j in range(num_proposal):
            point_in_box[i].append([])
            heading_angle = config_dict['dataset_config'].class2angle(pred_heading_class[i, j].detach().cpu().numpy(),
                                                                      pred_heading_residual[
                                                                          i, j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size(int(pred_size_class[i, j].detach().cpu().numpy()),
                                                                pred_size_residual[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

            box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
            box3d = flip_axis_to_depth(box3d)
            pc_in_box, ind = extract_pc_in_box3d(pc, box3d)  #

            if len(pc_in_box) < 5:
                nonempty_box_mask[i, j] = 0
                continue

            inds = np.where(ind == True)[0]
            # pred_point_sem_score = pred_sem_score_map[i,inds,sem_cls_label[i,j]]
            pred_point_sem_score = pred_sem_score_map[i, inds, pred_sem_cls[i, j]]
            pred_point_sem_score = torch.sigmoid(pred_point_sem_score)
            pred_point_sem_score /= pred_point_sem_score.max()
            # max_prop_sem_ind = torch.argmax(pred_sem_score_map[i,inds], -1) == sem_cls_label[i,j]
            # maxk_ind = pred_point_sem_score.topk(int(occupy_ratio[sem_cls_label[i,j]]*pred_point_sem_score.shape[0]))[1].cpu().numpy()
            # maxk_ind = (max_prop_sem_ind & torch.ge(pred_point_sem_score,0.7)).nonzero().long().view(1,-1).squeeze(dim=0).cpu().numpy()
            maxk_ind = (torch.ge(pred_point_sem_score, 0.7)).nonzero().long().view(1, -1).squeeze(dim=0).cpu().numpy()
            ind_bool = np.array([False] * N)
            ind_bool[inds[maxk_ind]] = True
            ins_seg_ind[i][j] = ind_bool

        proposal = ins_seg_ind[i][nonempty_box_mask[i, :] == 1, :]
        confidence = obj_prob[i][nonempty_box_mask[i, :] == 1]
        intersection = np.matmul(proposal, proposal.T)  # (nProposal, nProposal), float
        proposals_pointnum = proposal.sum(1)  # (nProposal), float
        proposals_pn_h = np.expand_dims(proposals_pointnum, axis=1).repeat(proposals_pointnum.shape[0], 1)
        proposals_pn_v = np.expand_dims(proposals_pointnum, axis=0).repeat(proposals_pointnum.shape[0], 0)
        cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection + 1e-6)
        pick_idxs = non_max_suppression(cross_ious, confidence, config_dict['nms_iou'])  # int, (nCluster, N)
        assert (len(pick_idxs) > 0)
        nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
        pred_mask[i, nonempty_box_inds[pick_idxs]] = 1
    end_points['pred_mask'] = pred_mask

    end_points['seg_ind'] = ins_seg_ind  # used for visualization

    batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            cur_list = []
            for ii in range(config_dict['dataset_config'].num_class):
                cur_list += [(ii, ins_seg_ind[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j]) \
                             for j in range(pred_center.shape[1]) if
                             pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']]
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append([(pred_sem_cls[i, j].item(), ins_seg_ind[i, j], obj_prob[i, j]) \
                                       for j in range(pred_center.shape[1]) if
                                       pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])
    end_points['batch_pred_map_cls'] = batch_pred_map_cls

    return batch_pred_map_cls


def parse_groundtruths(end_points, config_dict):
    """ Parse groundtruth labels to OBB parameters.
    
    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}
    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    batch_pc = end_points['point_clouds'].cpu().numpy()[:, :, 0:3]  # B,N,3
    N = batch_pc.shape[1]

    # 用于生成box
    center_label = end_points['center_label']
    heading_class_label = end_points['heading_class_label']
    heading_residual_label = end_points['heading_residual_label']
    size_class_label = end_points['size_class_label']
    size_residual_label = end_points['size_residual_label']
    box_label_mask = end_points['box_label_mask']  # allocate box sec label

    # 根据box内部的 point sec label 计算 ap
    sem_cls_label = end_points['sem_cls_label']
    instance_labels = end_points['instance_labels'].detach().cpu().numpy()
    semantic_labels = end_points['semantic_labels'].detach().cpu().numpy()

    bsize = center_label.shape[0]
    K2 = center_label.shape[1]  # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:, :, 0:3].detach().cpu().numpy())
    seg_ind_gt = np.zeros((bsize, K2, N))
    for i in range(bsize):
        pc = batch_pc[i, :, :]  # (N,3)
        for j in range(K2):
            if box_label_mask[i, j] == 0: continue
            heading_angle = config_dict['dataset_config'].class2angle(heading_class_label[i, j].detach().cpu().numpy(),
                                                                      heading_residual_label[
                                                                          i, j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size(int(size_class_label[i, j].detach().cpu().numpy()),
                                                                size_residual_label[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i, j, :])
            gt_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

            box3d = gt_corners_3d_upright_camera[i, j, :, :]  # (8,3)
            box3d = flip_axis_to_depth(box3d)
            pc_in_box, ind = extract_pc_in_box3d(pc, box3d)  #

            inds = np.where(ind == True)[0]
            tmp = np.where(semantic_labels[i][ind] == sem_cls_label[i, j].item())[0]
            ind_bool = np.array([False] * N)
            ind_bool[inds[tmp]] = True
            seg_ind_gt[i, j] = ind_bool

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append(
            [(sem_cls_label[i, j].item(), seg_ind_gt[i, j]) for j in range(K2) if box_label_mask[i, j] == 1])
    end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls


class APCalculator(object):
    """ Calculating Average Precision """

    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0   IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()

    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]  should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert (bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh,
                                                 get_iou_func=get_iou_obb_seg)
        ret_dict = {}
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision' % (clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall' % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall' % (clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)

        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0
