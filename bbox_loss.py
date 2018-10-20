import math

import torch.nn as nn
import torch.nn.functional as F
import torch


# def hard_negative_mining(predicted_prob, gt_label, neg_pos_ratio=3.0):
#     """
#     The training sample has much more negative samples, the hard negative mining and produce balanced
#     positive and negative examples.
#     :param predicted_prob: predicted probability for each prior item, dim: (N, H*W*num_prior)
#     :param gt_label: ground_truth label, dim: (N, H*W*num_prior)
#     :param neg_pos_ratio:
#     :return:
#     """
#     pos_flag = gt_label > 0                                        # 0 = negative label
#
#     # Sort the negative samples
#     predicted_prob[pos_flag] = -1.0                                # temporarily remove positive by setting -1
#     _, indices = predicted_prob.sort(dim=1, descending=True)       # sort by descend order, the positives are at the end
#     _, orders = indices.sort(dim=1)                                # sort the negative samples by its original index
#
#     # Remove the extra negative samples
#     num_pos = pos_flag.sum(dim=1, keepdim=True)                     # compute the num. of positive examples
#     num_neg = neg_pos_ratio * num_pos                               # determine of neg. examples, should < neg_pos_ratio
#     neg_flag = orders < num_neg                                     # retain the first 'num_neg' negative samples index.
#
#     return pos_flag, neg_flag

def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask

class MultiboxLoss(nn.Module):

    def __init__(self, bbox_pre_var, iou_threshold=0.5, neg_pos_ratio=3.0):
        super(MultiboxLoss, self).__init__()
        self.bbox_center_var, self.bbox_size_var = bbox_pre_var[:2], bbox_pre_var[2:]
        self.iou_thres = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_label_idx = 0

    # def forward(self, confidence, pred_loc, gt_class_labels, gt_bbox_loc):
    #     """
    #      Compute the Multibox joint loss:
    #         L = (1/N) * L_{loc} + L_{class}
    #     :param confidence: predicted class probability, dim: (N, H*W*num_prior, num_classes)
    #     :param pred_loc: predicted prior bounding boxes, dim: (N, H*W*prior_num, 4)
    #     :param gt_class_labels: ground-truth class label, dim:(N, H*W*num_prior)
    #     :param gt_bbox_loc: ground-truth bounding box for prior, dim: (N, H*W*num_prior, 4)
    #     :return:
    #     """
    #     #Do the hard negative mining and produce balanced positive and negative examples
    #     with torch.no_grad():
    #         neg_class_prob = -F.log_softmax(confidence, dim=2)[:, :, self.neg_label_idx]      # select neg. class prob.
    #         pos_flag, neg_flag = hard_negative_mining(neg_class_prob, gt_class_labels, neg_pos_ratio=self.neg_pos_ratio)
    #         sel_flag = pos_flag | neg_flag
    #         num_pos = pos_flag.sum(dim=1, keepdim=True).float().cuda()
    #
    #     # Loss for the classification
    #     num_classes = confidence.shape[2]
    #     sel_conf = confidence[sel_flag]
    #     conf_loss = F.cross_entropy(sel_conf.reshape(-1, num_classes), gt_class_labels[sel_flag])
    #
    #     # Loss for the bounding box prediction
    #     sel_pred_loc = pred_loc[pos_flag]
    #     sel_gt_bbox_loc = gt_bbox_loc[pos_flag]
    #     loc_huber_loss = F.smooth_l1_loss(sel_pred_loc.view(-1, 4), sel_gt_bbox_loc.view(-1, 4),
    #                                       size_average=False).float().cuda()
    #     # conf_loss = conf_loss.mean(dim=0)
    #     # loc_huber_loss = loc_huber_loss.mean(dim=0)
    #     #print(conf_loss)
    #
    #     N = num_pos.data.sum()
    #     loc_huber_loss /= N
    #     conf_loss /= N
    #     return conf_loss, loc_huber_loss

    def forward(self, confidence, predicted_locations, labels, gt_locations):

        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask].long(), size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return classification_loss / num_pos, smooth_l1_loss / num_pos