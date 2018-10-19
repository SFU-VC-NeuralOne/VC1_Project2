import numpy as np
import torch

''' Prior Bounding Box  ------------------------------------------------------------------------------------------------
'''


def generate_prior_bboxes(prior_layer_cfg):
    """
    Generate prior bounding boxes on different feature map level. This function used in 'cityscape_dataset.py'

    Use VGG_SSD 300x300 as example:
    Feature map dimension for each output layers:
       Layer    | Map Dim (h, w) | Single bbox size that covers in the original image
    1. Conv5    | (38x38)        | (30x30) (unit. pixels)
    2. Conv11    | (19x19)        | (60x60)
    3. Conv14_2  | (10x10)        | (111x111)
    4. Conv15_2  | (5x5)          | (162x162)
    5. Conv16_2 | (3x3)          | (213x213)
    6. Conv17_2 | (1x1)          | (264x264)

    NOTE: The setting may be different using MobileNet v3, you have to set your own implementation.
    Tip: see the reference: 'Choosing scales and aspect ratios for default boxes' in original paper page 5.
    :param prior_layer_cfg: configuration for each feature layer, see the 'example_prior_layer_cfg' in the following.
    :return prior bounding boxes with form of (cx, cy, w, h), where the value range are from 0 to 1, dim (1, num_priors, 4)
    """

    sk_list = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]

    priors_bboxes = []
    for feat_level_idx in range(0, len(prior_layer_cfg)):               # iterate each layers
        layer_cfg = prior_layer_cfg[feat_level_idx]
        layer_feature_dim = layer_cfg['feature_dim_hw']
        layer_aspect_ratio = layer_cfg['aspect_ratio']

        # Todo: compute S_{k} (reference: SSD Paper equation 4.)
        sk = sk_list[feat_level_idx]
        fk = layer_cfg['feature_dim_hw'][0]

        for y in range(0, layer_feature_dim[0]):
            for x in range(0,layer_feature_dim[0]):

                # Todo: compute bounding box center
                cx = (x+0.5)/fk
                cy = (y+0.5)/fk

                # Todo: generate prior bounding box with respect to the aspect ratio
                for aspect_ratio in layer_aspect_ratio:
                    if aspect_ratio == '1t':
                        sk_ = np.sqrt(sk_list[feat_level_idx] * sk_list[feat_level_idx+1])
                        aspect_ratio=1.0
                        h = sk_ / np.sqrt(aspect_ratio)
                        w = sk_ * np.sqrt(aspect_ratio)
                        priors_bboxes.append([cx, cy, w, h])
                    else:
                        h = sk/np.sqrt(aspect_ratio)
                        w = sk* np.sqrt(aspect_ratio)
                        priors_bboxes.append([cx, cy, w, h])
    # np.set_printoptions(threshold=np.inf)
    # print(np.asarray(priors_bboxes))
    # Convert to Tensor
    priors_bboxes = torch.tensor(priors_bboxes)
    priors_bboxes = torch.clamp(priors_bboxes, 0.0, 1.0)
    num_priors = priors_bboxes.shape[0]

    # [DEBUG] check the output shape
    assert priors_bboxes.dim() == 2
    assert priors_bboxes.shape[1] == 4
    return priors_bboxes

def iou(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the Intersection over Union
    Note: function iou(a, b) used in match_priors
    :param a: bounding boxes, dim: (n_items, 4)
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference
    :return: iou value: dim: (n_item)
    """
    # [DEBUG] Check if input is the desire shape
    a = center2corner(a).cuda()
    b = center2corner(b).cuda()
    a_sh = a.size(0)
    b_sh = b.size(0)
    max_x_y = torch.min(a[:, 2:].unsqueeze(1).expand(a_sh, b_sh, 2), b[:, 2:].unsqueeze(0).expand(a_sh, b_sh, 2))
    min_x_y = torch.max(a[:, :2].unsqueeze(1).expand(a_sh, b_sh, 2), b[:, :2].unsqueeze(0).expand(a_sh, b_sh, 2))
    inter = torch.clamp((max_x_y - min_x_y), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter

    #temp = torch.transpose((inter / union), 0, 1)
    temp = inter/union
    return temp

def iou1(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the Intersection over Union
    Note: function iou(a, b) used in match_priors
    :param a: bounding boxes, dim: (n_items, 4)
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference
    :return: iou value: dim: (n_item)
    """
    # [DEBUG] Check if input is the desire shape

    assert a.dim() == 2
    assert a.shape[1] == 4
    #print ('b dim',b, b.dim())
    assert b.dim() == 2
    assert b.shape[1] == 4

    a = center2corner(a)
    b = center2corner(b)


    a_area =(a[:,2]-a[:,0])*(a[:,3]-a[:,1])
    b_area = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])

    x_max, y_max = np.maximum(a[:,0],b[:,0]), np.maximum(a[:,1], b[:,1])
    x_min, y_min = np.minimum(a[:,2],b[:,2]), np.minimum(a[:,3], b[:,3])
    temp_w = np.maximum((x_min-x_max),0)
    temp_h = np.maximum((y_min-y_max),0)

    a_and_b = temp_h.cuda()*temp_w.cuda()

    iou = a_and_b/(a_area.cuda()+b_area.cuda()-a_and_b)

    # [DEBUG] Check if output is the desire shape
    assert iou.dim() == 1
    assert iou.shape[0] == a.shape[0]
    return iou.view(1,a.shape[0])

def match_priors(prior_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, gt_labels: torch.Tensor, iou_threshold: float):
    """
    Match the ground-truth boxes with the priors.
    Note: Use this function in your ''cityscape_dataset.py', see the SSD paper page 5 for reference. (note that default box = prior boxes)

    :param gt_bboxes: ground-truth bounding boxes, dim:(n_samples, 4)
    :param gt_labels: ground-truth classification labels, negative (background) = 0, dim: (n_samples)
    :param prior_bboxes: prior bounding boxes on different levels, dim:(num_priors, 4)
    :param iou_threshold: matching criterion
    :return matched_boxes: real matched bounding box, dim: (num_priors, 4)
    :return matched_labels: real matched classification label, dim: (num_priors)
    """
    # [DEBUG] Check if input is the desire shape
    assert gt_bboxes.dim() == 2
    assert gt_bboxes.shape[1] == 4
    assert gt_labels.dim() == 1
    assert gt_labels.shape[0] == gt_bboxes.shape[0]
    assert prior_bboxes.dim() == 2
    assert prior_bboxes.shape[1] == 4

    # print('gt_bbox',gt_bboxes.dtype)
    iou_list = iou(gt_bboxes,prior_bboxes)

    # iou_list = torch.tensor([]).cuda()
    # for i in range(0, gt_bboxes.shape[0]):
    #     iou_list = torch.cat((iou_list, iou1(prior_bboxes, torch.reshape(gt_bboxes[i], (-1, 4)))), 0)

    matched_labels = torch.argmax(iou_list,dim=0)+1.0
    matched_labels.cuda()
    gt_idx = torch.argmax(iou_list, dim=1)
    size = gt_idx.shape[0]
    gt_label_idx = torch.arange(size).cuda()
    # print('gt lable idx dtype',gt_label_idx.dtype)
    #print(gt_idx.dtype, gt_label_idx.dtype, matched_labels.dtype, gt_labels.dtype)
    matched_labels[gt_idx] = gt_label_idx+1
    # print('ground truth matched bbox', matched_labels[gt_idx])
    matched_boxes = prior_bboxes.clone()
    # print('before',matched_boxes[gt_idx])
    matched_boxes[gt_idx] = bbox2loc(gt_bboxes[gt_label_idx], prior_bboxes[gt_idx])
    iou_list[gt_label_idx, gt_idx] = 1
    # print('iou of ground truth',iou_list[gt_label_idx,gt_idx])
    # print('iou of ground truth', iou_list[0, 11154])
    # print('after',print(matched_boxes[gt_idx]))
    #make sure every grouth truth has one bbox
    # for i in range (0, gt_bboxes.shape[0]):
    #     idx = torch.argmax(iou_list[i,:])
    #     matched_labels[idx] = gt_labels[i]
    #     matched_boxes[idx] = bbox2loc(gt_bboxes[i].float(), prior_bboxes[idx].float())
    #     gt_idx.append(idx)
    # gt_idx = np.asarray(gt_idx)
    # print('ground truth bbox',matched_boxes[gt_idx])
    #zero out labels below 0.5
    zero = torch.zeros(iou_list.shape).cuda()
    iou_list = torch.where(iou_list < iou_threshold, zero, iou_list)
    # print('ground truth matched bbox', matched_labels[gt_idx])
    zero_index = (torch.max(iou_list, dim=0)[0] == 0).nonzero()
    matched_labels[zero_index.view(1, -1)] = 0
    matched_boxes[zero_index.view(1, -1)] = torch.Tensor([0.,0.,0.,0.]).cuda()
    # print('gt labels should be',np.where(matched_labels>0))
    possitive_sample_idx = matched_labels.nonzero()
    temp = matched_labels[possitive_sample_idx.view(1, -1)]
    # print('this is 2628 before loc:', matched_boxes[2628])
    # # print('possitive sample dtype',temp)
    # print('2628 matches to', matched_labels[2628])

    matched_boxes[possitive_sample_idx.view(1, -1)] = bbox2loc(gt_bboxes[matched_labels[possitive_sample_idx.view(1, -1)] - 1], prior_bboxes[possitive_sample_idx.view(1, -1)])
    matched_labels[possitive_sample_idx.view(1, -1)] = gt_labels[matched_labels[possitive_sample_idx.view(1, -1)] - 1].long()
    # print('this is 2628 after:', matched_boxes[2628])
    # print('if i recall it back', loc2bbox(matched_boxes[2628], prior_bboxes[2628]))
    # print('the matched gt is', gt_bboxes[7])


    # for i in range(gt_bboxes.shape[0]):
    #     matched_boxes_for_this = ((matched_labels[possitive_sample_idx.view(1, -1)] - 1) ==i).nonzero()
    #     matched_boxes[matched_boxes_for_this] = bbox2loc(gt_bboxes[i], prior_bboxes[matched_boxes_for_this])
    #     #matched_boxes[possitive_sample_idx.view(1, -1)] = bbox2loc(gt_bboxes[matched_labels[possitive_sample_idx.view(1, -1)] - 1], prior_bboxes[possitive_sample_idx.view(1, -1)])


    # for i in range(0, iou_list.shape[1]):
    #     if i in gt_idx:
    #         continue
    #     elif torch.max(iou_list[:,i]) < iou_threshold:
    #         matched_labels[i] = 0
    #         matched_boxes[i] = torch.Tensor([0.,0.,0.,0.])
    #     else:
    #         ground_truth_bbox = torch.Tensor(gt_bboxes[matched_labels[i]-1])
    #         #print(gt_labels[matched_labels[i]-1])
    #         matched_labels[i] = gt_labels[matched_labels[i]-1]
    #         #gt_bboxes[i]=gt_bboxes[i] -1
    #         if(i==2400):
    #             print('hahh',matched_boxes[i])
    #         matched_boxes[i] = bbox2loc(ground_truth_bbox.float(), prior_bboxes[i].float())
    # print('ground truth bbox', matched_labels[gt_idx])
    # print('ground truth bbox',matched_labels[np.where(matched_labels>0)])
    #matched_boxes = prior_bboxes

    # [DEBUG] Check if output is the desire shape
    assert matched_boxes.dim() == 2
    assert matched_boxes.shape[1] == 4
    assert matched_labels.dim() == 1
    assert matched_labels.shape[0] == matched_boxes.shape[0]

    return matched_boxes, matched_labels

# def match_priors1(prior_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, gt_labels: torch.Tensor, iou_threshold: float):
#     overlaps = iou(prior_bboxes,gt_bboxes)
#         truths,
#         point_form(priors)
#     )
#     # (Bipartite Matching)
#     # [1,num_objects] best prior for each ground truth
#     best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
#     # [1,num_priors] best ground truth for each prior
#     best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
#     best_truth_idx.squeeze_(0)
#     best_truth_overlap.squeeze_(0)
#     best_prior_idx.squeeze_(1)
#     best_prior_overlap.squeeze_(1)
#     best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
#     # TODO refactor: index  best_prior_idx with long tensor
#     # ensure every gt matches with its prior of max overlap
#     for j in range(best_prior_idx.size(0)):
#         best_truth_idx[best_prior_idx[j]] = j
#     matches = truths[best_truth_idx]  # Shape: [num_priors,4]
#     conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
#     conf[best_truth_overlap < threshold] = 0  # label as background
#     loc = encode(matches, priors, variances)
#     loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
#     conf_t[idx] = conf  # [num_priors] top class label for each prior

def nms_bbox(bbox_loc, bbox_confid_scores, overlap_threshold=0.5, prob_threshold=0.6):
    """
    Non-maximum suppression for computing best overlapping bounding box for a object
    Use this function when testing the samples.

    :param bbox_loc: bounding box loc and size, dim: (num_priors, 4)
    :param bbox_confid_scores: bounding box confidence probabilities, dim: (num_priors, num_classes)
    :param overlap_threshold: the overlap threshold for filtering out outliers
    :return: selected bounding box with classes
    """

    # [DEBUG] Check if input is the desire shape
    assert bbox_loc.dim() == 2
    assert bbox_loc.shape[1] == 4
    assert bbox_confid_scores.dim() == 2
    assert bbox_confid_scores.shape[0] == bbox_loc.shape[0]

    sel_bbox = torch.tensor([[0., 0., 0., 0.]])
    sel_ind = []
    _, indices = torch.max(bbox_confid_scores, 0)
    # print(indices)

    # Todo: implement nms for filtering out the unnecessary bounding boxes
    num_classes = bbox_confid_scores.shape[1]
    for class_idx in range(1, num_classes):
        # print(class_idx)
        # the max probability bbox:
        temp_bbox = bbox_loc[indices[class_idx]]
        sel_ind.append(indices[class_idx])
        temp_bbox = temp_bbox.view(-1,4)
        sel_bbox = torch.cat((sel_bbox,temp_bbox),0)

        #eliminating bbox with classes score less than 0.6
        bbox_class_flag_index = np.where(bbox_confid_scores[:,class_idx] >= prob_threshold)
        bbox_class_flag = bbox_confid_scores[:,class_idx] >= prob_threshold
        selected_class_bbox = bbox_loc[bbox_class_flag]
        #print('selected flag in nums', selected_class_bbox.sum())
        #eliminate overlapping bboxes
        intersection = iou(selected_class_bbox,temp_bbox)
        print("intersection",intersection)
        not_overlap_flag = intersection<overlap_threshold
        print('nof',not_overlap_flag, not_overlap_flag.shape)
        print('bbcl',bbox_class_flag_index[0],  len(bbox_class_flag_index[0]))
        new_index = np.asarray(bbox_class_flag_index[0])*np.asarray(not_overlap_flag)
        sel_ind.append(new_index.nonzero())
        print('selected box in num', len(sel_ind))
        #not_overlapping_bboxes = selected_class_bbox[not_overlap_flag[0]]

        #sel_bbox = torch.cat((sel_bbox,not_overlapping_bboxes),0)
        # Tip: use prob_threshold to set the prior that has higher scores and filter out the low score items for fast
        # computation

        pass

    sel_bbox = sel_bbox[1:]
    print('selected box in num', sel_ind)
    return sel_ind

''' Bounding Box Conversion --------------------------------------------------------------------------------------------
'''


def loc2bbox(loc, priors, center_var=0.1, size_var=0.2):
    """
    Compute SSD predicted locations to boxes(cx, cy, h, w).
    :param loc: predicted location, dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: boxes: (cx, cy, h, w)
    """
    # assert priors.shape[0] == 1
    # assert priors.dim() == 3

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    l_center = loc[..., :2]
    l_size = loc[..., 2:]

    # real bounding box
    return torch.cat([
        center_var * l_center * p_size + p_center,      # b_{center}
        p_size * torch. exp(size_var * l_size)           # b_{size}
    ], dim=-1)


def bbox2loc(bbox, priors, center_var=0.1, size_var=0.2):
    """
    Compute boxes (cx, cy, h, w) to SSD locations form.
    :param bbox: bounding box (cx, cy, h, w) , dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: loc: (cx, cy, h, w)
    """
    # assert priors.shape[0] == 1
    # assert priors.dim() == 3

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    b_center = bbox[..., :2]
    b_size = bbox[..., 2:]

    temp = torch.cat([
        1 / center_var * ((b_center - p_center) / p_size),
        torch.log(b_size / p_size) / size_var
    ], dim=-1)
    return temp

def center2corner(center):
    """
    Convert bounding box in center form (cx, cy, w, h) to corner form (x,y) (x+w, y+h)
    :param center: bounding box in center form (cx, cy, w, h)
    :return: bounding box in corner form (x,y) (x+w, y+h)
    """
    return torch.cat([center[..., :2] - center[..., 2:]/2,
                      center[..., :2] + center[..., 2:]/2], dim=-1)


def corner2center(corner):
    """
    Convert bounding box in center form (cx, cy, w, h) to corner form (x,y) (x+w, y+h)
    :param center: bounding box in center form (cx, cy, w, h)
    :return: bounding box in corner form (x,y) (x+w, y+h)
    """
    return torch.cat([corner[..., :2]/2 + corner[..., 2:]/2,
                      corner[..., 2:] - corner[..., :2]],dim=-1)