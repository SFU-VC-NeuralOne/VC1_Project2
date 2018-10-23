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
    1. Conv5    | (38x38)        | (60x30) (unit. pixels)
    2. Conv11    | (19x19)        | (105x60)
    3. Conv14_2  | (10x10)        | (150x111)
    4. Conv15_2  | (5x5)          | (195x162)
    5. Conv16_2 | (3x3)          | (240x213)
    6. Conv17_2 | (1x1)          | (285x264)

    NOTE: The setting may be different using MobileNet v3, you have to set your own implementation.
    Tip: see the reference: 'Choosing scales and aspect ratios for default boxes' in original paper page 5.
    :param prior_layer_cfg: configuration for each feature layer, see the 'example_prior_layer_cfg' in the following.
    :return prior bounding boxes with form of (cx, cy, w, h), where the value range are from 0 to 1, dim (1, num_priors, 4)
    """

    sk_list = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1]

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

    temp = torch.transpose((inter / union), 0, 1)
    #temp = inter/union
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
    iou_list = iou(prior_bboxes, gt_bboxes)

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


def nms_bbox(bbox_loc, prior, bbox_confid_scores, overlap_threshold=0.5, prob_threshold=0.6):
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
    sel_ind = np.array([]).reshape((1,-1))
    _, indices = torch.max(bbox_confid_scores, 0)
    print("indices", indices)

    bbox_loc = loc2bbox(bbox_loc, prior)
    bbox_loc = center2corner(bbox_loc)

    # Todo: implement nms for filtering out the unnecessary bounding boxes
    num_classes = bbox_confid_scores.shape[1]
    for class_idx in range(1, num_classes):
        print(indices[class_idx])
        print("First thing first : ",bbox_confid_scores[indices[class_idx]])
        if bbox_confid_scores[indices[class_idx]][class_idx] >= prob_threshold :
            # print(class_idx)
            # the max probability bbox:
            temp_bbox = bbox_loc[indices[class_idx]]
            # temp_bbox = loc2bbox(temp_bbox)
            sel_ind = np.concatenate((sel_ind.reshape((1,-1)),np.asarray(indices[class_idx]).reshape((1,-1))),axis=1)
            temp_bbox = temp_bbox.view(-1,4)
            sel_bbox = torch.cat((sel_bbox,temp_bbox),0)

            # eliminating bbox with classes score less than 0.6
            bbox_class_flag_index = np.where(bbox_confid_scores[:,class_idx] >= prob_threshold)[0]
            bbox_class_flag = bbox_confid_scores[:,class_idx] >= prob_threshold
            if (bbox_class_flag.sum()) >= 1 :
                selected_class_bbox = bbox_loc[bbox_class_flag]
                # print('selected flag in nums', selected_class_bbox.sum())
                # eliminate overlapping bboxes
                intersection = iou(selected_class_bbox,temp_bbox)
                print("intersection",intersection)
                not_overlap_flag = intersection<overlap_threshold
                print('nof',not_overlap_flag, not_overlap_flag.shape)
                print('bbcl',bbox_class_flag_index)
                bbox_class_flag_index = np.asarray(bbox_class_flag_index).reshape(-1,1)
                not_overlap_flag = np.asarray(not_overlap_flag)
                new_index = bbox_class_flag_index * not_overlap_flag
                new_index = new_index[new_index.nonzero()[0]].reshape(1,-1)
                sel_ind = np.concatenate((sel_ind.reshape((1,-1)), new_index.reshape(1,-1)),axis=1)
                print('selected box in num', len(sel_ind))
            #not_overlapping_bboxes = selected_class_bbox[not_overlap_flag[0]]

            #sel_bbox = torch.cat((sel_bbox,not_overlapping_bboxes),0)
            # Tip: use prob_threshold to set the prior that has higher scores and filter out the low score items for fast
            # computation



    sel_bbox = sel_bbox[1:]
    print('selected box in num', sel_ind)
    return sel_ind

def nms_bbox1(bbox_loc, prior, bbox_confid_scores, overlap_threshold=0.5, prob_threshold=0.6):
    bbox = loc2bbox(bbox_loc, prior)
    bbox = center2corner(bbox)

    vehicle_cof = bbox_confid_scores[:,1]
    zeros = torch.zeros(vehicle_cof.shape)
    print('zero shape',zeros.shape)
    print('vehicle_cof', vehicle_cof.shape)
    vehicle_cof = torch.where(vehicle_cof > prob_threshold, vehicle_cof, zeros)
    print('vehicle non zero',(vehicle_cof == 0).nonzero().shape, (vehicle_cof == 0).nonzero().reshape(1, -1))

    #bbox[(vehicle_conf == 0).nonzero().reshape(1, -1)] = torch.Tensor([0, 0, 0, 0])

    non_zero_idx = vehicle_cof.nonzero().reshape(1, -1)
    sel_idx = []
    while (vehicle_cof.nonzero().shape[0] != 0):
        highest_idx = torch.argmax(vehicle_cof, dim=0)
        print('the highest',vehicle_cof[highest_idx])
        sel_idx.append(highest_idx)
        iou_list = iou(bbox, bbox[highest_idx].reshape(1, -1))
        overlapped_idx = (iou_list[0] > overlap_threshold).nonzero().reshape(1, -1)
        #print((iou_list[0] > overlap_threshold).nonzero())
        vehicle_cof[overlapped_idx] = 0
        print(vehicle_cof.nonzero().shape[0])
        if(vehicle_cof.nonzero().shape[0] <50 ):
            print(vehicle_cof[vehicle_cof.nonzero().reshape(1,-1)])

    return np.asarray(sel_idx)

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

def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    """Assign ground truth boxes and targets to priors.
    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]

    # boxes = corner2center(boxes)
    # center_from_priors = corner2center(corner_form_priors)
    # locations = bbox2loc(boxes, center_from_priors)

    return boxes, labels

def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)
