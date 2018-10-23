import os
import pickle
import unittest
import time

import cv2
import numpy as np
from PIL import Image
from matplotlib import patches

import ssd_net
from vehicle_detection import load_data
import matplotlib.pyplot as plt
import torch
import mobilenet
from util import module_util
from bbox_helper import generate_prior_bboxes, match_priors, bbox2loc, center2corner, corner2center, loc2bbox, \
    nms_bbox, iou, nms_bbox1
from cityscape_dataset import CityScapeDataset
from skimage.transform import resize



class TestLoadingData(unittest.TestCase):
    def test_Loading(self):
        test_list = load_data('../cityscapes_samples','../cityscapes_samples_labels')
        item = test_list[1]
        ground_truth = item['label']
        labels=ground_truth[0]
        bbox = np.asarray(ground_truth[1],dtype=np.float32)
        bbox=bbox/ [2048, 1024, 2048, 1024]
        # for i in range (0, len(ground_truth)):
        #     labels.append()
        # labels = ground_truth['class']
        print(labels)
        print(bbox)
        # print(test_list[0]['file_path'])
        # print(test_list[0]['label'][0]['class'])
        # print(test_list[0]['label'][0]['position'][0])
        # print(test_list[0]['label'][0]['position'][1])
        # print(test_list[0]['label'][1]['class'])
        # print(test_list[0]['label'][1]['position'][0])
        # print(test_list[0]['label'][1]['position'][1])
        # img = Image.open(test_list[0]['file_path'])
        # plt.imshow(img)
        # plt.show()
        self.assertEqual('foo'.upper(), 'FOO')

class TestPiorBB(unittest.TestCase):
    def ssd_size_bounds_to_values(self, size_bounds, n_feat_layers, img_shape=(300, 300)):
        """Compute the reference sizes of the anchor boxes from relative bounds.
        The absolute values are measured in pixels, based on the network
        default size (300 pixels).

        This function follows the computation performed in the original
        implementation of SSD in Caffe.

        Return:
        list of list containing the absolute sizes at each scale. For each scale,
        the ratios only apply to the first value.
        """
        assert img_shape[0] == img_shape[1]

        img_size = img_shape[0]
        min_ratio = int(size_bounds[0] * 100)
        max_ratio = int(size_bounds[1] * 100)
        temp = (max_ratio - min_ratio) / (n_feat_layers - 2)
        step = int(np.math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
        # Start with the following smallest sizes.
        sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
        for ratio in range(min_ratio, max_ratio + 1, step):
            sizes.append((img_size * ratio / 100.,
                          img_size * (ratio + step) / 100.))
        return sizes

    def ssd_anchor_one_layer(self, img_shape,
                             feat_shape,
                             sizes,
                             ratios,
                             step,
                             offset=0.5,
                             dtype=np.float32):
        """Computer SSD default anchor boxes for one feature layer.
        Determine the relative position grid of the centers, and the relative
        width and height.
        Arguments:
          feat_shape: Feature shape, used for computing relative position grids;
          size: Absolute reference sizes;
          ratios: Ratios to use on these features;
          img_shape: Image shape, used for computing height, width relatively to the
            former;
          offset: Grid offset.
        Return:
          y, x, h, w: Relative x and y grids, and height and width.
        """
        # Compute the position grid: simple way.
        y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        y = (y.astype(dtype) + offset) / feat_shape[0]
        x = (x.astype(dtype) + offset) / feat_shape[1]
        # Weird SSD-Caffe computation using steps values...
        # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        # y = (y.astype(dtype) + offset) * step / img_shape[0]
        # x = (x.astype(dtype) + offset) * step / img_shape[1]

        # Expand dims to support easy broadcasting.
        y = np.expand_dims(y, axis=-1)  # [size, size, 1]
        x = np.expand_dims(x, axis=-1)  # [size, size, 1]

        # Compute relative height and width.
        # Tries to follow the original implementation of SSD for the order.
        num_anchors = len(sizes) + len(ratios)
        h = np.zeros((num_anchors,), dtype=dtype)  # [n_anchors]
        w = np.zeros((num_anchors,), dtype=dtype)  # [n_anchors]
        # Add first anchor boxes with ratio=1.
        h[0] = sizes[0] / img_shape[0]
        w[0] = sizes[0] / img_shape[1]
        di = 1
        if len(sizes) > 1:
            h[1] = np.math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
            w[1] = np.math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
            di += 1
        for i, r in enumerate(ratios):
            h[i + di] = sizes[0] / img_shape[0] / np.math.sqrt(r)
            w[i + di] = sizes[0] / img_shape[1] * np.math.sqrt(r)
        return y, x, h, w

    def test_priorbb(self):
        prior_layer_cfg = [
            # Example:
            {'layer_name': 'Conv5', 'feature_dim_hw': (38, 38), 'bbox_size': (30, 30),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv11', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv14_2', 'feature_dim_hw': (10, 10), 'bbox_size': (111, 111),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv15_2', 'feature_dim_hw': (5, 5), 'bbox_size': (162, 162),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv16_2', 'feature_dim_hw': (3, 3), 'bbox_size': (213, 213),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv17_2', 'feature_dim_hw': (1, 1), 'bbox_size': (264, 264),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)}
        ]
        pp=generate_prior_bboxes(prior_layer_cfg)

        print(pp[0:1], pp[39:40])
        temp = iou(pp[0:6], pp[0:1])
        print('iou',temp)
        gt_label = torch.tensor([1])
        # print(gt_label.dim[0])
        print('matching', match_priors(pp[0:38],pp[38:39],gt_label,0.5))
        np.set_printoptions(threshold=np.inf)
        size_bounds=[0.2,0.9]
        img_shape = [300,300]
        # list = self.ssd_size_bounds_to_values(size_bounds,6,img_shape)
        # print(list)
        #prior_bbox = self.ssd_anchor_one_layer((300,300),(38,38),(30,60), [2, .5, 3, 1. / 3], 3)
        #print(prior_bbox)

        self.assertEqual('foo'.upper(), 'FOO')

class TestMin(unittest.TestCase):
    def test_min(self):
        a = torch.randn(4, 4)
        print(a)
        print(torch.argmin(a, dim=0)+1)
        b = torch.tensor([0, 1, 2, 3])
        print('b:',b)
        b=torch.reshape(b, (1, 4))
        print(b)
        for i in range(0,1):
            print('i',i)
        x = torch.randn(2, 3)
        torch.cat((x, x, x), 0)
        self.assertEqual('foo'.upper(), 'FOO')

class TestNN(unittest.TestCase):
    def test_nn(self):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # model = mobilenet.MobileNet()
        # module_util.summary_layers(model,(3,300,300))
        model = ssd_net.SSD(3)
        module_util.summary_layers(model, (3, 300, 300))
        self.assertEqual('foo'.upper(), 'FOO')

class TestBbox2loc(unittest.TestCase):
    def test_bbox2loc(self):
        prior_layer_cfg = [
            # Example:
            {'layer_name': 'Conv5', 'feature_dim_hw': (38, 38), 'bbox_size': (30, 30),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv11', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv14_2', 'feature_dim_hw': (10, 10), 'bbox_size': (111, 111),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv15_2', 'feature_dim_hw': (5, 5), 'bbox_size': (162, 162),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv16_2', 'feature_dim_hw': (3, 3), 'bbox_size': (213, 213),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
            {'layer_name': 'Conv17_2', 'feature_dim_hw': (1, 1), 'bbox_size': (264, 264),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)}
        ]
        pp = generate_prior_bboxes(prior_layer_cfg)

        #print(pp[0:1], pp[39:40])
        print(bbox2loc(pp[0:5],pp[0:1]))

        self.assertEqual('foo'.upper(), 'FOO')

class TestDataLoad(unittest.TestCase):
    def test_dataLoad(self):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.set_printoptions(precision=10)
        prior_layer_cfg = [
            {'layer_name': 'Conv5', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv11', 'feature_dim_hw': (10, 10), 'bbox_size': (105, 105),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv14_2', 'feature_dim_hw': (5, 5), 'bbox_size': (150, 150),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv15_2', 'feature_dim_hw': (3, 3), 'bbox_size': (195, 195),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv16_2', 'feature_dim_hw': (2, 2), 'bbox_size': (240, 240),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv17_2', 'feature_dim_hw': (1, 1), 'bbox_size': (285, 285),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')}
        ]
        pp = generate_prior_bboxes(prior_layer_cfg)


        #test_list = load_data('../Debugimage', '../Debuglabel')
        test_list = load_data('../cityscapes_samples', '../cityscapes_samples_labels')
        print(test_list)
        gt_bbox = np.asarray(test_list[0]['label'][1])*[300/2048, 300/1024, 300/2048, 300/1024]
        print('ground truth from file:', test_list[0]['label'][0])
        test_dataset = CityScapeDataset(test_list)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=1,
                                                        shuffle=True,
                                                        num_workers=0)
        idx, (img, bbox, label) = next(enumerate(test_data_loader))
        bbox = bbox[0]
        label= label[0]
        print(bbox.shape, label.shape)

        print('matched label', label[np.where(label > 0)], np.where(label > 0), label.shape)
        print('first bbox from data_set:', bbox[0], label[0])
        bbox_center = loc2bbox(bbox, pp)
        bbox_corner = center2corner(bbox_center)
        img = img[0].cpu().numpy()
        img = img.reshape((300, 300, 3))
        img = (img*128+np.asarray([[127, 127, 127]]))/255
        # for i in range(0, bbox.shape[0]):
        #     cv2.rectangle(img, (bbox[i,0], bbox[i,1]), (bbox[i,2], bbox[i,3]), (0, 255, 0), 3)
        #cv2.imshow("img", img)
        # Create figure and axes
        fig, ax = plt.subplots(1)
        imageB_array = resize(img, (300, 300), anti_aliasing=True)
        ax.imshow(imageB_array, cmap='brg')
        bbox_corner = bbox_corner.cpu().numpy()
        bbox_corner = bbox_corner[np.where(label > 0)]
        temp_lab = label[np.where(label > 0)]
        print('matched bbox ======', bbox_corner)
        pp = center2corner(pp)
        pp = pp[np.where(label > 0)]
        print('864 tensor: ',pp)
        for i in range(0,bbox_corner.shape[0]):
            if temp_lab[i] == 1 :
            # print('i point', bbox_corner[i, 0]*600, bbox_corner[i, 1]*300,(bbox_corner[i, 2]-bbox_corner[i, 0])*600, (bbox_corner[i, 3]-bbox_corner[i, 1])*300)
                rect = patches.Rectangle((bbox_corner[i, 0]*300, bbox_corner[i, 1]*300), (bbox_corner[i, 2]-bbox_corner[i, 0])*300, (bbox_corner[i, 3]-bbox_corner[i, 1])*300, linewidth=2, edgecolor='r', facecolor='none') # Create a Rectangle patch
                ax.add_patch(rect) # Add the patch to the Axes
            else:
                rect = patches.Rectangle((bbox_corner[i, 0] * 1200, bbox_corner[i, 1] * 600),
                                         (bbox_corner[i, 2] - bbox_corner[i, 0]) * 1200,
                                         (bbox_corner[i, 3] - bbox_corner[i, 1]) * 600, linewidth=2, edgecolor='y',
                                         facecolor='none')  # Create a Rectangle patch
                ax.add_patch(rect)  # Add the patch to the Axes
        for i in range(0, pp.shape[0]):
            rect = patches.Rectangle((pp[i, 0] * 300, pp[i, 1] * 300),
                                     (pp[i, 2] - pp[i, 0]) * 300,
                                     (pp[i, 3] - pp[i, 1]) * 300, linewidth=1, edgecolor='blue',
                                     facecolor='none')  # Create a Rectangle patch
            ax.add_patch(rect)  # Add the patch to the Axes

        # for i in range(0, gt_bbox.shape[0]):
        #     rect = patches.Rectangle((gt_bbox[i][0], gt_bbox[i][1]),
        #                              (gt_bbox[i][2] - gt_bbox[i][0]),
        #                              (gt_bbox[i][3] - gt_bbox[i][1]), linewidth=1, edgecolor='g',
        #                              facecolor='none')  # Create a Rectangle patch
        #     ax.add_patch(rect)  # Add the patch to the Axes



        plt.show()
        #cv2.waitKey(0)
        # print('bbox',bbox)
        # print('label',label)


class TestCorner2(unittest.TestCase):
    def test_corner2(self):
        prior_layer_cfg = [
            {'layer_name': 'Conv5', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv11', 'feature_dim_hw': (10, 10), 'bbox_size': (105, 105),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv14_2', 'feature_dim_hw': (5, 5), 'bbox_size': (150, 150),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv15_2', 'feature_dim_hw': (3, 3), 'bbox_size': (195, 195),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv16_2', 'feature_dim_hw': (2, 2), 'bbox_size': (240, 240),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv17_2', 'feature_dim_hw': (1, 1), 'bbox_size': (285, 285),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')}
        ]
        pp = generate_prior_bboxes(prior_layer_cfg)
        print('original',pp[0])
        test = center2corner(pp[0])
        print('corner',test)
        test = corner2center(test)
        print('center',test)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print('Pytorch CUDA Enabled?:', torch.cuda.is_available())
        b = 0.5 * torch.eye(3)
        b_gpu = b.cuda()
        print(b_gpu)

class TestLoadingNN(unittest.TestCase):
    def test_loadNN(self):
        temp_state = torch.load('pretrained/mobienetv2.pth')
        self.base_net = mobilenet.MobileNet(2)

        cur_dict = self.base_net.state_dict()
        input_state = {k: v for k, v in temp_state.items() if
                       k in cur_dict and v.size() == cur_dict[k].size()}
        cur_dict.update(input_state)
        self.base_net.load_state_dict(cur_dict)

        print(input_state.keys())

class TestMatching(unittest.TestCase):
    def test_Match(self):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        pp=np.asarray([[0,0,2,2],[1,0,3,2  ],[0,0,1,3],[3,3,4,4]],dtype=np.float32)
        pp = corner2center(torch.Tensor(pp))
        gt = torch.Tensor([[1,1,1,1]])
        test = iou(pp,torch.Tensor(gt))
        print('iou test',test)

        test_iou = np.asarray([[0, 0, 0.2, 0.2, 0.6],
                               [0.1, 0, 0.2, 0.8, 0.2],
                               [0, 0.4, 0.1, 0.3, 0.3],
                               [0.3, 0.3, 0.5, 0.9, 0.1]], dtype=np.float32)
        test_iou = torch.Tensor(test_iou)
        print('test argmax',torch.argmax(test_iou,dim=1))
        zero =torch.zeros(test_iou.shape)
        test_iou = torch.where(test_iou<0.5, zero, test_iou)
        print(test_iou)
        #zero_idx = torch.where(torch.max(test_iou, dim=0)[0] == 0, )
        gt_label = torch.tensor([1,1,2,2])
        matched_label = torch.tensor(([1,3, 4, 2, 2]))
        print('variable matched_label', matched_label, matched_label.dtype)
        zero_index = (torch.max(test_iou, dim=0)[0] == 0).nonzero()
        print('index of below 0.5',zero_index, zero_index.dtype, zero_index, zero_index.view(1, -1))
        matched_label[zero_index.view(1, -1)] = 0
        print('after clear out the zero',matched_label)
        possitive_sample_idx = matched_label.nonzero()
        matched_label[possitive_sample_idx.view(1,-1)] = gt_label[matched_label[possitive_sample_idx.view(1,-1)]-1]
        print('non zero labels',possitive_sample_idx, possitive_sample_idx.dtype)
        print('final label', matched_label)
        print('test where',torch.max(test_iou, dim=0))
        for i in range(0, test_iou.shape[1]):
            if torch.max(test_iou[:, i]) < 0.5:
                print(i)
        timestamp = time.time()
        filename = 'ssd_net' + str(timestamp) + '.pth'
        print(filename)

class TestIntercetion(unittest.TestCase):
    def test_intersect(self):
        """ We resize both tensors to [A,B,2] without new malloc:
            [A,2] -> [A,1,2] -> [A,B,2]
            [B,2] -> [1,B,2] -> [A,B,2]
            Then we compute the area of intersect between box_a and box_b.
            Args:
              box_a: (tensor) bounding boxes, Shape: [A,4].
              box_b: (tensor) bounding boxes, Shape: [B,4].
            Return:
              (tensor) intersection area, Shape: [A,B].
            """
        box_a = torch.Tensor([[0,0,2,2],[1,0,3,2],[0,0,1,3],[3,3,4,4]])
        box_b = torch.Tensor([[2,0,3,1],[3,0,4,1]])
        A = box_a.size(0)
        B = box_b.size(0)
        # print(box_a[:, 2:].unsqueeze(1))
        # print(box_a[:, 2:].unsqueeze(0))
        # print(box_a[:, 2:].unsqueeze(1).expand(A, B, 2))
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] -
                   box_a[:, 1])).unsqueeze(1).expand_as(inter)
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] -
                   box_b[:, 1])).unsqueeze(0).expand_as(inter)
        union = area_a + area_b - inter
        # print(inter)
        temp = torch.transpose((inter / union), 0, 1)
        print(temp)
        print(inter / union)
        print(iou(box_a, box_b))

class TestBbox2Loc(unittest.TestCase):
    def test_bbox2loc(self):
        prior  = torch.Tensor([[1,2,3,4], [2,0,3,4], [1,5,6,7]])
        gt = torch.Tensor([[2,6,7,8], [5,6,7,8], [2,0,4,8]])

        temp = bbox2loc(gt,prior)
        temp = bbox2loc(gt, prior)
        back = loc2bbox(temp, prior)
        print(back)

class TestRabdom(unittest.TestCase):
    def test_random(self):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.set_printoptions(precision=10)
        prior_layer_cfg = [
            {'layer_name': 'Conv5', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv11', 'feature_dim_hw': (10, 10), 'bbox_size': (105, 105),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv14_2', 'feature_dim_hw': (5, 5), 'bbox_size': (150, 150),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv15_2', 'feature_dim_hw': (3, 3), 'bbox_size': (195, 195),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv16_2', 'feature_dim_hw': (2, 2), 'bbox_size': (240, 240),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv17_2', 'feature_dim_hw': (1, 1), 'bbox_size': (285, 285),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')}
        ]
        pp = generate_prior_bboxes(prior_layer_cfg)

        # test_list = load_data('../Debugimage', '../Debuglabel')
        test_list = load_data('../cityscapes_samples', '../cityscapes_samples_labels')
        #print(test_list)

        test_dataset = CityScapeDataset(test_list)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       num_workers=0)
        lfw_dataset_dir = '../'
        test_net = ssd_net.SSD(2)
        test_net_state = torch.load(os.path.join(lfw_dataset_dir, 'ssd_net.pth'))
        test_net.load_state_dict(test_net_state)
        #test_net.eval()
        idx, (img, bbox, label) = next(enumerate(test_data_loader))
        pred_cof, pred_loc = test_net.forward(img)
        pred_loc = pred_loc[0]
        pred_cof = pred_cof[0]

        import torch.nn.functional as F
        test_cof_score = F.softmax(pred_cof)
        print(test_cof_score)
        sel_idx = nms_bbox1(pred_loc.detach(), pp, test_cof_score.detach(),overlap_threshold=0.5, prob_threshold=0.24)
        # sel_idx = np.flatten(sel_idx)
        # print('select idx, keep',sel_idx, keep)
        sel_bboxes = pred_loc.detach()[sel_idx]
        print('slected bbox',sel_bboxes)
        bbox_center = loc2bbox(sel_bboxes, pp[sel_idx])
        conf = pred_cof[sel_idx]
        img = img[0].cpu().numpy()
        img = img.reshape((300, 300, 3))
        img = (img * 128 + np.asarray([[127, 127, 127]])) / 255
        fig, ax = plt.subplots(1)
        imageB_array = resize(img, (600, 600), anti_aliasing=True)
        ax.imshow(imageB_array, cmap='brg')
        # print(conf)
        bbox_corner = center2corner(bbox_center)

        for i in range(0,bbox_corner.shape[0]):
            # print('i point', bbox_corner[i, 0]*600, bbox_corner[i, 1]*300,(bbox_corner[i, 2]-bbox_corner[i, 0])*600, (bbox_corner[i, 3]-bbox_corner[i, 1])*300)
            rect = patches.Rectangle((bbox_corner[i, 0]*600, bbox_corner[i, 1]*600), (bbox_corner[i, 2]-bbox_corner[i, 0])*600, (bbox_corner[i, 3]-bbox_corner[i, 1])*600, linewidth=1, edgecolor='r', facecolor='none') # Create a Rectangle patch
            ax.add_patch(rect) # Add the patch to the Axes
        plt.show()
class TestPlot(unittest.TestCase):
    def test_plot(self):
        with open('train_loc.pkl', 'rb') as f:
            train_loc = pickle.load(f)
            train_loc=np.asarray(train_loc)
        with open('train_cof.pkl', 'rb') as f:
            train_cof = pickle.load(f)
            train_cof = np.asarray(train_cof)
        with open('val_loc.pkl', 'rb') as f:
            val_loc = pickle.load(f)
            val_loc = np.asarray(val_loc)
        with open('val_cof.pkl', 'rb') as f:
            val_cof = pickle.load(f)
            val_cof = np.asarray(val_cof)
        # train_loss = train_loc + train_cof
        # val_loss =
        fig, ax = plt.subplots(2)
        ax[0].plot(train_loc[:, 0], train_loc[:, 1])  # loss value
        ax[0].plot(val_loc[:, 0], val_loc[:, 1])  # loss value
        ax[1].plot(train_cof[:, 0], train_cof[:, 1])  # loss value
        ax[1].plot(val_cof[:, 0], val_cof[:, 1])  # loss value
        plt.show()

class TestRabdom2(unittest.TestCase):
    def test_random2(self):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.set_printoptions(precision=10)
        prior_layer_cfg = [
            {'layer_name': 'Conv5', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv11', 'feature_dim_hw': (10, 10), 'bbox_size': (105, 105),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv14_2', 'feature_dim_hw': (5, 5), 'bbox_size': (150, 150),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv15_2', 'feature_dim_hw': (3, 3), 'bbox_size': (195, 195),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv16_2', 'feature_dim_hw': (2, 2), 'bbox_size': (240, 240),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv17_2', 'feature_dim_hw': (1, 1), 'bbox_size': (285, 285),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')}
        ]
        pp = generate_prior_bboxes(prior_layer_cfg)

        # test_list = load_data('../Debugimage', '../Debuglabel')
        test_list = load_data('../cityscapes_samples', '../cityscapes_samples_labels')
        #print(test_list)

        test_dataset = CityScapeDataset(test_list)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       num_workers=0)
        lfw_dataset_dir = '../'
        test_net = ssd_net.SSD(3)
        test_net_state = torch.load(os.path.join(lfw_dataset_dir, 'ssd_net.pth'))
        test_net.load_state_dict(test_net_state)
        idx, (img, bbox, label) = next(enumerate(test_data_loader))
        pred_cof, pred_loc = test_net.forward(img)
        print(pred_loc.shape)
        import torch.nn.functional as F
        pred_loc = pred_loc.detach()
        bbox_center = loc2bbox(pred_loc[0], pp)
        pred_cof =F.softmax(pred_cof[0])
        ind = np.where(pred_cof>0.7)
        # pred_cof = F.softmax(pred_cof[ind[0]])
        bbox_center = bbox_center[ind[0]]
        print(ind,pred_cof)
        img = img[0].cpu().numpy()
        img = img.reshape((300, 300, 3))
        img = (img * 128 + np.asarray([[127, 127, 127]])) / 255
        fig, ax = plt.subplots(1)
        imageB_array = resize(img, (600, 1200), anti_aliasing=True)
        ax.imshow(imageB_array, cmap='brg')

        bbox_corner = center2corner(bbox_center)

        for i in range(0,bbox_corner.shape[0]):
            # print('i point', bbox_corner[i, 0]*600, bbox_corner[i, 1]*300,(bbox_corner[i, 2]-bbox_corner[i, 0])*600, (bbox_corner[i, 3]-bbox_corner[i, 1])*300)
            rect = patches.Rectangle((bbox_corner[i, 0]*1200, bbox_corner[i, 1]*600), (bbox_corner[i, 2]-bbox_corner[i, 0])*1200, (bbox_corner[i, 3]-bbox_corner[i, 1])*600, linewidth=2, edgecolor='r', facecolor='none') # Create a Rectangle patch
            ax.add_patch(rect) # Add the patch to the Axes
        plt.show()

class TestNms(unittest.TestCase):
    def test_nms(self):
        conf = torch.Tensor([[0.5,0.5], [0.4,0.6], [0.7, 0.3], [0.2,0.8]])
        bbox = torch.Tensor([[1,2,3,4],[0,0,1,1],[1,1,2,2],[3,3,6,6]])
        zeros = torch.Tensor(conf.shape[0])
        prob_threshold = 0.6
        overlap_threshold = 0.5
        # vehicle_conf = torch.where(conf>prob_threshold, conf, zeros)

        # print(vehicle_conf==0)
        # print((vehicle_conf==0).nonzero().reshape(1,-1))
        #bbox[(vehicle_conf==0).nonzero().reshape(1,-1)] = torch.Tensor([0,0,0,0])
        # print(bbox)
        # non_zero_idx = vehicle_conf.nonzero().reshape(1, -1)
        # sel_idx = []
        # while (vehicle_conf.nonzero().shape[0] != 0):
        #     highest_idx = torch.argmax(vehicle_conf, dim=0)
        #     sel_idx.append(highest_idx)
        #     iou_list = iou(bbox, bbox[highest_idx].reshape(1, -1))
        #     print(iou_list)
        #     overlapped_idx = (iou_list>overlap_threshold).nonzero().reshape(1,-1)
        #     vehicle_conf[overlapped_idx] = 0

        print('nms idx', nms_bbox1(bbox,bbox,conf))
