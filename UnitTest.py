import unittest

import cv2
import numpy as np
from PIL import Image

import ssd_net
from vehicle_detection import load_data
import matplotlib.pyplot as plt
import torch
import mobilenet
from util import module_util
from bbox_helper import generate_prior_bboxes, iou, match_priors, bbox2loc,center2corner,corner2center
from cityscape_dataset import CityScapeDataset



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
        test_list = load_data('../cityscapes_samples', '../cityscapes_samples_labels')
        test_dataset = CityScapeDataset(test_list)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=1,
                                                        shuffle=True,
                                                        num_workers=0)
        idx, (img, bbox, label) = next(enumerate(test_data_loader))

        bbox=bbox[0].cpu().numpy()
        print(np.where(bbox>0))

        img = img[0].cpu().numpy()
        img = (img*128+np.asarray((127, 127, 127)))/255
        # for i in range(0, bbox.shape[0]):
        #     cv2.rectangle(img, (bbox[i,0], bbox[i,1]), (bbox[i,2], bbox[i,3]), (0, 255, 0), 3)
        print(img.shape)
        #cv2.imshow("img", img)
        plt.imshow(img, cmap='brg')
        plt.show()
        #cv2.waitKey(0)
        print('bbox',bbox)
        print('label',label)


class TestCorner2(unittest.TestCase):
    def test_corner2(self):
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
