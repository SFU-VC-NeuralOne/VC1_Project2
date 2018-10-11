import unittest

import numpy as np
from PIL import Image
from vehicle_detection import load_data
import matplotlib.pyplot as plt
import torch
import mobilenet
from util import module_util
from bbox_helper import generate_prior_bboxes
from bbox_helper import iou


class TestLoadingData(unittest.TestCase):
    def test_Loading(self):
        test_list = load_data('../cityscapes_samples','../cityscapes_samples_labels')
        print(test_list[0])
        print(test_list[0]['file_path'])
        print(test_list[0]['label'][0]['class'])
        print(test_list[0]['label'][0]['position'][0])
        print(test_list[0]['label'][0]['position'][1])
        print(test_list[0]['label'][1]['class'])
        print(test_list[0]['label'][1]['position'][0])
        print(test_list[0]['label'][1]['position'][1])
        img = Image.open(test_list[0]['file_path'])
        plt.imshow(img)
        plt.show()
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
        print(iou(pp[0:39], pp[0:39]))
        np.set_printoptions(threshold=np.inf)
        size_bounds=[0.2,0.9]
        img_shape = [300,300]
        # list = self.ssd_size_bounds_to_values(size_bounds,6,img_shape)
        # print(list)
        #prior_bbox = self.ssd_anchor_one_layer((300,300),(38,38),(30,60), [2, .5, 3, 1. / 3], 3)
        #print(prior_bbox)

        self.assertEqual('foo'.upper(), 'FOO')

