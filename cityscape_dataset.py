import random

import numpy as np
import torch.nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from bbox_helper import generate_prior_bboxes, match_priors,corner2center


class CityScapeDataset(Dataset):

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        prior_layer_cfg = [
            {'layer_name': 'Conv5', 'feature_dim_hw': (38, 38), 'bbox_size': (30, 30),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv11', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv14_2', 'feature_dim_hw': (10, 10), 'bbox_size': (111, 111),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv15_2', 'feature_dim_hw': (5, 5), 'bbox_size': (162, 162),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv16_2', 'feature_dim_hw': (3, 3), 'bbox_size': (213, 213),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')},
            {'layer_name': 'Conv17_2', 'feature_dim_hw': (1, 1), 'bbox_size': (264, 264),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, '1t')}
        ]
        self.prior_bboxes = generate_prior_bboxes(prior_layer_cfg)

        # Pre-process parameters:
        #  Normalize: (I-self.mean)/self.std
        self.mean = np.asarray((127, 127, 127))
        self.std = 128.0

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes.
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4)
        :return bbox_label: matched classification label, dim: (num_priors)
        """


        # 1. Load image as well as the bounding box with its label
        item = self.dataset_list[idx]
        file_path = item['file_path']
        ground_truth = item['label']
        sample_labels = ground_truth[0]
        sample_bboxes = np.asarray(ground_truth[1], dtype=np.float32)
        sample_img = Image.open(file_path)

        # augmentation = np.random.randint(0, 5)
        augmentation=None
        if augmentation == 0:
            sample_img = ImageEnhance.Brightness(sample_img).enhance(np.random.randint(5, 25) / 10.0)

        # horizontal flip
        if augmentation == 1:
            sample_img = sample_img.transpose(Image.FLIP_LEFT_RIGHT)
            width = sample_img.size[0]
            flipped_boxes = sample_bboxes.copy()
            # sample_bboxes = [float(width), float(top), float(left), float(top)] - flipped_bboxes
            sample_bboxes[:, 0] = width - flipped_boxes[:, 2]
            sample_bboxes[:, 2] = width - flipped_boxes[:, 0]
            # flipped_boxes = sample_bboxes.copy()
            # sample_bboxes[:, 0] = flipped_boxes[:, 2]
            # sample_bboxes[:, 2] = flipped_boxes[:, 0]

        if augmentation == 2:
            if random.choice([True, False]) == True:
                sample_img = sample_img.filter(ImageFilter.BLUR)
            else:
                sample_img = sample_img.filter(ImageFilter.SHARPEN)

        # if augmentation == 3:
        #     w, h = sample_img.size[:2]
        #     left = np.random.randint(0, np.min(sample_bboxes[:, 0])-(np.min(sample_bboxes[:, 0])/5).astype(int))
        #     # print("left---------------",left)
        #     top = np.random.randint(0, np.min(sample_bboxes[:, 1])-(np.min(sample_bboxes[:, 1])/5).astype(int))
        #     right = np.random.randint(np.max(sample_bboxes[:, 2])+((w-np.max(sample_bboxes[:, 2]))/5).astype(int), w)
        #     # print("right--------------",right)
        #     bottom = np.random.randint( np.max(sample_bboxes[:, 3])+((h-np.max(sample_bboxes[:, 3]))/5).astype(int), h)
        #     # print("bottom-------------",bottom)
        #
        #     sample_img = sample_img.crop((left, top, right, bottom))
        #     # print(sample_bboxes[0])
        #     # print("left", left)
        #     sample_bboxes = sample_bboxes - [float(left), float(top), float(left), float(top)]
        #     # print(sample_bboxes[0])



        # 2. Normalize the image with self.mean and self.std
        img = sample_img.resize((300, 300))
        img_array = np.asarray(img)
        img_array = (img_array-self.mean)/self.std
        h, w, c = img_array.shape[0], img_array.shape[1], img_array.shape[2]

        # 3. Convert the bounding box from corner form (left-top, right-bottom): [(x,y), (x+w, y+h)] to
        #    center form: [(center_x, center_y, w, h)]
        #print([sample_img.size[0],sample_img.size[1],sample_img.size[0],sample_img.size[1]])
        sample_bboxes = torch.Tensor(sample_bboxes)/torch.Tensor([sample_img.size[0],sample_img.size[1],sample_img.size[0],sample_img.size[1]])

        # 4. Normalize the bounding box position value from 0 to 1
        sample_bboxes = corner2center(sample_bboxes)

        # 4. Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box
        # TODO: data augmentation
        # 5. Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes.cuda(), sample_bboxes.cuda(), torch.Tensor(sample_labels).cuda(), iou_threshold=0.5)
        img_tensor = torch.Tensor(img_array)
        img_tensor = img_tensor.view(c, h, w)
        #print(img_tensor.shape)
        # [DEBUG] check the output.
        assert isinstance(bbox_label_tensor, torch.Tensor)
        assert isinstance(bbox_tensor, torch.Tensor)
        assert bbox_tensor.dim() == 2
        assert bbox_tensor.shape[1] == 4
        assert bbox_label_tensor.dim() == 1
        assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]
        temp = bbox_tensor.cpu()
        return img_tensor, bbox_tensor, bbox_label_tensor