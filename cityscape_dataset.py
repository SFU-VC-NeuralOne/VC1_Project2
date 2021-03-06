import random

import numpy as np
import torch.nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from bbox_helper import generate_prior_bboxes, match_priors, corner2center, center2corner, bbox2loc


class CityScapeDataset(Dataset):

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
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
        self.prior_bboxes = generate_prior_bboxes(prior_layer_cfg)

        # Pre-process parameters:
        #  Normalize: (I-self.mean)/self.std
        self.mean = np.asarray((127, 127, 127))
        self.std = 128.0

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.dataset_list)

    def crop(self, sample_img, sample_bboxes, sample_labels, iter=0):
        selected_idx = np.zeros(len(sample_bboxes))
        w, h = sample_img.size[:2]

        if h < w:
            diff = w - h
            left = np.random.randint(0, diff)
            top = 0
            right = left + diff
            bottom = h
            if len(sample_labels) == 1:
                sample_bboxes = sample_bboxes.reshape((1, 4))
                if sample_bboxes[0, 0] <= 50:
                    left = 0
                    right = diff
                elif sample_bboxes[0, 2] >= (w - 50):
                    left = diff
                    right = w
        else:
            diff = h - w
            left = 0
            top = np.random.randint(0, diff)
            right = w
            bottom = top + diff
            if len(sample_labels) == 1:
                sample_bboxes = sample_bboxes.reshape((1, 4))
                if sample_bboxes[0, 1] <= 50:
                    top = 0
                    bottom = diff
                elif sample_bboxes[0, 3] >= (h - 50):
                    top = diff
                    bottom = h

        for i in range(0, len(sample_bboxes)):
            if sample_bboxes[i, 0] >= left and sample_bboxes[i, 1] >= top and sample_bboxes[i, 2] <= right and \
                    sample_bboxes[i, 3] <= bottom:
                selected_idx[i] = 1

        if np.sum(selected_idx) == 0:
            recursion_count = iter + 1
            if (recursion_count <= 10):
                new_image, new_bboxes, new_labels = self.crop(sample_img, sample_bboxes, sample_labels, iter=recursion_count)
            else:
                new_image = sample_img.copy()
                new_labels = sample_labels.copy()
                new_bboxes = sample_bboxes.copy()
        else:
            new_image = sample_img.crop((left, top, right, bottom))
            sample_bboxes = sample_bboxes[selected_idx == 1]
            new_labels = sample_labels.copy()
            new_labels = new_labels[selected_idx == 1]
            new_bboxes = sample_bboxes - [float(left), float(top), float(left), float(top)]

        return new_image, new_bboxes, new_labels

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
        sample_labels = np.asarray(ground_truth[0], dtype=np.float32)
        sample_bboxes = np.asarray(ground_truth[1], dtype=np.float32)
        sample_img = Image.open(file_path)

        augmentation = np.random.randint(0, 4)
        sample_img, sample_bboxes, sample_labels = self.crop(sample_img,sample_bboxes,sample_labels)
        #augmentation=None
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
        #self.prior_bboxes = center2corner(self.prior_bboxes)

        # 4. Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box
        # TODO: data augmentation
        # 5. Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes.cuda(), sample_bboxes.cuda(), torch.Tensor(sample_labels).cuda(), iou_threshold=0.45)
        #bbox_tensor, bbox_label_tensor = assign_priors(sample_bboxes.cuda(), torch.Tensor(sample_labels).cuda(), self.prior_bboxes.cuda(), iou_threshold=0.5)


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
        return img_tensor, bbox_tensor, bbox_label_tensor

