import numpy as np
import torch.nn
from torch.utils.data import Dataset
from bbox_helper import generate_prior_bboxes, match_priors


class CityScapeDataset(Dataset):

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list

        # TODO: implement prior bounding box
        self.prior_bboxes = generate_prior_bboxes(prior_layer_cfg='Todo, use your own setting, please refer bbox_helper.py for an example')

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

        # TODO: implement data loading
        # 1. Load image as well as the bounding box with its label
        # 2. Normalize the image with self.mean and self.std
        # 3. Convert the bounding box from corner form (left-top, right-bottom): [(x,y), (x+w, y+h)] to
        #    center form: [(center_x, center_y, w, h)]
        # 4. Normalize the bounding box position value from 0 to 1
        sample_labels = None
        sample_bboxes = None
        sample_img = None

        # 4. Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box

        # 5. Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes, sample_bboxes, sample_labels, iou_threshold=0.5)

        # [DEBUG] check the output.
        assert isinstance(bbox_label_tensor, torch.Tensor)
        assert isinstance(bbox_tensor, torch.Tensor)
        assert bbox_tensor.dim() == 2
        assert bbox_tensor.shape[1] == 4
        assert bbox_label_tensor.dim() == 1
        assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]

        return bbox_tensor, bbox_label_tensor