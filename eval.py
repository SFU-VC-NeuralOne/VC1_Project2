import json
import random
import re
import time

from cityscape_dataset import CityScapeDataset
from PIL import Image
import os
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from bbox_helper import nms_bbox
from ssd_net import SSD
import pickle

import sys



def load_model() :
    #defining the model
    net = SSD(3)

    #ToDo : Load the pretrained model here


def plot_output(image, sel_bboxes) :
    plt.imshow(image)

    fig, ax = plt.subplots(1)
    imageB_array = resize(image, (600, 1200), anti_aliasing=True)
    ax.imshow(imageB_array, cmap='brg')
    sel_bboxes = sel_bboxes.cpu().numpy()
    for i in range(0, sel_bboxes.shape[0]):
        rect = patches.Rectangle((sel_bboxes[i, 0] * 1200, sel_bboxes[i, 1] * 600),
                                 (sel_bboxes[i, 2] - sel_bboxes[i, 0]) * 1200,
                                 (sel_bboxes[i, 3] - sel_bboxes[i, 1]) * 600, linewidth=1, edgecolor='r',
                                 facecolor='none')  # Create a Rectangle patch
        ax.add_patch(rect)  # Add the patch to the Axes
    plt.show()



def main() :
    # loading the test image
    img_file_path = sys.argv[1]
    img = Image.open(img_file_path)
    plot_img = img.copy()

    # loading test input to run test on
    test_data_loader = torch.utils.data.DataLoader(test_image,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    num_workers=0)

    # Setting model to evaluate mode
    net.eval()

    # Forward
    test_input = Variable(test_input.cuda())
    test_cof, test_loc = net.forward(test_input)

    # normalizing the loss to add up to 1 (for probability)
    test_cof_score = F.softmax(test_cof)

    # running NMS
    sel_bboxes = nms_bbox(test_loc, test_cof_score)

    # plotting the output
    plot_output(plot_img, sel_bboxes)

if __name__ == '__main__':
    main()
