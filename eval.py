from matplotlib import patches
from PIL import Image
import os
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from bbox_helper import nms_bbox1, center2corner, loc2bbox
from ssd_net import SSD
import torch.nn.functional as F
from bbox_helper import generate_prior_bboxes
import sys



# def load_model() :
#     #defining the model
#     net = SSD(3)
#     net = torch.load(os.path.join('../', 'ssd_net.pth'))
#
#     return model

def plot_output(image, sel_bboxes) :

    fig, ax = plt.subplots(1)
    # imageB_array = resize(image, (600, 1200), anti_aliasing=True)
    ax.imshow(image, cmap='brg')
    for i in range(0, sel_bboxes.shape[0]):
        rect = patches.Rectangle((sel_bboxes[i, 0] * image.size[0], sel_bboxes[i, 1] * image.size[0]),
                                 (sel_bboxes[i, 2] - sel_bboxes[i, 0]) * image.size[0],
                                 (sel_bboxes[i, 3] - sel_bboxes[i, 1]) * image.size[0], linewidth=1, edgecolor='r',
                                 facecolor='none')  # Create a Rectangle patch
        ax.add_patch(rect)  # Add the patch to the Axes
    plt.show()



def main() :
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
    prior_bboxes = generate_prior_bboxes(prior_layer_cfg)

    # loading the test image
    img_file_path = sys.argv[1]
    # img_file_path = 'image.png'
    img = Image.open(img_file_path)
    img = img.resize((300, 300))
    plot_img = img.copy()
    img_array = np.asarray(img)[:,:,:3]
    mean = np.asarray((127, 127, 127))
    std = 128.0
    img_array = (img_array - mean) / std
    h, w, c = img_array.shape[0], img_array.shape[1], img_array.shape[2]
    img_tensor = torch.Tensor(img_array)
    test_input = img_tensor.view(1, c, h, w)

    # # loading test input to run test on
    # test_data_loader = torch.utils.data.DataLoader(test_input,
    #                                                 batch_size=1,
    #                                                 shuffle=True,
    #                                                 num_workers=0)
    # idx, (img) = next(enumerate(test_data_loader))
    # # Setting model to evaluate mode
    net = SSD(2)
    test_net_state = torch.load('ssd_net.pth')
    net.load_state_dict(test_net_state)
    # net.eval()
    net.cuda()
    # Forward
    test_input = Variable(test_input.cuda())
    test_cof, test_loc = net.forward(test_input)

    test_loc = test_loc.detach()
    test_loc_clone = test_loc.clone()

    # normalizing the loss to add up to 1 (for probability)
    test_cof_score = F.softmax(test_cof[0], dim = 1)
    # print(test_cof_score.shape)
    # print(test_cof_score)

    # running NMS
    sel_idx = nms_bbox1(test_loc_clone[0], prior_bboxes, test_cof_score.detach(), overlap_threshold=0.5, prob_threshold=0.24)

    test_loc = loc2bbox(test_loc[0], prior_bboxes)
    test_loc = center2corner(test_loc)

    sel_bboxes = test_loc[sel_idx]

    # plotting the output
    plot_output(plot_img, sel_bboxes.cpu().detach().numpy())

if __name__ == '__main__':
    main()