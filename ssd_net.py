import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.module_util as module_util
from mobilenet import MobileNet


class SSD(nn.Module):
    
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes

        # Setup the backbone network (base_net)
        self.base_net = MobileNet(num_classes)

        # The feature map will extracted from layer[11] and layer[13] in (base_net)
        self.base_output_layer_indices = (5, 11, 12)

        # Define the Additional feature extractor
        self.additional_feat_extractor = nn.ModuleList([
            # Conv8_2
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            # Conv9_2
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
                nn.ReLU()
            )
        ])

        # Bounding box offset regressor
        num_prior_bbox = 6                                                               # num of prior bounding boxes
        self.loc_regressor = nn.ModuleList([
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1)
        ])

        # Bounding box classification confidence for each label
        self.classifier = nn.ModuleList([
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1)
        ])

        # Todo: load the pre-trained model for self.base_net, it will increase the accuracy by fine-tuning
        temp_state = torch.load('pretrained/mobienetv2.pth')
        #self.base_net.load_state_dict(cur_state)

        cur_dict = self.base_net.state_dict()
        input_state = {k: v for k, v in temp_state.items() if
                       k in cur_dict and v.size() == cur_dict[k].size()}
        cur_dict.update(input_state)
        self.base_net.load_state_dict(cur_dict)

        def init_with_xavier(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        self.loc_regressor.apply(init_with_xavier)
        self.classifier.apply(init_with_xavier)
        self.additional_feat_extractor.apply(init_with_xavier)

    def feature_to_bbbox(self, loc_regress_layer, confidence_layer, input_feature):
        """
        Compute the bounding box class scores and the bounding box offset
        :param loc_regress_layer: offset regressor layer to run forward
        :param confidence_layer: confidence layer to run forward
        :param input_feature: feature map to be feed in
        :return: confidence and location, with dim:(N, num_priors, num_classes) and dim:(N, num_priors, 4) respectively.
        """
        conf = confidence_layer(input_feature)
        loc = loc_regress_layer(input_feature)

        # Confidence post-processing:
        # 1: (N, num_prior_bbox * n_classes, H, W) to (N, H*W*num_prior_bbox, n_classes) = (N, num_priors, num_classes)
        #    where H*W*num_prior_bbox = num_priors
        conf = conf.permute(0, 2, 3, 1).contiguous()
        num_batch = conf.shape[0]
        c_channels = int(conf.shape[1]*conf.shape[2]*conf.shape[3] / self.num_classes)
        #print('conf shape',conf.shape)

        conf = conf.view(num_batch, c_channels, self.num_classes)


        # Bounding Box loc and size post-processing
        # 1: (N, num_prior_bbox*4, H, W) to (N, num_priors, 4)
        loc = loc.permute(0, 2, 3, 1).contiguous()
        #print('loc shape',loc.shape)
        l_channels = int(loc.shape[1] * loc.shape[2] * loc.shape[3] / 4)
        #print('l chanel', l_channels)
        loc = loc.view(num_batch, l_channels, 4)

        return conf, loc

    def forward(self, input):

        confidence_list = []
        loc_list = []

        # Run the backbone network from [0 to 11, and fetch the bbox class confidence
        # as well as position and size
        y = module_util.forward_from(self.base_net.base_net, 0, self.base_output_layer_indices[0]+1, input)
        #print('y',y.shape)
        confidence, loc = self.feature_to_bbbox(self.loc_regressor[0], self.classifier[0], y)
        confidence_list.append(confidence)
        loc_list.append(loc)
        #print('cof, loc size', confidence.shape, loc.shape)

        # Todo: implement run the backbone network from [11 to 13] and compute the corresponding bbox loc and confidence
        y = module_util.forward_from(self.base_net.base_net, self.base_output_layer_indices[0], self.base_output_layer_indices[1] + 1, y)
        #print('y', y.shape)
        confidence, loc = self.feature_to_bbbox(self.loc_regressor[1], self.classifier[1], y)
        confidence_list.append(confidence)
        loc_list.append(loc)
        #print('cof, loc size', confidence.shape, loc.shape)

        #conv to 12
        #y = module_util.forward_from(self.base_net.base_net, self.base_output_layer_indices[1], self.base_output_layer_indices[2]+1, y)
        # Todo: forward the 'y' to additional layers for extracting coarse features
        for idx in range (0, len(self.additional_feat_extractor)) :
            #print('current idx', idx)
            #print('y', y.shape)
            y = module_util.forward_from(self.additional_feat_extractor[idx], 0, 4, y)
            confidence, loc = self.feature_to_bbbox(self.loc_regressor[idx+2], self.classifier[idx+2], y)
            confidence_list.append(confidence)
            loc_list.append(loc)
            #print('cof, loc size', confidence.shape, loc.shape)


        confidences = torch.cat(confidence_list, 1)
        locations = torch.cat(loc_list, 1)
        #print('cof, loc size after cat',  np.asarray(confidences).shape, np.asarray(locations).shape)

        # [Debug] check the output
        assert confidences.dim() == 3  # should be (N, num_priors, num_classes)
        assert locations.dim() == 3   # should be (N, num_priors, 4)
        assert confidences.shape[1] == locations.shape[1]
        assert locations.shape[2] == 4

        if not self.training:
            # If in testing/evaluating mode, normalize the output with Softmax
            confidences = F.softmax(confidences, dim=2)

        return confidences, locations



