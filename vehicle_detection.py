import json
import random
import re

from cityscape_dataset import CityScapeDataset
from PIL import Image
import os
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

def load_data(picture_path, label_path):
    data_list = []
    vehicle_list = ['car', 'truck', 'bus', 'train','tram', 'motorcycle', 'bicycle', 'caravan', 'trailer']
    human_list = ['person','rider', 'person group', 'rider group']
    for root, dirs, files in os.walk(label_path):
        for name in files:
            if name.endswith((".json")):
                subfolder = name[0:re.search('\d',name).start()-1]
                json_file_path = os.path.join(label_path, subfolder, name)
                img_file_name_idx = name.find('_',20)
                img_file_path = os.path.join(picture_path, subfolder, name[0:img_file_name_idx] + '_leftImg8bit.png')
                #print(img_file_path)
                with open(json_file_path, 'r') as f:
                    frame_info = json.load(f)
                    label = []
                    bbox = []
                    for object in frame_info['objects']:
                        if (object['label'] in vehicle_list):
                            polygons = np.asarray(object['polygon'], dtype=np.float32)
                            left_top = np.min(polygons, axis=0)
                            right_bottom = np.max(polygons, axis=0)
                            label.append(1)     #1 is vehicle
                            bbox.append(np.asarray([left_top, right_bottom]).flatten())
                            # print('left',left_top)
                            # print('right', left_top)
                        if (object['label'] in human_list):
                            polygons = np.asarray(object['polygon'], dtype=np.float32)
                            left_top = np.min(polygons, axis=0)
                            right_bottom = np.max(polygons, axis=0)
                            label.append(2)
                            bbox.append(np.asarray([left_top, right_bottom]).flatten())
                            #classes.append({'class': 'human', 'position':[left_top, right_bottom]})
                    if(len(label)!=0 & len(bbox)):
                        data_list.append({'file_path':img_file_path, 'label':[label,bbox]})
    return data_list

def train(net, train_data_loader, validation_data_loader):
    net.cuda()
    criterion = torch.nn.MSELoss()
    for params in net.features.parameters():
        params.requires_grad = False

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    train_losses = []
    valid_losses = []

    max_epochs = 10
    itr = 0

    for epoch_idx in range(0, max_epochs):
        for train_batch_idx, (train_input, train_label) in enumerate(train_data_loader):
            itr += 1
            net.train()
            optimizer.zero_grad()

            # Forward
            train_input = Variable(train_input.cuda())
            train_out = net.forward(train_input)

            # compute loss
            train_label = Variable(train_label.cuda())
            loss = criterion(train_out, train_label)
            loss.backward()
            optimizer.step()
            train_losses.append((itr, loss.item()))

            # if train_batch_idx % 50 == 0:
            print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))

            # Run the validation every 200 iteration:
            if train_batch_idx % 50 == 0:
                net.eval()
                valid_loss_set = []
                valid_itr = 0

                for valid_batch_idx, (valid_input, valid_label) in enumerate(validation_data_loader):
                    valid_input = Variable(valid_input.cuda())  # use Variable(*) to allow gradient flow
                    valid_out = net.forward(valid_input)  # forward once

                    # compute loss
                    valid_label = Variable(valid_label.cuda())
                    valid_loss = criterion(valid_out, valid_label)
                    valid_loss_set.append(valid_loss.item())

                    valid_itr += 1
                    if valid_itr > 5:
                        break

                # Compute the avg. validation loss
                avg_valid_loss = np.mean(np.asarray(valid_loss_set))
                print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, avg_valid_loss))
                valid_losses.append((itr, avg_valid_loss))

    train_losses = np.asarray(train_losses)
    valid_losses = np.asarray(valid_losses)

    plt.plot(train_losses[:, 0],  # iteration
             train_losses[:, 1])  # loss value
    plt.plot(valid_losses[:, 0],  # iteration
             valid_losses[:, 1])  # loss value
    plt.show()

def main():
    data_list = load_data('')
    random.shuffle(data_list)
    num_total_items = len(data_list)

    # Training set, ratio: 80%
    num_train_sets = 0.8 * num_total_items
    train_set_list = data_list[: int(num_train_sets)]
    validation_set_list = data_list[int(num_train_sets):]
    test_set_list = load_data('')

    # Create dataloaders for training and validation
    train_dataset = CityScapeDataset(train_set_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=120,
                                                    shuffle=True,
                                                    num_workers=4)
    print('Total training items', len(train_dataset), ', Total training mini-batches in one epoch:',
          len(train_data_loader))

    validation_dataset = CityScapeDataset(validation_set_list)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                         batch_size=32,
                                                         shuffle=True,
                                                         num_workers=4)
    print('Total validation items:', len(validation_dataset))

if __name__ == '__main__':
    main()