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
from bbox_loss import MultiboxLoss
from ssd_net import SSD
import pickle
import torch.nn.functional as F

cityscape_label_dir = '../cityscapes_samples_labels'
cityscape_img_dir ='../cityscapes_samples'
learning_rate = 1e-4
Tuning = False


# cityscape_label_dir = '/home/yza476/SSD/cityscapes_samples_labels'
# cityscape_img_dir ='/home/yza476/SSD/cityscapes_samples'

# cityscape_label_dir = '/home/datasets/full_dataset_labels/train_extra'
# cityscape_img_dir ='/home/datasets/full_dataset/train_extra'

pth_path='../'

def load_data(picture_path, label_path):
    data_list = []
    vehicle_list = ['car', 'truck', 'bus']
    vehicle_grp_list = ['cargroup','truckgroup','busgroup']
    human_list = ['person']
    human_grp_list = ['persongroup']

    # 'train', 'traingroup', 'tram',
    #                 'motorcycle', 'motorcyclegroup', 'bicycle', 'bicyclegroup', 'caravan', 'trailer']

    if os.path.isfile('datalist.pkl'):
        with open('datalist.pkl', 'rb') as f:
            data_list = pickle.load(f)
    else:
        for root, dirs, files in os.walk(label_path):
            for name in files:
                if name.endswith((".json")):
                    subfolder = name[0:re.search('\d',name).start()-1]
                    json_file_path = os.path.join(label_path, subfolder, name)
                    img_file_name_idx = name.find('_',21)
                    if(name[0:img_file_name_idx] == 'troisdorf_000000_000073'):
                        print('skipped:',name[0:img_file_name_idx])
                        continue
                    img_file_path = os.path.join(picture_path, subfolder, name[0:img_file_name_idx] + '_leftImg8bit.png')
                    # print(img_file_path)
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
                            if (object['label'] in vehicle_grp_list):
                                polygons = np.asarray(object['polygon'], dtype=np.float32)
                                left_top = np.min(polygons, axis=0)
                                right_bottom = np.max(polygons, axis=0)
                                label.append(2)     #1 is vehicle
                                bbox.append(np.asarray([left_top, right_bottom]).flatten())
                            if (object['label'] in human_list):
                                polygons = np.asarray(object['polygon'], dtype=np.float32)
                                left_top = np.min(polygons, axis=0)
                                right_bottom = np.max(polygons, axis=0)
                                label.append(3)
                                bbox.append(np.asarray([left_top, right_bottom]).flatten())
                            if (object['label'] in human_grp_list):
                                polygons = np.asarray(object['polygon'], dtype=np.float32)
                                left_top = np.min(polygons, axis=0)
                                right_bottom = np.max(polygons, axis=0)
                                label.append(4)  # 1 is vehicle
                                bbox.append(np.asarray([left_top, right_bottom]).flatten())
                                #classes.append({'class': 'human', 'position':[left_top, right_bottom]})
                        if(len(label)!=0):
                            data_list.append({'file_path':img_file_path, 'label':[label,bbox]})
                            print('file_path',img_file_path)
        with open('datalist.pkl', 'wb') as f:
            pickle.dump(data_list, f)
    return data_list

def train(net, train_data_loader, validation_data_loader):
    net.cuda()
    criterion = MultiboxLoss([0.1,0.2])
    pa_count = 0
    for params in net.base_net.parameters():
        pa_count+=1
        #print(pa_count,params.shape)
        params.requires_grad = False
        if(pa_count>25):
            break

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []
    tr_loc_loss=[]
    tr_conf_loss=[]
    val_loc_loss=[]
    val_conf_loss=[]
    best_valid_loss = 1000

    max_epochs = 30
    itr = 0

    for epoch_idx in range(0, max_epochs):
        for train_batch_idx, (train_input, train_bbox, train_label) in enumerate(train_data_loader):
            itr += 1
            net.train()
            optimizer.zero_grad()

            # Forward
            train_input = Variable(train_input.cuda())
            train_cof, train_loc = net.forward(train_input)

            # compute loss
            train_label = Variable(train_label.cuda())
            loss_cof, loss_loc = criterion(train_cof, train_loc, train_label, train_bbox)
            loss = loss_cof + loss_loc
            #print('training cof loc loss', loss_cof.data, loss_loc.data)
            # loss = loss.sum()
            loss.backward()
            optimizer.step()
            train_losses.append((itr, loss))
            tr_conf_loss.append((itr, loss_cof))
            tr_loc_loss.append((itr, loss_loc))

            # if train_batch_idx % 50 == 0:
            print('Epoch: %d Itr: %d total: %f loc: %f cof: %f' % (epoch_idx, itr, loss.item(), loss_loc.item(), loss_cof.item()))

            # Run the validation every 200 iteration:
            if train_batch_idx % 50 == 0:
                net.eval()
                valid_loss_set = []
                val_cof_set=[]
                val_loc_set=[]
                valid_itr = 0

                for valid_batch_idx, (valid_input, valid_bbox, valid_label) in enumerate(validation_data_loader):
                    valid_input = Variable(valid_input.cuda())  # use Variable(*) to allow gradient flow
                    valid_cof, valid_loc = net.forward(valid_input)  # forward once

                    # compute loss
                    valid_label = Variable(valid_label.cuda())
                    loss_cof, loss_loc = criterion(valid_cof, valid_loc, valid_label, valid_bbox)
                    temp_loss_cof = valid_cof.clone()
                    temp_loss_cof=temp_loss_cof.detach().cpu().numpy()
                    # temp_valid_label = valid_label.clone()
                    # temp_valid_label=temp_valid_label.detach().cpu().numpy()
                    # pred_cof = np.max(temp_loss_cof,axis = 0)
                    #print(np.sum(np.where(temp_valid_label == pred_cof)))
                    # print(F.softmax(valid_cof[0]))
                    # print(np.sum(len(temp_valid_label)))
                    valid_loss = loss_cof+loss_loc
                    # valid_loss = valid_loss.sum()
                    #print('validation cof loc loss', loss_cof.data, loss_loc.data)
                    valid_loss_set.append(valid_loss.item())
                    val_loc_set.append(loss_loc.item())
                    val_cof_set.append(loss_cof.item())
                    valid_itr += 1
                    if valid_itr > 5:
                        break

                # Compute the avg. validation loss
                avg_valid_loss = np.mean(np.asarray(valid_loss_set))
                avg_conf_loss = np.mean(np.asarray(val_cof_set))
                avg_loc_loss = np.mean(np.asarray(val_loc_set))
                print('Valid Epoch: %d Itr: %d total: %f loc: %f cof: %f' % (epoch_idx, itr, avg_valid_loss, avg_loc_loss, avg_conf_loss))
                valid_losses.append((itr, avg_valid_loss))
                val_conf_loss.append((itr, avg_conf_loss))
                val_loc_loss.append((itr, avg_loc_loss))
                if (avg_valid_loss < best_valid_loss):
                    net_state = net.state_dict()  # serialize trained model
                    filename = 'ssd_net.pth'
                    torch.save(net_state, os.path.join(pth_path, filename))
                    best_valid_loss = avg_valid_loss
        with open('train_loc.pkl', 'wb') as f:
            pickle.dump(tr_loc_loss, f)
        with open('train_cof.pkl', 'wb') as f:
            pickle.dump(tr_conf_loss, f)
        with open('val_loc.pkl', 'wb') as f:
            pickle.dump(val_loc_loss, f)
        with open('val_cof.pkl', 'wb') as f:
            pickle.dump(val_conf_loss, f)
    train_losses = np.asarray(train_losses).reshape((-1,2))
    valid_losses = np.asarray(valid_losses).reshape((-1,2))

    net_state = net.state_dict()  # serialize trained model
    timestamp = time.time()
    filename = 'ssd_net'+str(timestamp)+'.pth'
    torch.save(net_state, os.path.join(pth_path, filename))

    plt.plot(train_losses[:, 0], train_losses[:, 1])  # loss value
    plt.plot(valid_losses[:, 0], valid_losses[:, 1])  # loss value
    plt.show()

def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.multiprocessing.set_start_method('spawn')
    data_list = load_data(cityscape_img_dir, cityscape_label_dir)
    random.shuffle(data_list)
    num_total_items = len(data_list)
    net = SSD(5)

    # Training set, ratio: 80%
    num_train_sets = 0.8 * num_total_items
    train_set_list = data_list[: int(num_train_sets)]
    validation_set_list = data_list[int(num_train_sets):]

    # Create dataloaders for training and validation
    train_dataset = CityScapeDataset(train_set_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=8,
                                                    shuffle=True,
                                                    num_workers=0)
    print('Total training items', len(train_dataset), ', Total training mini-batches in one epoch:',
          len(train_data_loader))

    validation_dataset = CityScapeDataset(validation_set_list)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                         batch_size=8,
                                                         shuffle=True,
                                                         num_workers=0)
    print('Total validation items:', len(validation_dataset))
    if Tuning:
        net_state = torch.load(os.path.join(pth_path, 'ssd_net.pth'))
        print('Loading trained model: ', os.path.join(pth_path, 'ssd_net.pth'))
        net.load_state_dict(net_state)
    train(net, train_data_loader, validation_data_loader)

if __name__ == '__main__':
    main()