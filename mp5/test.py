import numpy as np
import time
import torch
import pickle
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
import os
from utils import progress_bar
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import sys, getopt
from torchvision import models as t_models

def test(embedding_array,train_image_name):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    net = t_models.resnet18(pretrained=True)
    num_inp = net.fc.in_features
    net.fc = nn.Linear(num_inp, embedding_size)
    if torch.cuda.is_available():
        print('cuda')
        device = torch.device('cuda:0')
    else:
        print('cpu')
        device = torch.device('cpu')

    net.to(device)

    embedding_array = np.load(embedding_array)
    print(embedding_array.shape)
    train_image_name = pickle.load(open(train_image_name, 'rb'))
    print(len(train_image_name))
    time3 = time.time()
    net.eval()
    testset = TripleDataset(triplelist = 'testlist.pkl', root_dir = 'tiny-imagenet-200/val/images/', train = 0,
                             transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size= 128,shuffle=True, num_workers = 32)
    #label_list = pickle.load(open("testlist_label.pkl", 'rb'))
    #tree_array = np.vstack((outputs, embedding_array))
    neigh =NearestNeighbors(n_neighbors=30)

    total_acc = 0
    train_image_name_counter = 0
    for i, data in enumerate(testloader, 0):
        # get the inputs
        accuracy = 0
        inputs, labels = data

        print(len(labels))
        inputs = inputs.to(device)
        outputs = net(inputs)
        print(outputs.shape)
        neigh.fit(samples)
        for ind, group in enumerate(train_image_name):
            for group_ind, image in enumerate(group):
                train_image_name[ind][group_ind] = train_image_name[image].split('_')[0]
        true_label = label_list[i]
        counter = 0
        for ind, group in enumerate(index):
            train_image_name_counter += 32

            counter = np.sum()
        print(train_image_name_counter)
        accuracy = counter/30
        total_acc += accuracy

    print("average acc of testing: ", total_acc/100)
    print('One time: ', time.time()- time3)

test('embedding.pkl', 'train_image_name.pkl')
