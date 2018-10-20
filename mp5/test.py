import numpy as np
import time
import torch
import pickle
from scipy import spatial
from sklearn.neighbors import KNeighborsClassifier
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

class TripleDataset(Dataset):
    def __init__(self,triplelist, root_dir,train, transform=None):
        self.triplelist = pickle.load(open(triplelist,'rb'))
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.triplelist)

    def __getitem__(self, idx):
        if self.train :
            label = self.triplelist[idx][0].split('_')[0]
            subdir = label
            query_image_path = self.root_dir + '/'+ subdir + '/images/' + self.triplelist[idx][0]
            positive_image_path = self.root_dir + '/'+ subdir + '/images/' + self.triplelist[idx][1]
            negative_label = self.triplelist[idx][2].split('_')[0]
            subdir = negative_label
            negative_image_path = self.root_dir + '/'+ subdir + '/images/' + self.triplelist[idx][2]
            positive_image = Image.open(positive_image_path).convert('RGB')
            query_image = Image.open(query_image_path).convert('RGB')
            negative_image = Image.open(negative_image_path).convert('RGB')
            sample = {'positive_image': positive_image, 'query_image': query_image, 'negative_image' : negative_image}
            if self.transform:
                for i,v in sample.items():
                    sample[i] = self.transform(v)
            return sample, self.triplelist[idx]
        else:
            query_image_path = self.root_dir + self.triplelist[idx]
            label_list = pickle.load(open("testlist_label.pkl", 'rb'))
            query_image = Image.open(query_image_path).convert('RGB')
            if self.transform:
                sample = self.transform(query_image)
            return sample, label_list[idx]

def test(embedding_array,train_image_name):
    embedding_size = 4096
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
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
    print(embedding_array.shape, "embedding array shape")
    train_image_name = np.load(open(train_image_name, 'rb'))
    print(len(train_image_name), "image_label_length")
    time3 = time.time()
    net.eval()
    testset = TripleDataset(triplelist = 'testlist.pkl', root_dir = 'tiny-imagenet-200/val/images/', train = 0,
                             transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size= 128,shuffle=True, num_workers = 32)
    #label_list = pickle.load(open("testlist_label.pkl", 'rb'))
    #tree_array = np.vstack((outputs, embedding_array))
    neigh = KNeighborsClassifier(n_neighbors=30, n_jobs= 32 )

    test_output =[]
    test_label = []
    for i, data in enumerate(testloader, 0):
        # get the inputs
        if i == 2:
            break
        inputs, labels = data

        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs_c = outputs.cpu().data.numpy()
        del outputs
        #print(outputs_c.shape)
        for s_label in range(outputs_c.shape[0]):
            test_output.append(outputs_c[s_label])
            test_label.append(labels[s_label])
        progress_bar(i, len(testloader))
    accuracy = 0
    neigh.fit(embedding_array[:32], train_image_name[:32])

    for i, data in enumerate(outputs_c):
        test_array = np.repeat(test_label[i], 30, axis = 0)
        print(data.shape)
        labels = np.asarray(neigh.predict(data.reshape(1, -1)))
        count = np.sum(labels == test_array)
        tmp_accuracy = count/30
        accuracy += tmp_accuracy
        progress_bar(i, len(outputs_c))
    print("average acc of testing: ", (accuracy/100)/100000)
    print('One time: ', time.time()- time3)

test('embedding.pkl', 'train_image_name.pkl')
