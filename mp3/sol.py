import numpy as np
import time
import copy
from random import randint
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 1, 2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv2_max_pol = nn.MaxPool2d(2, 2)
        self.conv2_dropout = nn.Dropout()
        self.conv3 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv4_max_pol = nn.MaxPool2d(2, 2)
        self.conv4_dropout = nn.Dropout()
        self.conv5 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv6_dropout = nn.Dropout()
        self.conv7 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv7_bn = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv8_bn = nn.BatchNorm2d(64)
        self.conv8_dropout = nn.Dropout()
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv2(x)
        x = self.conv2_max_pol(x)
        x = self.conv2_dropout(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.conv4(x)
        x = self.conv4_max_pol(x)
        x = self.conv4_dropout(x)
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = self.conv6(x)
        x = self.conv6_dropout(x)
        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = self.conv8(x)
        x = self.conv8_bn(x)
        x = self.conv8_dropout(x)
        x = x.reshape(x.size(0), -1)
        #print(x.size())
        x = self.fc1(x)
        #print(x.size())
        x = self.fc2(x)
        #print(x.size())
        x = self.fc3(x)
        x = F.softmax(x)
        #print(x.size())
        return x


def main():
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers = 0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers = 0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)
    if torch.cuda.is_available():
        print('cuda')
        device = torch.device('cuda:0')
    else:
        print('cpu')
        device = torch.device('cpu')
    print(device)
    net.to(device)
    time1 = time.time()
    for epoch in range(40):  # loop over the dataset multiple times
        time2 = time.time()
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for j in range(100):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 29))
                running_loss = 0.0
        total_acc = 0
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
            total_acc += 100 * class_correct[i] / class_total[i]
        print("average acc: ", total_acc/10)
        print('One time: ', time.time()- time2)
    print('Total time: ', time.time() -time1)

    print('Finished Training')
    print('Start Testing')
    time3 = time.time()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    running_loss = 0.0
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for j in range(100):
            label = labels[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 30 == 29:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 29))
            running_loss = 0.0
    total_acc = 0
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        total_acc += 100 * class_correct[i] / class_total[i]
    print("average acc of testing: ", total_acc/10)
    print('One time: ', time.time()- time3)

main()