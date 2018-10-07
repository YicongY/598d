import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
class B_Block(nn.Module):
    def __init__(self, inlayer, outlayer, filter_size = 3, first_stride = 1, padding = 1, downsample_net = None):
        super(B_Block, self).__init__()
        self.conv1 = nn.Conv2d(inlayer, outlayer, filter_size, first_stride, padding)
        self.conv1_bn = nn.BatchNorm2d(outlayer)
        self.conv2 = nn.Conv2d(outlayer, outlayer, filter_size, 1, padding)
        self.conv2_bn = nn.BatchNorm2d(outlayer)
        self.downsample_net = downsample_net

    def forward(self,x):
        out = self.conv1(x)
        if self.downsample_net:
            x = self.downsample_net(x)

        out = self.conv1_bn(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = x + out

        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inputplane = 32
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_dropout = nn.Dropout(0.6)
        self.bb1 = self.block_layer(32, 2, 1)
        self.bb2 = self.block_layer(64, 4, 2)
        self.bb3 = self.block_layer(128, 4, 2)
        self.bb4 = self.block_layer(256, 2, 2)
        self.max_pol = nn.MaxPool2d(4, 1)
        self.fc = nn.Linear(256, 128)
        self.fc1 = nn.Linear(128, 100)
        self.fc2 = nn.Linear(256, 100)



    def block_layer(self, out_layers, num_layer, stride):
        downsample_net = None
        if stride != 1:
            downsample_net = nn.Sequential(nn.Conv2d(self.inputplane, out_layers, kernel_size=3, stride=stride, padding = 1, bias=False), nn.BatchNorm2d(out_layers),)
        block = []
        block.append(B_Block(self.inputplane, out_layers, 3, stride, 1, downsample_net))
        self.inputplane = out_layers
        for i in range(1, num_layer):
            block.append(B_Block(self.inputplane, out_layers))
        return nn.Sequential(*block)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv1_dropout(x)
        x = self.bb1(x)
        x = self.bb2(x)
        x = self.bb3(x)
        x = self.bb4(x)
        x = self.max_pol(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        #x = self.fc1(F.relu(x))
        #x = self.fc(x)
        x = self.fc2(x)
        return x


def main():
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         #transforms.RandomCrop(32),
         transforms.RandomRotation(25),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers = 0)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers = 0)


    net = ResNet()
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
    net.train()
    for epoch in range(70):  # loop over the dataset multiple times
        net.train()
        time2 = time.time()
        class_correct = list(0. for i in range(100))
        class_total = list(0. for i in range(100))
        running_loss = 0.0
        if epoch >= 5 and epoch % 5 == 0:
            test(testloader, net, device)
        if (epoch > 6):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if ('step' in state and state['step'] >= 1024):
                        state['step'] = 1000
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
            for j in range(len(labels)):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99 :  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 99))
                running_loss = 0.0
        total_acc = 0
        for i in range(100):
            total_acc += 100 * class_correct[i] / class_total[i]
        print("average trainning acc: ", total_acc/100)
        print('One time: ', time.time() - time2)
    print('Total time: ', time.time() -time1)

    print('Finished Training')
    print('Start Testing')
    ####testing###########
    test(testloader, net, device)

def test(testloader, net,device):
    time3 = time.time()
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))
    net.eval()
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
       # optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for j in range(len(labels)):
            label = labels[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # print statistics
        #running_loss += loss.item()
        # if i % 30 == 29:  # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 29))
        #     running_loss = 0.0
    total_acc = 0
    for i in range(100):
        total_acc += 100 * class_correct[i] / class_total[i]
    print("average acc of testing: ", total_acc/100)
    print('One time: ', time.time()- time3)

main()