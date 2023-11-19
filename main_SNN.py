import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.activation_based.model import parametric_lif_net
from copy import deepcopy
from dataloader import get_DVSDataloader
import torch.nn.functional as F
import time
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

class DVSNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 1
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 9 * 8, 128),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(128, 10),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
    
net = DVSNet(channels=64, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

functional.set_step_mode(net, 'm')

# functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

train_dataloader = get_DVSDataloader("./train.txt", 12, num_workers=4, shuffle=True)
test_dataloader = get_DVSDataloader("./test.txt", 1, num_workers=4, shuffle=True)
test_in_dataloader = get_DVSDataloader("./test_in.txt", 1, num_workers=4, shuffle=True)
test_out_dataloader = get_DVSDataloader("./test_out.txt", 1, num_workers=4, shuffle=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
device = torch.device('cuda')
net.to(device)
print("gpu:{}".format(torch.cuda.is_available()))
train_acc_ep=[]
test_acc_ep=[]
testin_acc_ep=[]
testout_acc_ep=[]
for epoch in range(100):
        net.train()
        train_loss = []
        train_acc = []
        for frame, label in train_dataloader:
            optimizer.zero_grad()
            frame = frame.to(device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(device)
            # label_onehot = F.one_hot(label, 2).float()


            out = net(frame).squeeze()[1,:]
            loss = F.mse_loss(out,label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append(torch.mean(((out>0.5)  == label).float()).item())
            functional.reset_net(net)
        print("loss:{} acc:{}".format(sum(train_loss)/len(train_loss), sum(train_acc)/len(train_acc)))
        train_acc_ep.append(sum(train_acc)/len(train_acc))
        with torch.no_grad():
            test_loss = []
            test_acc = []
            # time_d = []
            for frame, label in test_dataloader:
                frame = frame.to(device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(device)
                # label_onehot = F.one_hot(label, 2).float()

                # start_time = time.time()
                out = net(frame).squeeze(-1)[1,:]
                # end_time = time.time()
                # time_d.append(end_time-start_time)
                loss = F.mse_loss(out,label)
                test_loss.append(loss.item())
                test_acc.append(torch.mean(((out>0.5) == label).float()).item())
                functional.reset_net(net)
            print("t loss:{} t acc:{}".format(sum(test_loss)/len(test_loss), sum(test_acc)/len(test_acc)))
            # print("TIME",sum(time_d)/len(time_d))
            test_acc_ep.append(sum(test_acc)/len(test_acc))

            test_loss = []
            test_acc = []
            for frame, label in test_in_dataloader:
                frame = frame.to(device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(device)
                # label_onehot = F.one_hot(label, 2).float()


                out = net(frame).squeeze(-1)[1,:]
                loss = F.mse_loss(out,label)
                test_loss.append(loss.item())
                test_acc.append(torch.mean(((out>0.5) == label).float()).item())
                functional.reset_net(net)
            print("ti loss:{} ti acc:{}".format(sum(test_loss)/len(test_loss), sum(test_acc)/len(test_acc)))
            testin_acc_ep.append(sum(test_acc)/len(test_acc))

            test_loss = []
            test_acc = []
            for frame, label in test_out_dataloader:
                frame = frame.to(device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(device)
                # label_onehot = F.one_hot(label, 2).float()


                out = net(frame).squeeze(-1)[1,:]
                loss = F.mse_loss(out,label)
                test_loss.append(loss.item())
                test_acc.append(torch.mean(((out>0.5) == label).float()).item())
                functional.reset_net(net)
            print("to loss:{} to acc:{}".format(sum(test_loss)/len(test_loss), sum(test_acc)/len(test_acc)))
            testout_acc_ep.append(sum(test_acc)/len(test_acc))

import matplotlib.pyplot as plt
plt.plot(train_acc_ep,label="Training Accuracy")
plt.plot(test_acc_ep,label="Overall Testing Accuracy")
plt.plot(testin_acc_ep, label="In-domain Testing Accuracy")
plt.plot(testout_acc_ep, label="Out-of-domain Testing Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("SNN on DVS Data")
plt.show()