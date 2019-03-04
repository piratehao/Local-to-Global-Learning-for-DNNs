# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from torch.autograd import Variable
from models import *
from utils import progress_bar
from selection_strategy import clusters_chosen_random
from dataset_class import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=100, type=int,
                    help='number of total epochs (default: 200)')
parser.add_argument('--save-dir', default='Checkpoint_logger', type=str,
                    help='directory of saved model (default: model/saved)')
parser.add_argument('--data-dir', default='./Datasets/CIFAR100', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--group-num', default=2, type = int, 
                    help='the num of pre-train')

parser.add_argument('--cluster-num', default=100, type = int, 
                    help='cluster number')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Model
print('==> Building model..')
net = VGG('VGG16')
#net = net.to(device)
#if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    #cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def resume_checkpoint_group(net,resume_path):
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    net.load_state_dict(checkpoint['state_dict'])



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    lr = args.lr * pow(0.95,epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    net.train()
    if use_cuda:
        net.cuda()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = torch.FloatTensor(inputs), torch.LongTensor(targets)
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = torch.FloatTensor(inputs), torch.LongTensor(targets)
            inputs, targets = Variable(inputs), Variable(targets)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'state_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('Checkpoint_logger'):
            os.mkdir('Checkpoint_logger')
        torch.save(state, os.path.join(args.save_dir, 'model_best.pth.tar'))
        best_acc = acc



dataset_labels_num = 100
dataset_group_num = args.group_num
group_cut = np.zeros([args.group_num])
for i in range(args.group_num):
    group_cut[i] = math.ceil(dataset_labels_num/float(dataset_group_num))
    dataset_labels_num = dataset_labels_num - int(group_cut[i])
    dataset_group_num = dataset_group_num -1
group_cut = group_cut.astype(np.int64)

for i in range(args.group_num):
    best_acc = 0
    start_epoch = 1
    if i==0:
        clusters_chosen = np.array(random.sample(list(np.arange(args.cluster_num)), group_cut[i]))
        used_clusters = clusters_chosen
    else:
        clusters_chosen = clusters_chosen_random(args.cluster_num, used_clusters,group_cut[i])
        used_clusters = np.append(used_clusters, clusters_chosen)
    creat_dataset_group(args.data_dir, used_clusters,  i+1, args.group_num)
    mean,std = cal_mean_std_group(args.data_dir)
    train_data = dataset_train_group(args.data_dir, transform = transforms.Compose( [transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean,std)] ))
    test_data = dataset_test_group(args.data_dir, transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize(mean,std)]))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size =  args.batch_size, shuffle = True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size =  100, shuffle = False, num_workers=0)
    if i==0:
        net.classifier = nn.Linear(int(512),int(sum(group_cut[:i+1])))
        #print(group_cut[:i+1])
    else:
        resume_path = args.save_dir + '/model_best.pth.tar'
        resume_checkpoint_group(net, resume_path)
        params = net.state_dict()
        weight = params['classifier.weight']
        width,height = weight.shape
        bias = params['classifier.bias']

        net.classifier = nn.Linear(int(512),int(sum(group_cut[:i+1])))

        params_new = net.state_dict()
        weight_new = params_new['classifier.weight']
        bias_new = params_new['classifier.bias']
        weight_new[:width,:] = weight
        bias_new[:width] = bias
        net.load_state_dict(params_new)

    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        test(epoch)

    if i == args.group_num - 1:
        print(best_acc)


















