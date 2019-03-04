import torch
import numpy as np 
import torchvision
trainset = torchvision.datasets.CIFAR100(root='./Datasets/CIFAR100', train=True, download=True, transform=None)
testset = torchvision.datasets.CIFAR100(root='./Datasets/CIFAR100', train=False, download=True, transform=None)
train_data = trainset.train_data
train_labels = trainset.train_labels
test_data = testset.test_data
test_labels = testset.test_labels
np.save('./Datasets/CIFAR100/train.npy',train_data)
np.save('./Datasets/CIFAR100/test.npy',test_data)
np.save('./Datasets/CIFAR100/train_label.npy',train_labels)
np.save('./Datasets/CIFAR100/test_label.npy',test_labels)