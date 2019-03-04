# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
import PIL
from torch.utils.data import Dataset 


class dataset_train_group(Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.train = np.load(self.data_dir + '/train_group.npy').astype(np.uint8)
        self.train_label = np.load(self.data_dir + '/train_group_label.npy')
        #self.train = torch.Tensor(self.train)
        #self.train_label = torch.IntTensor(self.train_label)
        #self.train = torch.from_numpy(self.train)
        #self.train_label = torch.from_numpy(self.train_label)
    
    def __getitem__(self, index):
        img ,target = self.train[index], self.train_label[index]
        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.train.shape[0]


class dataset_test_group(Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.test = np.load(self.data_dir + '/test_group.npy').astype(np.uint8)
        self.test_label = np.load(self.data_dir + '/test_group_label.npy')
        #self.test = torch.Tensor(self.test)
        #self.test_label = torch.IntTensor(self.test_label)
        #self.test = torch.from_numpy(self.test.astype(np.float32))
        #self.test_label = torch.from_numpy(self.test_label)
    
    def __getitem__(self, index):
        img ,target = self.test[index], self.test_label[index]
        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.test.shape[0]

def creat_dataset_group(original_data_dir,used_clusters,group_order,all_num_clusters):

    original_data_dir = original_data_dir
    used_clusters = used_clusters
    group_order = group_order
    all_num_clusters = all_num_clusters


    train_data = np.load(original_data_dir + '/train.npy')
    train_label = np.load(original_data_dir + '/train_label.npy')
    test_data = np.load(original_data_dir + '/test.npy')
    test_label = np.load(original_data_dir + '/test_label.npy')
    #train_itemindex = np.squeeze(np.argwhere(train_label==9))
    #test_itemindex = np.squeeze(np.argwhere(test_label==9))
    if group_order == all_num_clusters:
        train_itemindex = np.squeeze(np.argwhere(train_label>99))
        test_itemindex = np.squeeze(np.argwhere(test_label>99))
        train_group = np.delete(train_data,train_itemindex,axis=0)
        train_group_label = np.delete(train_label,train_itemindex,axis=0)
        test_group = np.delete(test_data,test_itemindex,axis=0)
        test_group_label = np.delete(test_label,test_itemindex,axis=0)
    else:
        train_exist = np.squeeze(np.isin(train_label,used_clusters))
        test_exist = np.squeeze(np.isin(test_label,used_clusters))
        train_group = train_data[train_exist,:]
        train_group_label = train_label[train_exist]
        test_group = test_data[test_exist,:]
        test_group_label = test_label[test_exist]
        train_group_label =convert_to_range(train_group_label)
        test_group_label = convert_to_range(test_group_label)
    np.save(original_data_dir + '/train_group.npy',train_group)
    np.save(original_data_dir + '/train_group_label.npy',train_group_label)
    np.save(original_data_dir + '/test_group.npy',test_group)
    np.save(original_data_dir + '/test_group_label.npy',test_group_label)



def convert_to_range(x):
    original_array = x
    no_repeat_array = np.array(list(set(x)))
    for i in range(len(original_array)):
        where_c = np.where(no_repeat_array==original_array[i])
        original_array[i] = where_c[0]
    return original_array


def cal_mean_std_group(data_dir):
    data_dir = data_dir
    train = np.load(data_dir + '/train_group.npy')
    train_label = np.load(data_dir + '/train_group_label.npy')
    means = tuple(train.mean(axis=(0,1,2))/255.0)
    stds = tuple(train.std(axis=(0,1,2))/255.0)
    return means,stds




