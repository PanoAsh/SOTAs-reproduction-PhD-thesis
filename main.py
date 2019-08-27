#------------------------ step 0 : load the necessary packages ------------------------
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
import glob
import random
import shutil
from config import *
from utils import *
from models import *

#------------------------ step 1 : load and preprocess SOD360 ------------------------
if __name__ == '__main__':

    # ************ update the dataset info ************
    if update_dts:
        TXT_gnr(ids_imgs_train_path, imgs_train_path)
        TXT_gnr(ids_imgs_val_path, imgs_val_path)
        TXT_gnr(ids_imgs_test_path, imgs_test_path)
        TXT_gnr(ids_objms_train_path, objms_train_path)
        TXT_gnr(ids_objms_val_path, objms_val_path)
        TXT_gnr(ids_objms_test_path, objms_test_path)
        print('************************')
        print('dataset updated !')
        print('************************')

    # ************ load the dataset for training ************
    if train_on:

        # ************ calculate the mean and std of trainSet ************
        normTransformation = data_norm(num_train)

        # ************ load the dataset for training ************
        data_train = MyDataset(ids_imgs_train_path, ids_objms_train_path,
                               myTrans=transform_on, normTrans=normTransformation)
        print('************************')
        print('training dataset loaded !')
        print('************************')
        data_val = MyDataset(ids_imgs_val_path, ids_objms_val_path,
                             myTrans=transform_on, normTrans=normTransformation)
        print('************************')
        print('validate dataset loaded !')
        print('************************')

        train_loader = DataLoader(dataset=data_train, batch_size=batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(dataset=data_val, batch_size=batch_size,
                                  shuffle=True)

#------------------------ step 2 : define the models ------------------------

#------------------------ step 3 : train and validate the models ------------------------
        for epoch in range(Epochs):
            loss_sigma = 0.0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(train_loader):
                print('debugging done!')

#------------------------ step 4 : test and evaluate the results ------------------------
    if test_on:
        data_test = MyDataset(ids_imgs_test_path, ids_objms_test_path)
        print('************************')
        print('testing dataset loaded !')
        print('************************')

