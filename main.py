# ------------------------ step 0 : load the necessary packages ------------------------
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
from torchvision.models import resnet34, vgg16
from tensorboardX import SummaryWriter
from config import *
from utils import *
from models import *

# ------------------------ step 1 : define the models / optimizers / metrics ------------------------
net_pretrained = vgg16(pretrained=pretrain_on)
net = FCN8s(num_classes, net_pretrained)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
net.to(device)

optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_num, gamma=0.1)

criterionBCE = nn.BCELoss()
criterionBCE.to(device)

criterionDC = DiceCoef()
criterionDC.to(device)

# ------------------------ step 2 : load and preprocess SOD360 ------------------------
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

        # ************ initialize the log recorder ************
        writer = SummaryWriter(log_dir=log_dir, comment='SOD360')

        # ************ calculate the mean and std of trainSet ************
        normTransformation = data_norm(num_train)

        # ************ load the dataset for training ************
        data_train = MyDataset(ids_imgs_train_path, ids_objms_train_path,
                               normTrans=normTransformation)
        print('************************')
        print('training dataset loaded !')
        print('************************')
        data_val = MyDataset(ids_imgs_val_path, ids_objms_val_path,
                             normTrans=normTransformation)
        print('************************')
        print('validate dataset loaded !')
        print('************************')

        train_loader = DataLoader(dataset=data_train, batch_size=batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(dataset=data_val, batch_size=batch_size,
                                  shuffle=True)

# ------------------------ step 3 : train and validate the models ------------------------
        for epoch in range(Epochs):
            loss_sigma = 0.0
            loss_val = 0.0
            correct = 0.0
            total = 0.0
            correct_val = 0.0
            total_val = 0.0
            scheduler.step()

            net.train()
            for i, data in enumerate(train_loader):
                inputs, masks = data
                inputs, masks = Variable(inputs.to(device)), \
                                Variable(masks.to(device))

                # visulize the model structure
                if epoch == 0:
                    if i == 0:
                        writer.add_graph(net, inputs)

                # forward
                optimizer.zero_grad()
                outputs = net(inputs)

                # compute the loss
                loss = criterionBCE(outputs, masks)

                # compute the metric
                metric = criterionDC(outputs, masks)

                # backward
                loss.backward()
                optimizer.step()

                # statistics
                correct += metric.item()
                total += 1
                loss_sigma += loss.item()

                if i % 1 == 0:
                    loss_avg = loss_sigma / 1
                    loss_sigma = 0.0
                    print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] "
                          "Loss: {:.4f} Acc:{:.2%}".format(epoch + 1, Epochs, i + 1,
                                                           len(train_loader), loss_avg,
                                                           correct / total))

                    # record training loss
                    writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
                    # record learning rate
                    writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
                    # record accuracy
                    writer.add_scalars('Accuracy_group',
                                       {'train_acc': correct / total}, epoch)

                # model visualization
                if i % 9 == 0:
                    # visualize the inputs, masks and outputs
                    show_inputs = make_grid(inputs)
                    show_maps = make_grid(masks)
                    show_outputs = make_grid(outputs)
                    writer.add_image('Input_group', show_inputs)
                    writer.add_image('Map_group', show_maps)
                    writer.add_image('Out_group', show_outputs)

            # record grads and weights
            for name, layer in net.named_parameters():
                writer.add_histogram(name + '_grad',
                                     layer.grad.cpu().data.numpy(), epoch)
                writer.add_histogram(name + '_data',
                                     layer.cpu().data.numpy(), epoch)

            net.eval()
            for i, data in enumerate(valid_loader):
                imgs, masks = data
                imgs, masks = Variable(imgs), Variable(masks)

                # forward
                outputs = net(imgs)
                outputs.detach_()

                # compute loss
                loss = criterionBCE(outputs, masks)

                # compute metric
                metric = criterionDC(outputs, masks)

                loss_val += loss.item()
                correct_val += metric
                total_val += 1

                loss_avg_val = loss_val
                loss_val = 0.0

                print("Validation: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] "
                      "Loss: {:.4f} Acc:{:.2%}".
                      format(epoch + 1, Epochs, i + 1, len(valid_loader),
                             loss_avg_val, correct_val / total_val))

                # record validation loss
                writer.add_scalars('Loss_group',
                                   {'valid_loss': loss_avg_val}, epoch)
                # record validation accuracy
                writer.add_scalars('Accuracy_group',
                                   {'valid_acc': correct_val / total_val}, epoch)

        print('************************')
        print('finished training !')
        print('************************')

        torch.save(net.state_dict(), mod_dir)
        print('************************')
        print('model successfully saved !')
        print('************************')

# ------------------------ step 4 : test and evaluate the results ------------------------
    if test_on:
        data_test = MyDataset(ids_imgs_test_path, ids_objms_test_path)
        print('************************')
        print('testing dataset loaded !')
        print('************************')

