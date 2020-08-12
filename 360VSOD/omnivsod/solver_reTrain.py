import torch
from torch.optim import Adam, SGD
from torch.autograd import Variable
from model import build_model
import numpy as np
import cv2
import os
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt
#from pthflops import count_ops
import torch.nn as nn
#from flopth import flopth


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

import retrain.BASNet.pytorch_ssim as pytorch_ssim
import retrain.BASNet.pytorch_iou as pytorch_iou
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):
    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)
    loss = bce_out + ssim_out + iou_out

    return loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):
    loss0 = bce_ssim_loss(d0,labels_v)
    loss1 = bce_ssim_loss(d1,labels_v)
    loss2 = bce_ssim_loss(d2,labels_v)
    loss3 = bce_ssim_loss(d3,labels_v)
    loss4 = bce_ssim_loss(d4,labels_v)
    loss5 = bce_ssim_loss(d5,labels_v)
    loss6 = bce_ssim_loss(d6,labels_v)
    loss7 = bce_ssim_loss(d7,labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa

    return loss0, loss

CE = nn.BCEWithLogitsLoss()

class SolverReTrain(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        # build model with corresponding init weights: define optimizer
        if self.config.benchmark_name == 'F3Net':
            from retrain.F3Net.retrain import model
            self.net = model
            self.print_network(self.net, 'F3Net')

            if self.config.fine_tune == True:
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/F3Net/fine_tune_init/model-32'))
                print('fine tuning ...')

        elif self.config.benchmark_name == 'BASNet':
            from retrain.BASNet.retrain import model
            self.net = model
            self.print_network(self.net, 'BASNet')

            if self.config.fine_tune == True:
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/BASNet/fine_tune_init/basnet.pth'))
                print('fine tuning ...')

        elif self.config.benchmark_name == 'CPD':
            from retrain.CPD.retrain import model
            self.net = model
            self.print_network(self.net, 'CPD')

            if self.config.fine_tune == True:
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/CPD/fine_tune_init/CPD-R.pth'))
                print('fine tuning ...')

        if self.config.cuda: self.net = self.net.cuda()
        self.lr = self.config.lr
        self.wd = self.config.wd

        # optimizer
        if self.config.optimizer_name == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                              weight_decay=self.wd)
        elif self.config.optimizer_name == 'SGD':
            if self.config.benchmark_name == 'F3Net':
                base, head = [], []
                for name, param in self.net.named_parameters():
                    if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
                        print(name)
                    elif 'bkbone' in name:
                        base.append(param)
                    else:
                        head.append(param)
                self.optimizer = SGD([{'params': base}, {'params': head}], lr=self.lr, momentum=0.9,
                                            weight_decay=self.wd, nesterov=True)
                self.optimizer.param_groups[0]['lr'] = self.lr * 0.1
                self.optimizer.param_groups[1]['lr'] = self.lr

        print('Now we are trying to retrain the benchmark models...')

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        f = open('%s/logs/num_paras.txt' % self.config.save_fold, 'w')
        f.write('name:  ' + name + '    number of parameters:  ' + str(num_params))
        f.close()

    def train(self):
        f = open('%s/logs/log_file.txt' % self.config.save_fold, 'w')
        f2 = open('%s/logs/loss_file.txt' % self.config.save_fold, 'w')
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0

        for epoch in range(self.config.epoch):
            G_loss = 0
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                ER_img, ER_msk = data_batch['ER_img'], data_batch['ER_msk']
                if ER_img.size()[2:] != ER_msk.size()[2:]:
                    print("Skip this batch")
                    continue
                ER_img, ER_msk = Variable(ER_img), Variable(ER_msk)
                if self.config.cuda:
                    ER_img, ER_msk = ER_img.cuda(), ER_msk.cuda()
                img_train, msk_train = ER_img, ER_msk

                if self.config.benchmark_name == 'F3Net':
                    out1u, out2u, out2r, out3r, out4r, out5r = self.net(img_train)
                    loss1u = structure_loss(out1u, msk_train)
                    loss2u = structure_loss(out2u, msk_train)
                    loss2r = structure_loss(out2r, msk_train)
                    loss3r = structure_loss(out3r, msk_train)
                    loss4r = structure_loss(out4r, msk_train)
                    loss5r = structure_loss(out5r, msk_train)
                    loss = (loss1u + loss2u) / 2 + loss2r / 2 + loss3r / 4 + loss4r / 8 + loss5r / 16
                elif self.config.benchmark_name == 'BASNet':
                    d0, d1, d2, d3, d4, d5, d6, d7 = self.net(img_train)
                    loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, msk_train)
                elif self.config.benchmark_name == 'CPD':
                    atts, dets = self.net(img_train)
                    loss1 = CE(atts, msk_train)
                    loss2 = CE(dets, msk_train)
                    loss = loss1 + loss2

                loss_currIter = loss / (self.config.nAveGrad * self.config.batch_size)
                G_loss += loss_currIter.data
                # with amp.scale_loss(ER_loss, self.optimizer) as scaled_loss: scaled_loss.backward()
                loss_currIter.backward()
                aveGrad += 1

                if aveGrad % self.config.nAveGrad == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % self.config.showEvery == 0:
                    if i > 0:
                        print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %10.4f' % (
                            epoch, self.config.epoch, i, iter_num, G_loss * self.config.nAveGrad
                            * self.config.batch_size / self.config.showEvery))
                        print('Learning rate: ' + str(self.lr))
                        f.write('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %10.4f' % (
                            epoch, self.config.epoch, i, iter_num, G_loss * self.config.nAveGrad
                            * self.config.batch_size / self.config.showEvery) + '  ||  lr:  ' + str(self.lr) + '\n')
                        f2.write(str(epoch) + '_' + '%10.4f' % (G_loss * self.config.nAveGrad *
                                                            self.config.batch_size / self.config.showEvery) + '\n')

                        G_loss = 0

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(),
                           '%s/models/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))

        f.close()
        f2.close()
        torch.save(self.net.state_dict(), '%s/models/final_bone.pth' % self.config.save_fold)
