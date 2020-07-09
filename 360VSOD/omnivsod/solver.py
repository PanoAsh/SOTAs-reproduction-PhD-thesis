import torch
from torch.optim import Adam
from torch.autograd import Variable
from model import build_model
import numpy as np
import cv2
import os
from torch.nn import utils, functional as F


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        if config.visdom:
            print('under built...')
        self.build_model()
        if self.config.pre_trained != '':
            self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model))
            self.net.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        if self.config.backbone == 'fcn_resnet101':
            self.net = build_model(self.config.backbone, self.config.fcn, self.config.mode)
        elif self.config.backbone == 'deeplabv3_resnet101':
            self.net = build_model(self.config.backbone, self.config.deeplab, self.config.mode)
        if self.config.cuda:
            self.net = self.net.cuda()
        self.lr = self.config.lr
        self.wd = self.config.wd
        self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                                   weight_decay=self.wd)

        self.print_network(self.net, 'GLOmniNet')

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0

        for epoch in range(self.config.epoch):
            gl_loss = 0
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                ER_img, ER_msk= data_batch['ER_img'], data_batch['ER_msk']
                if ER_img.size()[2:] != ER_msk.size()[2:]:
                    print("Skip this batch")
                    continue
                ER_img, ER_msk = Variable(ER_img), Variable(ER_msk)
                if self.config.cuda: 
                    ER_img, ER_msk = ER_img.cuda(), ER_msk.cuda()

                # ERP part
                ER_sal = self.net(ER_img)
                ER_loss = F.binary_cross_entropy_with_logits(ER_sal, ER_msk, reduction='sum') \
                          / (self.config.nAveGrad * self.config.batch_size)

                gl_loss += ER_loss.data
                ER_loss.backward()
                aveGrad += 1

                if aveGrad % self.config.nAveGrad == 0:
                    self.optimizer_bone.step()
                    self.optimizer_bone.zero_grad()
                    aveGrad = 0

                if i % self.config.showEvery == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sum : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num,
                        gl_loss * (self.config.nAveGrad * self.config.batch_size) / self.config.showEvery))

                    print('Learning rate: ' + str(self.lr))
                    gl_loss = 0

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d_bone.pth' %
                           (self.config.save_fold, epoch + 1))

            if epoch % self.config.lr_decay_epoch == 0:
                self.lr_bone = self.lr_bone * 0.1
                self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                           lr=self.lr, weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final_bone.pth' % self.config.save_fold)
