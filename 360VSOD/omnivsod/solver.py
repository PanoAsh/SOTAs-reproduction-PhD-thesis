import torch
from torch.optim import Adam
from torch.autograd import Variable
from model import build_model
import numpy as np
import cv2
import os
from torch.nn import functional as F
import time


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
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
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
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
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % self.config.showEvery == 0:
                    if i > 0:
                        print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %10.4f' % (
                            epoch, self.config.epoch, i, iter_num, gl_loss / self.config.showEvery)) # batch_size = 1

                        print('Learning rate: ' + str(self.lr))
                        gl_loss = 0

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d_bone.pth' %
                           (self.config.save_fold, epoch + 1))

            if (epoch + 1) % self.config.lr_decay_epoch == 0:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                           lr=self.lr, weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final_bone.pth' % self.config.save_fold)

    def test(self):
        time_total = 0.0

        for i, data_batch in enumerate(self.test_loader):
            ER_img, img_name = data_batch['ER_img'], data_batch['frm_name']

            with torch.no_grad():
                ER_img = Variable(ER_img)
                if self.config.cuda:
                    ER_img = ER_img.cuda()
                time_start = time.time()
                ER_sal = self.net(ER_img)
                torch.cuda.synchronize()
                time_end = time.time()
                time_total = time_total + time_end - time_start
                pred = np.squeeze(torch.sigmoid(ER_sal[-1]).cpu().data.numpy())
                pred = 255 * pred

                cv2.imwrite(self.config.test_fold + img_name[0], pred)

        print("--- %s seconds ---" % (time_total))
        print('Test Done!')
