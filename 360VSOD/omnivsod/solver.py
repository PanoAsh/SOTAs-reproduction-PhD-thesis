import torch
from torch.optim import Adam
from torch.autograd import Variable
from model import build_model
import numpy as np
import cv2
import os
from torch.nn import functional as F
import time
from apex import amp
opt_level = 'O1'
from thop import profile
from util import TI2ER
import matplotlib.pyplot as plt


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

            netStatic_dict = self.net.state_dict()
            netTest_dict = torch.load(self.config.model)
            netTest_dict = {
                k: v
                for k, v in netTest_dict.items()
                if k in netStatic_dict and v.shape == netStatic_dict[k].shape
            } # remove the dynamic parameters declared during TI-based training phase
            self.net.load_state_dict(netTest_dict, strict=False)
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
            self.net = build_model(self.config.backbone, self.config.fcn, self.config.mode, self.config.model_type,
                                   self.config.base_level)
        elif self.config.backbone == 'deeplabv3_resnet101':
            self.net = build_model(self.config.backbone, self.config.fcn, self.config.mode, self.config.model_type,
                                   self.config.base_level)
        if self.config.cuda:
            self.net = self.net.cuda()
        self.lr = self.config.lr
        self.wd = self.config.wd
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                                   weight_decay=self.wd)

        self.print_network(self.net, 'GTNet')

        # Apex acceleration
        #self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level=opt_level)

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        f = open('%s/logs/log_file.txt' % self.config.save_fold, 'w')
        f2 = open('%s/logs/loss_file.txt' % self.config.save_fold, 'w')

        for epoch in range(self.config.epoch):
            G_loss = 0
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                if self.config.model_type == 'G':
                    ER_img, ER_msk= data_batch['ER_img'], data_batch['ER_msk']
                    if ER_img.size()[2:] != ER_msk.size()[2:]:
                        print("Skip this batch")
                        continue
                    ER_img, ER_msk = Variable(ER_img), Variable(ER_msk)
                    if self.config.cuda:
                        ER_img, ER_msk = ER_img.cuda(), ER_msk.cuda()

                    img_train, msk_train = ER_img, ER_msk

                elif self.config.model_type == 'L':
                    TI_imgs, TI_msks = data_batch['TI_imgs'], data_batch['TI_msks']
                    TI_imgs, TI_msks = Variable(TI_imgs), Variable(TI_msks)
                    if self.config.cuda:
                        TI_imgs, TI_msks = TI_imgs.cuda(), TI_msks.cuda()

                    img_train, msk_train = TI_imgs, TI_msks

                else:
                    ER_img, ER_msk, TI_imgs, TI_msks = data_batch['ER_img'], data_batch['ER_msk'], \
                                                       data_batch['TI_imgs'], data_batch['TI_msks']
                    print('under built...')

                # FCN-backbone part
                # sal_ER = TI2ER(sal[0], self.config.base_level, self.config.sample_level) # ER mask as supervision
                sal = self.net(img_train)
                loss_currIter = F.binary_cross_entropy_with_logits(sal, msk_train) \
                                / (self.config.nAveGrad * self.config.batch_size)
                G_loss += loss_currIter.data
                #with amp.scale_loss(ER_loss, self.optimizer) as scaled_loss: scaled_loss.backward()
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
                            * self.config.batch_size / self.config.showEvery) + '  ||  lr:  ' +
                                str(self.lr) + '\n')
                        f2.write(str(epoch) + '_' + '%10.4f' % (G_loss * self.config.nAveGrad *
                                                                self.config.batch_size / self.config.showEvery) + '\n')

                        G_loss = 0

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))

            if (epoch + 1) % self.config.lr_decay_epoch == 0:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      lr=self.lr, weight_decay=self.wd)

        f.close()
        f2.close()
        torch.save(self.net.state_dict(), '%s/models/final_bone.pth' % self.config.save_fold)

    def test(self):
        time_total = 0.0

        for i, data_batch in enumerate(self.test_loader):
            if self.config.model_type == 'G':
                ER_img, img_name = data_batch['ER_img'], data_batch['frm_name']
                ER_img = Variable(ER_img)
                if self.config.cuda: ER_img = ER_img.cuda()
                img_test = ER_img
            elif self.config.model_type == 'L':
                TI_imgs, img_name = data_batch['TI_imgs'], data_batch['frm_name']
                TI_imgs = Variable(TI_imgs)
                if self.config.cuda: TI_imgs = TI_imgs.cuda()
                img_test = TI_imgs
            else:
                print('under built...')

            with torch.no_grad():
                time_start = time.time()
                sal = self.net(img_test)
                torch.cuda.synchronize()
                time_end = time.time()
                time_total = time_total + time_end - time_start

                if self.config.model_type == 'G':
                    pred = np.squeeze(torch.sigmoid(sal[-1]).cpu().data.numpy())
                elif self.config.model_type == 'L':
                    sal = sal[0,:,0,:,:]
                    for idx in range(sal.size()[0]):
                        sal[idx,:,:] = torch.sigmoid(sal[idx,:,:])
                    sal = sal.unsqueeze(0)
                    sal_ER = TI2ER(sal, self.config.base_level, self.config.sample_level)
                    pred = np.squeeze(sal_ER[-1].cpu().data.numpy())

                pred = 255 * pred
                cv2.imwrite(self.config.test_fold + img_name[0], pred)

        print("--- %s seconds ---" % (time_total))
        print('Test Done!')
