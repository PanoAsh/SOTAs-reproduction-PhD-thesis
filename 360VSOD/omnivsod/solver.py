import torch
from torch.optim import Adam
from torch.autograd import Variable
from model import build_model
import numpy as np
import cv2
import os


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        if config.visdom:
            print('under built...')
        self.build_model()
        if self.config.pre_trained != '':
            self.net_bone.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net_bone.load_state_dict(torch.load(self.config.model))
            self.net_bone.eval()

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
            self.net_bone = build_model(self.config.backbone, self.config.fcn, self.config.mode)
        elif self.config.backbone == 'deeplabv3_resnet101':
            self.net_bone = build_model(self.config.backbone, self.config.deeplab, self.config.mode)
        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()
        self.lr = self.config.lr
        self.wd = self.config.wd
        self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr,
                                   weight_decay=self.wd)

        self.print_network(self.net_bone, 'GLOmniNet')

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0

        for epoch in range(self.config.epoch):                          
            r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0
        #    self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                ER_img, ER_msk= data_batch['ER_img'], data_batch['ER_msk']
                if ER_img.size()[2:] != ER_msk.size()[2:]:
                    print("Skip this batch")
                    continue
                ER_img, ER_msk = Variable(ER_img), Variable(ER_msk)
                if self.config.cuda: 
                    ER_img, ER_msk = ER_img.cuda(), ER_msk.cuda()

                # sal part
                sal_loss1= []
                sal_loss2 = []
                for ix in up_sal:
                    sal_loss1.append(F.binary_cross_entropy_with_logits(ix, sal_label, reduction='sum'))

                for ix in up_sal_f:
                    sal_loss2.append(F.binary_cross_entropy_with_logits(ix, sal_label, reduction='sum'))
                sal_loss = (sum(sal_loss1) + sum(sal_loss2)) / (nAveGrad * self.config.batch_size)
              
                r_sal_loss += sal_loss.data
                loss = sal_loss + edge_loss
                r_sum_loss += loss.data
                loss.backward()
                aveGrad += 1

                if aveGrad % nAveGrad == 0:
       
                    self.optimizer_bone.step()
                    self.optimizer_bone.zero_grad()           
                    aveGrad = 0


                if i % showEvery == 0:

                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Edge : %10.4f  ||  Sal : %10.4f  ||  Sum : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num,  r_edge_loss*(nAveGrad * self.config.batch_size)/showEvery,
                                                                r_sal_loss*(nAveGrad * self.config.batch_size)/showEvery,
                                                                r_sum_loss*(nAveGrad * self.config.batch_size)/showEvery))

                    print('Learning rate: ' + str(self.lr_bone))
                    r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0

              #  if i % 200 == 0:

               #     vutils.save_image(torch.sigmoid(up_sal_f[-1].data), tmp_path+'/iter%d-sal-0.jpg' % i, normalize=True, padding = 0)

                #    vutils.save_image(sal_image.data, tmp_path+'/iter%d-sal-data.jpg' % i, padding = 0)
                 #   vutils.save_image(sal_label.data, tmp_path+'/iter%d-sal-target.jpg' % i, padding = 0)
            
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(), '%s/models/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))
                
            if epoch in lr_decay_epoch:
                self.lr_bone = self.lr_bone * 0.1  
                self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone, weight_decay=p['wd'])


        torch.save(self.net_bone.state_dict(), '%s/models/final_bone.pth' % self.config.save_fold)
        


