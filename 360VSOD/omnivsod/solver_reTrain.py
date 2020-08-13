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

# ---------------------------- utils ---------------------------------- #
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

fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
fx = np.reshape(fx, (1, 1, 3, 3))
fy = np.reshape(fy, (1, 1, 3, 3))
fx = Variable(torch.from_numpy(fx)).cuda()
fy = Variable(torch.from_numpy(fy)).cuda()
contour_th = 1.5

def label_edge_prediction(label):
    # convert label to edge
    label = label.gt(0.5).float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad

def bce_iou_loss(pred, mask):
    bce   = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou   = 1-(inter+1)/(union-inter+1)

    return (bce+iou).mean()

def bce2d(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)
# ----------------------------- utils above ----------------------------- #

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

        elif self.config.benchmark_name == 'SCRN':
            from retrain.SCRN.retrain import model
            self.net = model
            self.print_network(self.net, 'SCRN')
            if self.config.fine_tune == True:
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/SCRN/fine_tune_init/model.pth'))
                print('fine tuning ...')

        elif self.config.benchmark_name == 'GCPANet':
            from retrain.GCPANet.retrain import model
            self.net = model
            self.print_network(self.net, 'GCPANet')
            if self.config.fine_tune == True:
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/GCPANet/fine_tune_init/model-100045448.pt'))
                print('fine tuning ...')

        elif self.config.benchmark_name == 'RAS':
            from retrain.RAS.retrain import model
            self.net = model
            self.print_network(self.net, 'RAS')
            if self.config.fine_tune == True:
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/RAS/fine_tune_init/RAS.v2.pth'))
                print('fine tuning ...')

        elif self.config.benchmark_name == 'CSFRes2Net':
            from retrain.CSFRes2Net.retrain import model
            self.net = model
            self.print_network(self.net, 'CSFRes2Net')
            if self.config.fine_tune == True:
                self.net.load_state_dict(torch.load(os.getcwd() +
                                    '/retrain/CSFRes2Net/fine_tune_init/csf_res2net50_final.pth'), strict=False)
                print('fine tuning ...')

        elif self.config.benchmark_name == 'CSNet':
            from retrain.CSNet.retrain import model
            self.net = model
            self.print_network(self.net, 'CSNet')
            if self.config.fine_tune == True:
                self.net.load_state_dict(torch.load(os.getcwd() +
                                                    '/retrain/CSNet/checkpoints/csnet-L-x2/csnet-L-x2.pth.tar')
                                         ['state_dict'])
                print('fine tuning ...')

        elif self.config.benchmark_name == 'MINet':
            from retrain.MINet.retrain import model
            self.net = model
            self.print_network(self.net, 'MINet')
            if self.config.fine_tune == True:
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/MINet/fine_tune_init/MINet_Res50.pth'))
                print('fine tuning ...')

        elif self.config.benchmark_name == 'AADFNet':
            from retrain.AADFNet.retrain import model, convert_state_dict
            self.net = model
            self.print_network(self.net, 'AADFNet')
            if self.config.fine_tune == True:
                AADFNet_pretrain = convert_state_dict(torch.load(os.getcwd() +
                                                                 '/retrain/AADFNet/fine_tune_init/30000.pth'))
                self.net.load_state_dict(AADFNet_pretrain)
                print('fine tuning ...')

        elif self.config.benchmark_name == 'PoolNet':
            from retrain.PoolNet.retrain import model
            self.net = model
            self.print_network(self.net, 'PoolNet')
            if self.config.fine_tune == True:
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/PoolNet/fine_tune_init/final.pth'))
                print('fine tuning ...')

        if self.config.cuda: self.net = self.net.cuda()
        self.lr = self.config.lr
        self.wd = self.config.wd

        # optimizer
        if self.config.optimizer_name == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                              weight_decay=self.wd)
        elif self.config.optimizer_name == 'SGD':
            if self.config.benchmark_name == 'F3Net' or self.config.benchmark_name == 'GCPANet' \
                    or self.config.benchmark_name == 'RAS':
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
            elif self.config.benchmark_name == 'SCRN' or self.config.benchmark_name == 'MINet'\
                    or self.config.benchmark_name == 'AADFNet':
                params = self.net.parameters()
                self.optimizer = SGD(params, self.lr, momentum=0.9, weight_decay=self.wd)

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
                elif self.config.benchmark_name == 'SCRN':
                    msk_train_edge = label_edge_prediction(msk_train)
                    pred_sal, pred_edge = self.net(img_train)
                    loss1 = CE(pred_sal, msk_train)
                    loss2 = CE(pred_edge, msk_train_edge)
                    loss = loss1 + loss2
                elif self.config.benchmark_name == 'GCPANet':
                    out2, out3, out4, out5 = self.net(img_train)
                    loss2 = F.binary_cross_entropy_with_logits(out2, msk_train)
                    loss3 = F.binary_cross_entropy_with_logits(out3, msk_train)
                    loss4 = F.binary_cross_entropy_with_logits(out4, msk_train)
                    loss5 = F.binary_cross_entropy_with_logits(out5, msk_train)
                    loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4
                elif self.config.benchmark_name == 'RAS':
                    out2, out3, out4, out5 = self.net(img_train)
                    loss2 = bce_iou_loss(out2, msk_train)
                    loss3 = bce_iou_loss(out3, msk_train)
                    loss4 = bce_iou_loss(out4, msk_train)
                    loss5 = bce_iou_loss(out5, msk_train)
                    loss = loss2 + loss3 + loss4 + loss5
                elif self.config.benchmark_name == 'CSFRes2Net' or self.config.benchmark_name == 'CSNet':
                    sal_pred = self.net(img_train)
                    loss = F.binary_cross_entropy_with_logits(sal_pred, msk_train)
                elif self.config.benchmark_name == 'MINet':
                    sal_pred = self.net(img_train)
                    loss = CE(sal_pred, msk_train)
                elif self.config.benchmark_name == 'AADFNet':
                    print('under built ...')
                    break
                elif self.config.benchmark_name == 'PoolNet':
                    sal_pred = self.net(img_train, mode=1)
                    loss = F.binary_cross_entropy_with_logits(sal_pred, msk_train)

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
