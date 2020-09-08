import torch
from torch.optim import Adam
from torch.autograd import Variable
from model import build_model, build_OmniVNet
import numpy as np
import cv2
import os
import time
from apex import amp
opt_level = 'O1'
from thop import profile
import matplotlib.pyplot as plt
from util import normPRED
import flow_vis
from flopth import flopth
from model import convert_state_dict_omni


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.build_model()
        if self.config.benchmark_model == False:
            if self.config.pre_trained != '' and config.mode == 'train':
                print('Loading pretrained model from %s...' % self.config.pre_trained)

                netPretrain_dict = torch.load(self.config.pre_trained)
                netPretrain_dict = convert_state_dict_omni(netPretrain_dict)
                self.net.load_state_dict(netPretrain_dict, strict=False)

            if config.mode == 'test':
                print('Loading testing model from %s...' % self.config.model)

                netTest_dict = torch.load(self.config.model)
                self.net.load_state_dict(netTest_dict)
                self.net.eval()

        else:
              print('Now we are running benchmark models...')
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
        if self.config.benchmark_model == False:
            if self.config.backbone == 'rcrnet':
                self.net = build_OmniVNet()

            self.print_network(self.net, 'DAVPNet')

            if self.config.cuda: self.net = self.net.cuda()
            self.lr = self.config.lr
            self.wd = self.config.wd
            self.loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

            base, head = [], []
            for name, param in self.net.named_parameters():
                if 'refineGUN' in name:
                    print(name)
                    head.append(param)
                else:
                    base.append(param)
            self.optimizer = Adam([{'params': base}, {'params': head}], lr=self.lr, weight_decay=self.wd)
            self.optimizer.param_groups[0]['lr'] = self.lr * 0.01
            self.optimizer.param_groups[1]['lr'] = self.lr

        else:
            if self.config.benchmark_name == 'RCRNet':
                from retrain.RCRNet.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() +
                                                    '/retrain/RCRNet/fine_tune_init/epoch_10_bone.pth'))
                self.print_network(self.net, 'RCRNet')
            elif self.config.benchmark_name == 'COSNet':
                from retrain.COSNet.retrain import model, convert_state_dict
                self.net = model
             #   COSNet_pretrain = torch.load(os.getcwd() + '/retrain/COSNet/fine_tune_init/co_attention.pth')["model"]
              #  self.net.load_state_dict(convert_state_dict(COSNet_pretrain))
                COSNet_pretrain = torch.load(os.getcwd() + '/retrain/COSNet/fine_tune_init/epoch_8_bone.pth')
                self.net.load_state_dict(COSNet_pretrain)
                self.print_network(self.net, 'COSNet')
            elif self.config.benchmark_name == 'EGNet':
                from retrain.EGNet.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/EGNet/fine_tune_init/epoch_resnet.pth'))
                self.print_network(self.net, 'EGNet')
            elif self.config.benchmark_name == 'BASNet':
                from retrain.BASNet.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/BASNet/fine_tune_init/final_bone.pth'))
                self.print_network(self.net, 'BASNet')
            elif self.config.benchmark_name == 'CPD':
                from retrain.CPD.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/CPD/fine_tune_init/epoch_10_bone.pth'))
                self.print_network(self.net, 'CPD')
            elif self.config.benchmark_name == 'F3Net':
                from retrain.F3Net.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/F3Net/fine_tune_init/epoch_10_bone.pth'))
                self.print_network(self.net, 'F3Net')
            elif self.config.benchmark_name == 'PoolNet':
                from retrain.PoolNet.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/PoolNet/fine_tune_init/final_bone.pth'))
                self.print_network(self.net, 'PoolNet')
            elif self.config.benchmark_name == 'ScribbleSOD':
                from retrain.ScribbleSOD.retrain import model
                self.net = model
                self.net.load_state_dict(
                    torch.load(os.getcwd() + '/retrain/ScribbleSOD/fine_tune_init/epoch_10_bone.pth'))
                self.print_network(self.net, 'ScribbleSOD')
            elif self.config.benchmark_name == 'SCRN':
                from retrain.SCRN.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/SCRN/fine_tune_init/epoch_10_bone.pth'))
                self.print_network(self.net, 'SCRN')
            elif self.config.benchmark_name == 'GCPANet':
                from retrain.GCPANet.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/GCPANet/fine_tune_init/epoch_10_bone.pth'))
                self.print_network(self.net, 'GCPANet')
            elif self.config.benchmark_name == 'MINet':
                from retrain.MINet.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/MINet/fine_tune_init/epoch_10_bone.pth'))
                self.print_network(self.net, 'MINet')
            elif self.config.benchmark_name == 'Raft':
                from retrain.Raft.retrain import model, convert_state_dict
                self.net = model
                Raft_pretrain = torch.load(os.getcwd() + '/retrain/Raft/models/raft-sintel.pth')
                self.net.load_state_dict(convert_state_dict(Raft_pretrain))
                self.print_network(self.net, 'Raft')
            elif self.config.benchmark_name == 'CSNet':
                from retrain.CSNet.retrain import model
                self.net = model
             #   self.net.load_state_dict(torch.load(os.getcwd() +
              #                                      '/retrain/CSNet/checkpoints/csnet-L-x2/csnet-L-x2.pth.tar')
               #                          ['state_dict'])
                self.net.load_state_dict(torch.load(os.getcwd() +
                                                    '/retrain/CSNet/checkpoints/csnet-L-x2/epoch_10_bone.pth'))
                self.print_network(self.net, 'CSNet')
            elif self.config.benchmark_name == 'CSFRes2Net':
                from retrain.CSFRes2Net.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() +
                                       '/retrain/CSFRes2Net/fine_tune_init/final_bone.pth'), strict=False)
                self.print_network(self.net, 'CSFRes2Net')
            elif self.config.benchmark_name == 'RAS':
                from retrain.RAS.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/RAS/fine_tune_init/epoch_10_bone.pth'))
                self.print_network(self.net, 'RAS')
            elif self.config.benchmark_name == 'AADFNet':
                from retrain.AADFNet.retrain import model, convert_state_dict
                self.net = model
               # AADFNet_pretrain = convert_state_dict(torch.load(os.getcwd() +
                #                                                 '/retrain/AADFNet/fine_tune_init/epoch_1_bone.pth'))
                AADFNet_pretrain = torch.load(os.getcwd() + '/retrain/AADFNet/fine_tune_init/epoch_10_bone.pth')
                self.net.load_state_dict(AADFNet_pretrain)
                self.print_network(self.net, 'AADFNet')
            elif self.config.benchmark_name == 'MGA':
                from retrain.MGA.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/MGA/models/epoch_5_bone.pth'))
                self.print_network(self.net, 'MGA')
            elif self.config.benchmark_name == 'GICD':
                from retrain.GICD.retrain import model
                self.net = model
                self.net.set_mode('test')
                self.net.ginet.load_state_dict(torch.load(os.getcwd() + '/retrain/GICD/fine_tune_init/gicd_ginet.pth'))
                self.print_network(self.net, 'GICD')
            elif self.config.benchmark_name == 'LDF':
                from retrain.LDF.retrain import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/retrain/LDF/fine_tune_init/epoch_10_bone.pth'))
                self.print_network(self.net, 'LDF')

            if self.config.cuda: self.net = self.net.cuda()

        # Apex acceleration
        # self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level=opt_level)

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

                elif self.config.model_type == 'EC':
                    ER_img, ER_msk, CM_imgs, CM_msks = data_batch['ER_img'], data_batch['ER_msk'], \
                                                       data_batch['CM_imgs'], data_batch['CM_msks']
                    if ER_img.size()[2:] != ER_msk.size()[2:]:
                        print("Skip this batch")
                        continue
                    ER_img, CM_img_f, CM_img_r, CM_img_b, \
                    CM_img_l, CM_img_u, CM_img_d = Variable(ER_img), Variable(CM_imgs[0]), Variable(CM_imgs[1]), \
                                                   Variable(CM_imgs[2]), Variable(CM_imgs[3]), Variable(CM_imgs[4]), \
                                                   Variable(CM_imgs[5])
                    ER_msk, CM_msk_f, CM_msk_r, CM_msk_b, CM_msk_l, CM_msk_u, CM_msk_d = Variable(ER_msk), \
                    Variable(CM_msks[0]), Variable(CM_msks[1]), Variable(CM_msks[2]), Variable(CM_msks[3]), \
                                                                    Variable(CM_msks[4]), Variable(CM_msks[5])
                    if self.config.cuda:
                        ER_img, CM_img_f, CM_img_r, CM_img_b, CM_img_l, CM_img_u, CM_img_d , \
                        ER_msk, CM_msk_f, CM_msk_r, CM_msk_b, CM_msk_l, CM_msk_u, CM_msk_d = \
                        ER_img.cuda(), CM_img_f.cuda(), CM_img_r.cuda(), CM_img_b.cuda(), CM_img_l.cuda(), \
                        CM_img_u.cuda(), CM_img_d.cuda(), \
                        ER_msk.cuda(), CM_msk_f.cuda(), CM_msk_r.cuda(), CM_msk_b.cuda(), CM_msk_l.cuda(), \
                        CM_msk_u.cuda(), CM_msk_d.cuda()

                salER = self.net(ER_img, CM_img_f, CM_img_r, CM_img_b, CM_img_l,
                                                                     CM_img_u, CM_img_d)
                loss = self.loss(salER, ER_msk)
                #loss = self.loss(salER, ER_msk) + 1 / 6 * (self.loss(salF, CM_msk_f) + self.loss(salR, CM_msk_r) +
                 #                                          self.loss(salB, CM_msk_b) + self.loss(salL, CM_msk_l) +
                  #                                         self.loss(salU, CM_msk_u) + self.loss(salD, CM_msk_d))

                loss_currIter = loss / (self.config.nAveGrad * self.config.batch_size)
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
                ER_img = ER_img.cuda()
                img_test = ER_img
                if self.config.benchmark_model == True and self.config.benchmark_name == 'RCRNet':
                    img_test = img_test.unsqueeze(0)
                if self.config.benchmark_model == True and self.config.benchmark_name == 'GICD':
                    img_test = img_test.unsqueeze(0)
                if self.config.benchmark_model == True and self.config.needRef == True:
                    Ref = data_batch['Ref_img']
                    Ref = torch.stack(Ref)
                    Ref = Variable(Ref).cuda()
                if self.config.benchmark_model == True and self.config.needPair == True:
                    ER_next = data_batch['ER_img_next']
                    ER_next = Variable(ER_next)
                    ER_next = ER_next.cuda()
                if self.config.benchmark_model == True and self.config.needFlow == True:
                    ER_flow = data_batch['ER_flow']
                    ER_flow = Variable(ER_flow)
                    ER_flow = ER_flow.cuda()

            elif self.config.model_type == 'L':
                TI_imgs, img_name = data_batch['TI_imgs'], data_batch['frm_name']
                TI_imgs = Variable(TI_imgs)
                if self.config.cuda: TI_imgs = TI_imgs.cuda()
                img_test = TI_imgs

            elif self.config.model_type == 'EC':
                ER_img, img_name, CM_imgs = data_batch['ER_img'], data_batch['frm_name'], data_batch['CM_imgs']
                ER_img, CM_f, CM_r, CM_b, CM_l, CM_u, CM_d = Variable(ER_img), Variable(CM_imgs[0]), \
                Variable(CM_imgs[1]), Variable(CM_imgs[2]), Variable(CM_imgs[3]), Variable(CM_imgs[4]),\
                                                             Variable(CM_imgs[5])
                ER_img, CM_f, CM_r, CM_b, CM_l, CM_u, CM_d = ER_img.cuda(), CM_f.cuda(), CM_r.cuda(), CM_b.cuda(), \
                                                             CM_l.cuda(), CM_u.cuda(), CM_d.cuda()
                img_test = ER_img

            with torch.no_grad():
                if self.config.benchmark_model == True and self.config.benchmark_name == 'COSNet':
                    time_start = time.time()
                    sal_sum = 0
                    for idx in range(Ref.size()[0]):
                        sal_sum = sal_sum + self.net(img_test, Ref[idx])[0][0,0,:,:]
                    sal = sal_sum / Ref.size()[0]
                elif self.config.benchmark_model == True and self.config.benchmark_name == 'PoolNet':
                    time_start = time.time()
                    sal = self.net(img_test, 1)
                elif self.config.benchmark_model == True and self.config.benchmark_name == 'Raft':
                    time_start = time.time()
                    _, sal = self.net(img_test, ER_next, test_mode=True)
                elif self.config.benchmark_model == True and self.config.benchmark_name == 'MGA':
                    time_start = time.time()
                    sal = self.net(img_test, ER_flow)
                elif self.config.benchmark_model == False:
                    time_start = time.time()
                    sal = self.net(img_test, CM_f, CM_r, CM_b, CM_l, CM_u, CM_d)
                else:
                    time_start = time.time()
                    sal = self.net(img_test)
                torch.cuda.synchronize()
                time_end = time.time()
                time_total = time_total + time_end - time_start

                flow_output = False
                if self.config.model_type == 'G':

                    # depending on the forward function of each of the SOD methods
                    if self.config.benchmark_model == True and self.config.benchmark_name == 'RCRNet':
                        salT = sal[0]
                        pred = torch.sigmoid(salT)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'EGNet':
                        salT = sal[2][-1]
                        pred = torch.sigmoid(salT)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'COSNet':
                        pred = sal
                        pred = (pred - torch.min(pred) + 1e-8) / (torch.max(pred) - torch.min(pred) + 1e-8)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'BASNet':
                        pred = sal[0]
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'CPD':
                        salT = sal[1]
                        pred = torch.sigmoid(salT)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'F3Net':
                        salT = sal[1]
                        pred = torch.sigmoid(salT)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'PoolNet':
                        pred = torch.sigmoid(sal)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'ScribbleSOD':
                        salT = sal[2]
                        pred = torch.sigmoid(salT)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'SCRN':
                        salT = sal[0]
                        pred = torch.sigmoid(salT)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'GCPANet':
                        salT = sal[0]
                        pred = torch.sigmoid(salT)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'MINet':
                        pred = torch.sigmoid(sal)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'Raft':
                        flow_output = True
                        salT = sal[0]
                        salT = salT.permute(1, 2, 0)
                        pred = flow_vis.flow_to_color(salT.cpu().data.numpy(), convert_to_bgr=True)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'CSNet':
                        pred = torch.sigmoid(sal)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'CSFRes2Net':
                        pred = torch.sigmoid(sal)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'RAS':
                        salT = sal[0]
                        pred = torch.sigmoid(salT)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'AADFNet':
                        pred = sal
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'MGA':
                        salT = sal[0]
                        pred = torch.sigmoid(salT)
                        pred = (pred - torch.min(pred) + 1e-8) / (torch.max(pred) - torch.min(pred) + 1e-8)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'GICD':
                        pred = sal[0][1]
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'LDF':
                        salT = sal[5]
                        pred = torch.sigmoid(salT)

                    if flow_output == False: pred = np.squeeze(pred.cpu().data.numpy())  # to cpu

                elif self.config.model_type == 'L':
                    sal = sal[0, :, 0, :, :]
                    for idx in range(sal.size()[0]):
                        sal[idx,:,:] = torch.sigmoid(sal[idx,:,:])
                    sal = sal.unsqueeze(0)
                    sal_ER = TI2ER(sal, self.config.base_level, self.config.sample_level)
                    pred = np.squeeze(sal_ER[-1].cpu().data.numpy())

                elif self.config.model_type == 'EC':
                    pred = torch.sigmoid(sal)
                    pred = np.squeeze(pred.cpu().data.numpy())

                if flow_output == False: pred = 255 * pred
                cv2.imwrite(self.config.test_fold + img_name[0], pred)

        print("--- %s seconds ---" % (time_total))
        print("--- %s fps ---" % (3778 / time_total))
        f = open(os.getcwd() + '/results/fps.txt', 'w')
        f.write(str(3778 / time_total))
        f.close()
       # GFlops = flopth(self.net)
       # f = open(os.getcwd() + '/results/GFlops.txt', 'w')
        #f.write('GFlops:  ' + GFlops)
        #f.close()
        print('Test Done!')

    # if self.config.benchmark_model == True and self.config.benchmark_name == 'COSNet':
     #               sal_sum = 0
      #              for idx in range(Ref.size()[0]):
       #                 sal_sum = sal_sum + self.net(img_test, Ref[idx])[0][0, 0, :, :]
        #            sal = sal_sum / Ref.size()[0]
         #           sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
          #      else:
           #         sal = self.net(img_test)