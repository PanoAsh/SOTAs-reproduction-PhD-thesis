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
from util import normPRED
import flow_vis
from flopth import flopth


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.build_model()
        if self.config.benchmark_model == False:
            if self.config.pre_trained != '' and config.mode == 'train':
                print('Loading pretrained model from %s...' % self.config.pre_trained)

                netStatic_dict = self.net.state_dict()
                netPretrain_dict = torch.load(self.config.pre_trained)
                netPretrain_dict = {
                    k: v
                    for k, v in netPretrain_dict.items()
                    if k in netStatic_dict and v.shape == netStatic_dict[k].shape
                }  # remove the dynamic parameters declared during TI-based training phase
                self.net.load_state_dict(netPretrain_dict, strict=False)

            if config.mode == 'test':
                print('Loading testing model from %s...' % self.config.model)

                netStatic_dict = self.net.state_dict()
                netTest_dict = torch.load(self.config.model)
                netTest_dict = {
                    k: v
                    for k, v in netTest_dict.items()
                    if k in netStatic_dict and v.shape == netStatic_dict[k].shape
                }  # remove the dynamic parameters declared during TI-based training phase
                self.net.load_state_dict(netTest_dict, strict=False)
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
            if self.config.backbone == 'fcn_resnet101':
                self.net = build_model(self.config.backbone, self.config.fcn, self.config.mode, self.config.model_type,
                                       self.config.base_level)
            elif self.config.backbone == 'deeplabv3_resnet101':
                self.net = build_model(self.config.backbone, self.config.fcn, self.config.mode, self.config.model_type,
                                       self.config.base_level)

            self.print_network(self.net, 'GTNet')

        else:
            if self.config.benchmark_name == 'RCRNet':
                from benchmark.RCRNet.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/RCRNet/models/video_best_model.pth'))
                self.print_network(self.net, 'RCRNet')
            elif self.config.benchmark_name == 'COSNet':
                from benchmark.COSNet.benchmark import model, convert_state_dict
                self.net = model
                COSNet_pretrain = torch.load(os.getcwd() + '/benchmark/COSNet/models/co_attention.pth')["model"]
                self.net.load_state_dict(convert_state_dict(COSNet_pretrain))
                self.print_network(self.net, 'COSNet')
            elif self.config.benchmark_name == 'EGNet':
                from benchmark.EGNet.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/EGNet/pretrained/epoch_resnet.pth'))
                self.print_network(self.net, 'EGNet')
            elif self.config.benchmark_name == 'BASNet':
                from benchmark.BASNet.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/BASNet/models/final_bone.pth'))
                self.print_network(self.net, 'BASNet')
            elif self.config.benchmark_name == 'CPD':
                from benchmark.CPD.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/CPD/models/CPD-R.pth'))
                self.print_network(self.net, 'CPD')
            elif self.config.benchmark_name == 'F3Net':
                from benchmark.F3Net.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/F3Net/models/epoch_9_bone.pth'))
                self.print_network(self.net, 'F3Net')
            elif self.config.benchmark_name == 'PoolNet':
                from benchmark.PoolNet.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/PoolNet/models/final.pth'))
                self.print_network(self.net, 'PoolNet')
            elif self.config.benchmark_name == 'ScribbleSOD':
                from benchmark.ScribbleSOD.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/ScribbleSOD/models/scribble_30.pth'))
                self.print_network(self.net, 'ScribbleSOD')
            elif self.config.benchmark_name == 'SCRN':
                from benchmark.SCRN.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/SCRN/models/model.pth'))
                self.print_network(self.net, 'SCRN')
            elif self.config.benchmark_name == 'GCPANet':
                from benchmark.GCPANet.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/GCPANet/models/model.pt'))
                self.print_network(self.net, 'GCPANet')
            elif self.config.benchmark_name == 'MINet':
                from benchmark.MINet.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/MINet/models/MINet_Res50.pth'))
                self.print_network(self.net, 'MINet')
            elif self.config.benchmark_name == 'Raft':
                from benchmark.Raft.benchmark import model, convert_state_dict
                self.net = model
                Raft_pretrain = torch.load(os.getcwd() + '/benchmark/Raft/models/raft-kitti.pth')
                self.net.load_state_dict(convert_state_dict(Raft_pretrain))
                self.print_network(self.net, 'Raft')
            elif self.config.benchmark_name == 'CSNet':
                from benchmark.CSNet.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() +
                                                    '/benchmark/CSNet/checkpoints/csnet-L-x2/csnet-L-x2.pth.tar')
                                         ['state_dict'])
                self.print_network(self.net, 'CSNet')
            elif self.config.benchmark_name == 'CSFRes2Net':
                from benchmark.CSFRes2Net.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() +
                                        '/benchmark/CSFRes2Net/models/csf_res2net50_final.pth'), strict=False)
                self.print_network(self.net, 'CSFRes2Net')
            elif self.config.benchmark_name == 'RAS':
                from benchmark.RAS.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/RAS/models/RAS.v2.pth'))
                self.print_network(self.net, 'RAS')
            elif self.config.benchmark_name == 'AADFNet':
                from benchmark.AADFNet.benchmark import model, convert_state_dict
                self.net = model
                AADFNet_pretrain = convert_state_dict(torch.load(os.getcwd() + '/benchmark/AADFNet/models/30000.pth'))
                self.net.load_state_dict(AADFNet_pretrain)
                self.print_network(self.net, 'AADFNet')
            elif self.config.benchmark_name == 'MGA':
                from benchmark.MGA.benchmark import model
                self.net = model
                self.net.load_state_dict(torch.load(os.getcwd() + '/benchmark/MGA/models/MGA_trained.pth'))
                self.print_network(self.net, 'MGA')

        if self.config.cuda:
            self.net = self.net.cuda()
        self.lr = self.config.lr
        self.wd = self.config.wd
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                              weight_decay=self.wd)

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

                else:
                    ER_img, ER_msk, TI_imgs, TI_msks = data_batch['ER_img'], data_batch['ER_msk'], \
                                                       data_batch['TI_imgs'], data_batch['TI_msks']
                    print('under built...')

                # FCN-backbone part
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
                ER_img = ER_img.cuda()
                img_test = ER_img
                if self.config.benchmark_model == True and self.config.benchmark_name == 'RCRNet':
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

            else:
                print('under built...')

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
                else:
                    time_start = time.time()
                    sal = self.net(img_test)
                torch.cuda.synchronize()
                time_end = time.time()
                time_total = time_total + time_end - time_start

                flow_output = False
                if self.config.model_type == 'G':
                 #   pred = np.squeeze(torch.sigmoid(sal[-1]).cpu().data.numpy())

                    # depending on the forward function of each of the SOD methods
                    if self.config.benchmark_model == True and self.config.benchmark_name == 'RCRNet':
                        salT = sal[0]
                        pred = torch.sigmoid(salT)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'EGNet':
                        salT = sal[2][-1]
                        pred = torch.sigmoid(salT)
                    elif self.config.benchmark_model == True and self.config.benchmark_name == 'COSNet':
                        pred = sal
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

                    if flow_output == False: pred = np.squeeze(pred.cpu().data.numpy())  # to cpu

                elif self.config.model_type == 'L':
                    sal = sal[0, :, 0, :, :]
                    for idx in range(sal.size()[0]):
                        sal[idx,:,:] = torch.sigmoid(sal[idx,:,:])
                    sal = sal.unsqueeze(0)
                    sal_ER = TI2ER(sal, self.config.base_level, self.config.sample_level)
                    pred = np.squeeze(sal_ER[-1].cpu().data.numpy())

                if flow_output == False: pred = 255 * pred
                cv2.imwrite(self.config.test_fold + img_name[0], pred)

        print("--- %s seconds ---" % (time_total))
        print("--- %s fps ---" % (3778 / time_total))
        f = open(os.getcwd() + '/results/fps.txt', 'w')
        f.write(str(3778 / time_total))
        f.close()
        GFlops = flopth(self.net)
        f = open(os.getcwd() + '/results/GFlops.txt', 'w')
        f.write('GFlops:  ' + GFlops)
        f.close()
        print('Test Done!')

    # if self.config.benchmark_model == True and self.config.benchmark_name == 'COSNet':
     #               sal_sum = 0
      #              for idx in range(Ref.size()[0]):
       #                 sal_sum = sal_sum + self.net(img_test, Ref[idx])[0][0, 0, :, :]
        #            sal = sal_sum / Ref.size()[0]
         #           sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
          #      else:
           #         sal = self.net(img_test)