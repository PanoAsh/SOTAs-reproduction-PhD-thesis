import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np
import os, argparse
from datetime import datetime

from utils.data import get_loader
from utils.func import label_edge_prediction, AvgMeter

from model.ResNet_models import SCRN

pretrained_model = os.getcwd() + '/pretrained_model/model.pth'
save_path = os.getcwd() + '/results/models/'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate') # 1/10 ori_lr_init
parser.add_argument('--batchsize', type=int, default=1, help='batch size')
parser.add_argument('--trainsize', type=int, default=256, help='input size')
#parser.add_argument('--trainset', type=str, default='360ISOD', help='training  dataset')
opt = parser.parse_args()

# data preparing, set your own data path here
#data_path = os.getcwd() + '/data/360ISOD/'
#image_root = data_path + opt.trainset + '/images/'
#gt_root = data_path + opt.trainset + '/gts/'
image_root = os.getcwd() + '/data/360ISOD/360ISOD-Image/'
gt_root = os.getcwd() + '/data/360ISOD/360ISOD-Mask/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

# build models
model = SCRN()
model.load_state_dict(torch.load(pretrained_model))  # for fine-tuning
model.cuda()
params = model.parameters()
optimizer = torch.optim.SGD(params, opt.lr, momentum=0.9, weight_decay=5e-4)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
CE = torch.nn.BCEWithLogitsLoss()
size_rates = [0.75, 1, 1.25]  # multi-scale training

# training
for epoch in range(0, opt.epoch):
    #scheduler.step()
    model.train()
    loss_record1, loss_record2 = AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # edge prediction
            gt_edges = label_edge_prediction(gts)

            # multi-scale training samples
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gt_edges = F.upsample(gt_edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # forward
            pred_sal, pred_edge = model(images)

            loss1 = CE(pred_sal, gts)
            loss2 = CE(pred_edge, gt_edges)
            loss = loss1 + loss2 
            loss.backward()

            optimizer.step()
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)

        if i % 100 == 0:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}, Loss2: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(), loss_record2.show()))


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:  # save model every 5 epoches
        torch.save(model.state_dict(), save_path + '/' + str(epoch+1) + '.pth')

print('training done !')

