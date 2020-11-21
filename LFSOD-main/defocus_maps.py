import os
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from data import test_dataset
import torch.backends.cudnn as cudnn
import models
import cv2

model = models.__dict__['resskspp']()
model.cuda()
model.load_state_dict(torch.load(os.getcwd() + '/pretrained/res_best.pth'))

image_root = os.getcwd() + '/data/FS_rgb_train/'
gt_root = os.getcwd() + '/data/GT_train/'
test_loader = test_dataset(image_root, gt_root)

save_path_depth = os.getcwd() + '/data/depth_train/'
save_path_defocus = os.getcwd() + '/data/defocus_train/'

model.eval()
with torch.no_grad():
    for i in range(test_loader.size):
        images, gt, name = test_loader.load_data()
        defocus_list, depth_list = [], []
        for idx in range(len(images)):
            images[idx] = images[idx].cuda()
            defocus, depth = model(images[idx])
            defocus_list.append(defocus[0])
            depth_list.append(depth[0])
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        mae_list = []
        for idx in range(len(images)):
            res = F.upsample(defocus_list[idx], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_list.append(np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1]))
        best_idx = mae_list.index(min(mae_list))

        res = defocus_list[best_idx]
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ', save_path_defocus + name)
        cv2.imwrite(save_path_defocus + name, res * 255)

        res = depth_list[best_idx]
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ', save_path_depth + name)
        cv2.imwrite(save_path_depth + name, res * 255)
    print('Test Done!')



