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

#save_path_depth = os.getcwd() + '/data/depth_test/'
save_path_defocus = os.getcwd() + '/data/defocus_train/'
save_path_FS = os.getcwd() + '/data/FS_order_train/'

model.eval()
with torch.no_grad():
    for i in range(test_loader.size):
        images, gt, name, fs_name = test_loader.load_data()
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
        #best_idx = mae_list.index(min(mae_list))
        mae_index = sorted(range(len(mae_list)), key=lambda k: mae_list[k])  # small to large

        for idx in range(len(images)):
            res = defocus_list[mae_index[idx]]
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            print('save img to: ', save_path_defocus + fs_name[mae_index[idx]][:-4] + '.png')
            cv2.imwrite(save_path_defocus + fs_name[mae_index[idx]][:-4] + '.png', res * 255)

            fs_pth = os.path.join(image_root, fs_name[mae_index[idx]])
            new_name = name[:-4] + '_' + format(str(idx), '0>2s') + '.jpg'
            new_pth = os.path.join(save_path_FS, new_name)
            print('save img to: ', new_pth)
            os.rename(fs_pth, new_pth)
            print()

    print('Test Done!')



