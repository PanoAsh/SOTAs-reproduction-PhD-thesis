import torch
import torch.nn.functional as F
from torch.autograd import Variable
import cv2

import numpy as np
import os
from scipy import misc
from datetime import datetime

from utils.data import test_dataset
from model.ResNet_models import SCRN

model_test = os.getcwd() + '/results/models/50.pth'

model = SCRN()
model.load_state_dict(torch.load(model_test))
model.cuda()
model.eval()

#data_path = os.getcwd() + '/data/360ISOD-TE/Imgs_test/'
data_path = os.getcwd() + '/data/360ISOD-TE/Imgs_train/'
#gt_path = os.getcwd() + '/data/360ISOD-TE/gt_test/'
gt_path = os.getcwd() + '/data/360ISOD-TE/gt_train/'
save_path = os.getcwd() + '/results/predicted/'
test_loader = test_dataset(data_path, gt_path, testsize=256)

with torch.no_grad():
    count = 1
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        image = Variable(image).cuda()
            
        res, edge = model(image)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = res * 255

        cv2.imwrite(os.path.join(save_path, name + '.png'), res)
        print(" {} images processed".format(count))
        count += 1

print('test done !')