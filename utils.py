#------------------------ load the necessary packages ------------------------
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from config import *

#------------------------ functions for TXT genneration ------------------------
def TXT_gnr(TXT_path, IMG_dir):
    f = open(TXT_path, 'w')

    for root, dirs, files in os.walk(IMG_dir, topdown=False):
        for sub_file in files:
            abs_file = os.path.join(root, sub_file)
            line = abs_file + '\n'
            f.write(line)
    f.close()

#------------------------ DIY datset class ------------------------
class MyDataset(Dataset):
    def __init__(self, ids_imgs_path, ids_objms_path,
                 normTrans=None):
        ids_imgs = open(ids_imgs_path, 'r')
        ids_objms = open(ids_objms_path, 'r')
        imgs = []
        objms = []
        for line in ids_imgs:
            line = line.rstrip()
            imgs.append(line)
        for line in ids_objms:
            line = line.rstrip()
            objms.append(line)

        self.imgs = imgs
        self.objms = objms
        self.norm = normTrans

    def __getitem__(self, index):
        img_path = self.imgs[index]
        objm_path = self.objms[index]
        img = Image.open(img_path).convert('RGB')
        objm = Image.open(objm_path).convert('L')
        objm_db = Image.open(objm_path).convert('RGB')


        imgTrans, objmTrans = data_ForTrain(self.norm)
        imgs, objms, objms_db = data_MultiCrop(img, objm, objm_db,
                                           img.size[0], img.size[1])
        for i in range(10):
            imgs[i] = imgTrans(imgs[i])
            objms[i] = objmTrans(objms[i])

        return imgs[9], objms[9]

    def __len__(self):
        return len(self.imgs)

# ------------------------ preprocess the dataset ------------------------
def data_MultiCrop(img, objm, objm_db, Height, Width, Scale=1):
     """

        Based on the prior knowledge, 9 viewports(vr) will be located
        on the current image.

        The vrs are the cropping centers.

        The param:Scale decides the number of cropping scales applied
        on each of the vr.

     """
     # ************ 9 center crop on 1st scale ************
     h_s1 = Height/3*2
     w_s1 = Width/3*2
     imgs = []
     objms = []
     objms_db = []

     for i in range(3):
         for j in range(3):
             x1 = i*Height/6
             y1 = j*Width/6
             x2 = x1+h_s1
             y2 = y1+w_s1
             imgs.append(img.crop((x1,y1,x2,y2)))
             objms.append(objm.crop((x1,y1,x2,y2)))
             objms_db.append(objm_db.crop((x1,y1,x2,y2)))

     imgs.append(img)
     objms.append(objm)
     objms_db.append(objm_db)

     return imgs, objms, objms_db

def data_ForTrain(normTransform):
    imgsTransform = transforms.Compose([
        transforms.Resize((size_train,size_train*2)),
        transforms.ToTensor(),
        normTransform
    ])
    objmsTransform = transforms.Compose([
        transforms.Resize((size_train,size_train*2)),
        transforms.ToTensor()
    ])

    return imgsTransform, objmsTransform

# ------------------------ calculate the mean and std ------------------------
def data_norm(Num):
    imgs = np.zeros([size_train, size_train, 3, 1])
    means, stdevs = [], []

    with open(ids_imgs_train_path, 'r') as f:
        lines = f.readlines()
        print('************************')
        print('start calculating the mean and std of the training dataset...')
        for i in range(Num):
            img_path = lines[i].rstrip().split()[0]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (size_train, size_train))
            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)
            print("now calculate the image:{}".format(i+1))
    imgs = imgs.astype(np.float32) / 255.
    print('calculation finished!')

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    means.reverse()  # BGR(opencv default) --> RGB
    stdevs.reverse()

    print('************************')
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('************************')

    data_norm = transforms.Normalize(means, stdevs)

    return data_norm

# ------------------------ calculate the mean and std ------------------------
def debug_vision(img, objm):
    print('start debugging...')
    img = img.transpose(0,2)
    img = img.transpose(0,1)
    objm = objm.transpose(0,2)
    objm = objm.transpose(0,1)
    img = img.cpu().numpy()  #
    objm = objm.cpu().numpy()
    plt.figure
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(objm)
    plt.show()

# ------------------------ calculate the mean and std ------------------------
class DiceCoef(nn.Module):
    def __init__(self):
        super(DiceCoef, self).__init__()

    def forward(self, input, target):
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        DC = []

        for i in range(input.shape[0]):
            DC.append((2 * torch.sum(input[i] * target[i]) + epsilon_DC) / \
                      (torch.sum(input[i]) + torch.sum(target[i]) + epsilon_DC))
            DC[i].unsqueeze_(0)

        DC = torch.cat(DC, 0)
        DC = torch.mean(DC)

        return DC