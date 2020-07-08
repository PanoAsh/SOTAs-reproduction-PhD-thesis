import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random
import matplotlib.pyplot as plt


class ImageDataTrain(data.Dataset):
    def __init__(self):

        self.img_source = os.getcwd() + '/data/train_img.lst'
        self.msk_source = os.getcwd() + '/data/train_msk.lst'

        with open(self.img_source, 'r') as f:
            self.img_list = [x.strip() for x in f.readlines()]
        with open(self.msk_source, 'r') as f:
            self.msk_list = [x.strip() for x in f.readlines()]

        self.img_num = len(self.img_list)


    def __getitem__(self, item):

        ER_img = load_2dImg(self.img_list[item%self.img_num])
        ER_msk = load_2dMsk(self.msk_list[item%self.img_num])
        sample = {'ER_img': ER_img, 'ER_msk': ER_msk}

        return sample

    def __len__(self):
        return self.img_num

class ImageDataTest(data.Dataset):
    def __init__(self, test_mode='vr', sal_mode=''):
        if test_mode == 0:
            # self.image_root = '/home/liuj/dataset/saliency_test/ECSSD/Imgs/'
            # self.image_source = '/home/liuj/dataset/saliency_test/ECSSD/test.lst'
            self.image_root = '/home/liuj/dataset/HED-BSDS_PASCAL/HED-BSDS/test/'
            self.image_source = '/home/liuj/dataset/HED-BSDS_PASCAL/HED-BSDS/test.lst'


        elif test_mode == 1:
            if sal_mode == 'e':
                self.image_root = '/home/liuj/dataset/saliency_test/ECSSD/Imgs/'
                self.image_source = '/home/liuj/dataset/saliency_test/ECSSD/test.lst'
                self.test_fold = '/media/ubuntu/disk/Result/saliency/ECSSD/'
            elif sal_mode == 'p':
                self.image_root = '/home/liuj/dataset/saliency_test/PASCALS/Imgs/'
                self.image_source = '/home/liuj/dataset/saliency_test/PASCALS/test.lst'
                self.test_fold = '/media/ubuntu/disk/Result/saliency/PASCALS/'
            elif sal_mode == 'd':
                self.image_root = '/home/liuj/dataset/saliency_test/DUTOMRON/Imgs/'
                self.image_source = '/home/liuj/dataset/saliency_test/DUTOMRON/test.lst'
                self.test_fold = '/media/ubuntu/disk/Result/saliency/DUTOMRON/'
            elif sal_mode == 'h':
                self.image_root = '/home/liuj/dataset/saliency_test/HKU-IS/Imgs/'
                self.image_source = '/home/liuj/dataset/saliency_test/HKU-IS/test.lst'
                self.test_fold = '/media/ubuntu/disk/Result/saliency/HKU-IS/'
            elif sal_mode == 's':
                self.image_root = '/home/liuj/dataset/saliency_test/SOD/Imgs/'
                self.image_source = '/home/liuj/dataset/saliency_test/SOD/test.lst'
                self.test_fold = '/media/ubuntu/disk/Result/saliency/SOD/'
            elif sal_mode == 'm':
                self.image_root = '/home/liuj/dataset/saliency_test/MSRA/Imgs/'
                self.image_source = '/home/liuj/dataset/saliency_test/MSRA/test.lst'
            elif sal_mode == 'o':
                self.image_root = '/home/liuj/dataset/saliency_test/SOC/TestSet/Imgs/'
                self.image_source = '/home/liuj/dataset/saliency_test/SOC/TestSet/test.lst'
                self.test_fold = '/media/ubuntu/disk/Result/saliency/SOC/'
            elif sal_mode == 't':
                self.image_root = '/home/liuj/dataset/DUTS/DUTS-TE/DUTS-TE-Image/'
                self.image_source = '/home/liuj/dataset/DUTS/DUTS-TE/test.lst'
                self.test_fold = '/media/ubuntu/disk/Result/saliency/DUTS/'
        elif test_mode == 2:

            self.image_root = '/home/liuj/dataset/SK-LARGE/images/test/'
            self.image_source = '/home/liuj/dataset/SK-LARGE/test.lst'

        elif test_mode == 3:
            self.image_root = os.getcwd() + '/data/360ISOD-TE/Imgs/'
            self.image_source = os.getcwd() + '/data/360ISOD-TE/test.lst'
            self.test_fold = os.getcwd() + '/results/predicted/'

        with open(self.image_source, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.image_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item%self.image_num], 'size': im_size}
    def save_folder(self):
        return self.test_fold

    def __len__(self):
        # return max(max(self.edge_num, self.skel_num), self.sal_num)
        return self.image_num


# get the dataloader (Note: without data augmentation, except saliency with random flip)
def get_loader(batch_size, mode='train', num_thread=1):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain()
    else:
        dataset = ImageDataTest()

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return data_loader, dataset

def load_2dImg(pth):
    if not os.path.exists(pth):
        print('File Not Exists')
    img = Image.open(pth)
    preprocess = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)

    return img_tensor

def load_2dMsk(pth):
    if not os.path.exists(pth):
        print('File Not Exists')
    msk = Image.open(pth)
    preprocess = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
    ])
    msk_tensor = preprocess(msk)

    return msk_tensor