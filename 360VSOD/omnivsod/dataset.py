import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from util import ER2TI
import torch.nn.functional as F


class ImageDataTrain(data.Dataset):
    def __init__(self, data_type, base_level, sample_level):
        self.img_source = os.getcwd() + '/data/train_img.lst'
        self.msk_source = os.getcwd() + '/data/train_msk.lst'
        self.data_type = data_type
        self.base_level = base_level
        self.sample_level = sample_level

        with open(self.img_source, 'r') as f:
            self.img_list = [x.strip() for x in f.readlines()]
        with open(self.msk_source, 'r') as f:
            self.msk_list = [x.strip() for x in f.readlines()]

        self.img_num = len(self.img_list)

    def __getitem__(self, item):
        if self.data_type == 'G':
            ER_img = load_ERImg(self.img_list[item % self.img_num])
            ER_msk = load_ERMsk(self.msk_list[item % self.img_num])
            sample = {'ER_img': ER_img, 'ER_msk': ER_msk}

        elif self.data_type == 'L':
            TI_imgs = load_TIImg(self.img_list[item % self.img_num], self.base_level, self.sample_level)
            ER_msk = load_ERMsk(self.msk_list[item % self.img_num])
           # TI_msks = load_TIMsk(self.msk_list[item % self.img_num], self.base_level, self.sample_level)
            sample = {'TI_imgs': TI_imgs, 'ER_msk': ER_msk}

        else:
            ER_img = load_ERImg(self.img_list[item % self.img_num])
            ER_msk = load_ERMsk(self.msk_list[item % self.img_num])
            TI_imgs = load_TIImg(self.img_list[item % self.img_num], self.base_level, self.sample_level)
            TI_msks = load_TIMsk(self.msk_list[item % self.img_num], self.base_level, self.sample_level)
            sample = {'ER_img': ER_img, 'ER_msk': ER_msk, 'TI_imgs': TI_imgs, 'TI_msks': TI_msks}

        return sample

    def __len__(self):
        return self.img_num

class ImageDataTest(data.Dataset):
    def __init__(self, data_type, base_level, sample_level):
        self.img_source = os.getcwd() + '/data/test_img.lst'
        self.data_type = data_type
        self.base_level = base_level
        self.sample_level = sample_level

        with open(self.img_source, 'r') as f:
            self.img_list = [x.strip() for x in f.readlines()]

        self.img_num = len(self.img_list)

    def __getitem__(self, item):
        frm_name = self.img_list[item % self.img_num][52:]
        name_list = frm_name.split('/')
        frm_name = name_list[0] + '-' + name_list[1] + '-' + name_list[2]

        if self.data_type == 'G':
            ER_img = load_ERImg(self.img_list[item % self.img_num])
            sample = {'ER_img': ER_img, 'frm_name': frm_name}

        elif self.data_type == 'L':
            TI_imgs = load_TIImg(self.img_list[item % self.img_num], self.base_level, self.sample_level)
            sample = {'TI_imgs': TI_imgs, 'frm_name': frm_name}

        else:
            print('under built...')

        return sample

    def __len__(self):
        return self.img_num

# get the dataloader (Note: without data augmentation, except saliency with random flip)
def get_loader(batch_size, mode='train', num_thread=1, data_type='G', base_level = 1, sample_level=10):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(data_type=data_type, base_level=base_level, sample_level=sample_level)
    else:
        dataset = ImageDataTest(data_type=data_type, base_level=base_level, sample_level=sample_level)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)

    return data_loader, dataset

def load_ERImg(pth):
    if not os.path.exists(pth):
        print('File Not Exists')
    img = Image.open(pth)
    preprocess = transforms.Compose([
        transforms.Resize([256, 512]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)

    return img_tensor

def load_ERMsk(pth):
    if not os.path.exists(pth):
        print('File Not Exists')
    msk = Image.open(pth)
    preprocess = transforms.Compose([
        transforms.Resize([256, 512]),
        transforms.ToTensor(),
    ])
    msk_tensor = preprocess(msk)

    return msk_tensor

def load_TIImg(pth, base_level, sample_level):
    if not os.path.exists(pth):
        print('File Not Exists')
    ER_img = Image.open(pth)

    preprocess = transforms.Compose([
        transforms.Resize([int(2048 / 2 ** (10 - sample_level)), int(4096 / 2 ** (10 - sample_level))]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ER_img_tensor = preprocess(ER_img)
    TI_imgs = ER2TI(ER_img_tensor, base_level, sample_level)

    return TI_imgs

def load_TIMsk(pth, base_level, sample_level):
    if not os.path.exists(pth):
        print('File Not Exists')
    ER_msk = Image.open(pth)
    preprocess = transforms.Compose([
        transforms.Resize([int(2048 / 2 ** (10 - sample_level)), int(4096 / 2 ** (10 - sample_level))]),
        transforms.ToTensor(),
    ])
    ER_msk_tensor = preprocess(ER_msk)
    TI_msks = ER2TI(ER_msk_tensor, base_level, sample_level)

    return TI_msks