import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from util import ER2TI
import torch.nn.functional as F
import cv2
import numpy as np


class ImageDataTrain(data.Dataset):
    def __init__(self, data_type, base_level, sample_level, data_norm, data_pair, data_flow):
        self.img_source = os.getcwd() + '/data/train_img.lst'
        self.msk_source = os.getcwd() + '/data/train_msk.lst'
        self.data_type = data_type
        self.base_level = base_level
        self.sample_level = sample_level
        self.data_norm = data_norm
        self.data_pair = data_pair
        self.data_flow = data_flow

        with open(self.img_source, 'r') as f:
            self.img_list = [x.strip() for x in f.readlines()]
        with open(self.msk_source, 'r') as f:
            self.msk_list = [x.strip() for x in f.readlines()]

        self.img_num = len(self.img_list)

    def __getitem__(self, item):
        if self.data_type == 'G':
            ER_img = load_ERImg(self.img_list[item % self.img_num], self.data_norm)
            ER_msk = load_ERMsk(self.msk_list[item % self.img_num])
            sample = {'ER_img': ER_img, 'ER_msk': ER_msk}

            if self.data_pair == True:
                if item != self.img_num - 1:
                    ER_img_next = load_ERImg(self.img_list[(item+1) % self.img_num], self.data_norm)
                    ER_msk_next = load_ERMsk(self.msk_list[(item+1) % self.img_num])
                else:
                    ER_img_next = ER_img
                    ER_msk_next = ER_msk

                sample = {'ER_img': ER_img, 'ER_msk': ER_msk, 'ER_img_next': ER_img_next, 'ER_msk_next': ER_msk_next}

        elif self.data_type == 'L':
            TI_imgs = load_TIImg(self.img_list[item % self.img_num], self.base_level, self.sample_level)
            #ER_msk = load_ERMsk(self.msk_list[item % self.img_num])
            TI_msks = load_TIMsk(self.msk_list[item % self.img_num], self.base_level, self.sample_level)
            sample = {'TI_imgs': TI_imgs, 'TI_msks': TI_msks}

        else:
            ER_img = load_ERImg(self.img_list[item % self.img_num], self.data_norm)
            ER_msk = load_ERMsk(self.msk_list[item % self.img_num])
            TI_imgs = load_TIImg(self.img_list[item % self.img_num], self.base_level, self.sample_level)
            TI_msks = load_TIMsk(self.msk_list[item % self.img_num], self.base_level, self.sample_level)
            sample = {'ER_img': ER_img, 'ER_msk': ER_msk, 'TI_imgs': TI_imgs, 'TI_msks': TI_msks}

        return sample

    def __len__(self):
        return self.img_num

class ImageDataTest(data.Dataset):
    def __init__(self, data_type, base_level, sample_level, need_ref, data_norm, data_pair, data_flow):
        self.img_source = os.getcwd() + '/data/test_img.lst'
        self.data_type = data_type
        self.base_level = base_level
        self.sample_level = sample_level
        self.need_ref = need_ref
        self.data_norm = data_norm
        self.data_pair = data_pair
        self.data_flow = data_flow
        #self.ins_source = os.getcwd() + '/data/test_ins.lst'
        #self.gt_source = os.getcwd() + '/data/test_msk.lst'

        with open(self.img_source, 'r') as f:
            self.img_list = [x.strip() for x in f.readlines()]
        #with open(self.gt_source, 'r') as f:
         #   self.gt_list = [x.strip() for x in f.readlines()]

        self.img_num = len(self.img_list)

    def __getitem__(self, item):
        frm_name = self.img_list[item % self.img_num][52:]
        name_list = frm_name.split('/')
        frm_name = name_list[0] + '-' + name_list[1] + '-' + name_list[2]

        if self.data_type == 'G':
            ER_img = load_ERImg(self.img_list[item % self.img_num], self.data_norm)

            if self.need_ref == False:
                # ER_ins = load_ERMsk(self.ins_list[item % self.img_num])
                sample = {'ER_img': ER_img, 'frm_name': frm_name}
                #prep_score(self.gt_list[item % self.img_num], frm_name)
                # prep_ins(self.ins_list[item % self.img_num], frm_name)
            else:
                refFrm_pth = []
                Ref_img = []
                [refFrm_pth.append(idx) for idx in self.img_list if idx[:-10] == self.img_list[item][:-10]]
                Ref_img.append(load_ERImg(refFrm_pth[0], self.data_norm)) # only choose the first frame as reference
               # [Ref_img.append(load_ERImg(pth, self.data_norm)) for pth in refFrm_pth]

                sample = {'ER_img': ER_img, 'frm_name': frm_name, 'Ref_img': Ref_img}

            if self.data_pair == True:
                if item != self.img_num - 1:
                    ER_img_next = load_ERImg(self.img_list[(item+1) % self.img_num], self.data_norm)
                else:
                    ER_img_next = ER_img

                sample = {'ER_img': ER_img, 'frm_name': frm_name, 'ER_img_next': ER_img_next}

            if self.data_flow == True:
                flow_pth = os.path.join(os.getcwd(), 'result_analysis', 'Sal_test_raft_kitti', frm_name)
                ER_flow = load_ERImg(flow_pth, self.data_norm)
                sample = {'ER_img': ER_img, 'frm_name': frm_name, 'ER_flow': ER_flow}

        elif self.data_type == 'L':
            TI_imgs = load_TIImg(self.img_list[item % self.img_num], self.base_level, self.sample_level)
            sample = {'TI_imgs': TI_imgs, 'frm_name': frm_name}

        else:
            print('under built...')

        return sample

    def __len__(self):
        return self.img_num

# get the dataloader (Note: without data augmentation, except saliency with random flip)
def get_loader(batch_size, mode='train', num_thread=1, data_type='G', base_level = 1, sample_level=10, ref=False,
               norm='cv2', pair=False, flow=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(data_type=data_type, base_level=base_level, sample_level=sample_level,
                                 data_norm=norm, data_pair=pair, data_flow=flow)
    else:
        dataset = ImageDataTest(data_type=data_type, base_level=base_level, sample_level=sample_level,
                                need_ref=ref, data_norm=norm, data_pair=pair, data_flow=flow)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)

    return data_loader, dataset

def load_ERImg(pth, norm):
    if not os.path.exists(pth):
        print('File Not Exists')
    if norm == 'cv2':
        im = cv2.imread(pth)
        im = cv2.resize(im, (512, 256))
        in_ = np.array(im, dtype=np.float32)
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        in_ = in_.transpose((2, 0, 1))
        in_ = torch.Tensor(in_)
    elif norm == 'PIL':
        im = Image.open(pth)
        preprocess = transforms.Compose([
            transforms.Resize([256, 512]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        in_ = preprocess(im)

    return in_

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

def prep_demo(img_pth, gt_pth, name):
    img = Image.open(img_pth)
    img = img.resize((512, 256))
    gt = Image.open(gt_pth)
    gt = gt.resize((512, 256))
    img.save('/home/yzhang1/PythonProjects/omnivsod/results/Img/' + name)
    gt.save('/home/yzhang1/PythonProjects/omnivsod/results/GT/' + name)

def prep_score(gt_pth, name):
    gt = Image.open(gt_pth)
    gt.save('/home/yzhang1/PythonProjects/omnivsod/results/GT_ori/' + name)

def prep_ins(ins_pth, name):
    ins = Image.open(ins_pth)
    ins = ins.resize((512, 256))
    ins.save('/home/yzhang1/PythonProjects/omnivsod/results/InsGT/' + name)

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