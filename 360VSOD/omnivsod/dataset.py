import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms


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
        transforms.Resize([256, 512]),
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
        transforms.Resize([256, 512]),
        transforms.ToTensor(),
    ])
    msk_tensor = preprocess(msk)

    return msk_tensor