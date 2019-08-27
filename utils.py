#------------------------ load the necessary packages ------------------------
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
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
    def __init__(self, ids_imgs_path, ids_objms_path, myTrans=None,
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
        self.transform = myTrans
        self.norm = normTrans

    def __getitem__(self, index):
        img_path = self.imgs[index]
        objm_path = self.objms[index]
        img = Image.open(img_path).convert('RGB')
        objm = Image.open(objm_path).convert('L')
        objm_db = Image.open(objm_path).convert('RGB')

        if self.transform is not None:
            imgTrans, objmTrans = data_ForTrain(self.norm)
            img = imgTrans(img)
            objm = objmTrans(objm)
            if debug_on:
                objm_db = objmTrans(objm_db)

        if debug_on:
            debug_vision(img, objm_db)

        return img, objm

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

     print ('under built...')

def data_ForTrain(normTransform):
    imgsTransform = transforms.Compose([
        transforms.Resize(size_train),
        transforms.ToTensor(),
        normTransform
    ])
    objmsTransform = transforms.Compose([
        transforms.Resize(size_train),
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
    img.numpy()
    objm.numpy()
    img = img.transpose(0,2)
    img = img.transpose(0,1)
    objm = objm.transpose(0,2)
    objm = objm.transpose(0,1)
    plt.figure
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(objm)
    plt.show()