import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance

#several data augumentation strategies
def cv_random_flip(img, label,depth, fss):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(12):
            fss[i] = fss[i].transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth, fss
def randomCrop(image, label,depth, fss):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    for i in range(12):
        fss[i] = fss[i].crop(random_region)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region), fss
def randomRotation(image, label, depth, fss):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        for i in range(12):
            fss[i] = fss[i].rotate(random_angle, mode)
    return image, label, depth, fss
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img)  

# dataset for training
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, fs_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')]
        self.fss = [fs_root + f for f in os.listdir(fs_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.fss = sorted(self.fss)
        for idx in range(len(self.fss)):
            self.fss[idx] = self.fss[idx][74:]
        #self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])
        self.fs_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.binary_loader(self.depths[index])
        fss = self.fs_loader(self.images[index][:70], self.images[index][75:], self.fss)
        for i in range(12):
            fss[i] = fss[i].resize(image.size)
        image, gt, depth, fss = cv_random_flip(image, gt, depth, fss)
        image, gt, depth, fss = randomCrop(image, gt, depth, fss)
        image, gt, depth, fss = randomRotation(image, gt, depth, fss)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        for i in range(12):
            fss[i] = self.fs_transform(fss[i])
        
        return image, gt, depth, fss

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images)
        images = []
        gts = []
        depths=[]
        for img_path, gt_path,depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth= Image.open(depth_path)
            if img.size == gt.size and gt.size==depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths=depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def fs_loader(self, tr_pth, item, fs_list):
        fs_pth = tr_pth + '/FS/'
        m_list = []
        for fs in fs_list:
            temp = fs.split('_')
            m_list.append(temp[0])
        idxs = [i for i, x in enumerate(m_list) if x == item[:-4]]
        trg_fs_list = []
        for idx in range(len(idxs)):
            trg_fs_list.append(fs_list[idxs[idx]])
        temp_count_1 = int(12 / len(trg_fs_list))
        temp_count_2 = 12 % len(trg_fs_list)
        new_trg_fs_list = temp_count_1 * trg_fs_list + trg_fs_list[:temp_count_2]
        imgs = []
        for idx in range(12):
            img_pth = fs_pth + new_trg_fs_list[idx]
            with open(img_pth, 'rb') as f:
                img = Image.open(f)
                img.convert('RGB')
            imgs.append(img)

        return imgs

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size==depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size

#dataloader for training
def get_loader(image_root, gt_root, depth_root, fs_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, depth_root, fs_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

#test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, fs_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')]
        self.fss = [fs_root + f for f in os.listdir(fs_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.fss = sorted(self.fss)
        for idx in range(len(self.fss)):
            self.fss[idx] = self.fss[idx][73:]
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        fss = self.fs_loader(self.images[self.index][:69], self.images[self.index][74:], self.fss)
        for i in range(12):
            fss[i] = self.transform(fss[i]).unsqueeze(0)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, fss, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def fs_loader(self, te_pth, item, fs_list):
        fs_pth = te_pth + '/FS/'
        m_list = []
        for fs in fs_list:
            temp = fs.split('_')
            m_list.append(temp[0] + '_' + temp[1])
        idxs = [i for i, x in enumerate(m_list) if x == item[:-4]]
        trg_fs_list = []
        for idx in range(len(idxs)):
            trg_fs_list.append(fs_list[idxs[idx]])
        temp_count_1 = int(12 / len(trg_fs_list))
        temp_count_2 = 12 % len(trg_fs_list)
        new_trg_fs_list = temp_count_1 * trg_fs_list + trg_fs_list[:temp_count_2]
        imgs = []
        for idx in range(12):
            img_pth = fs_pth + new_trg_fs_list[idx]
            with open(img_pth, 'rb') as f:
                img = Image.open(f)
                img.convert('RGB')
            imgs.append(img)

        return imgs

    def __len__(self):
        return self.size

#test dataset and loader (with FS during testing)
class test_dataset_2:
    def __init__(self, image_root, gt_root, depth_root, fs_root, testsize, set_name):
        self.dataset_name = set_name
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')]
        self.fss = [fs_root + f for f in os.listdir(fs_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.fss = sorted(self.fss)
        for idx in range(len(self.fss)):
            if self.dataset_name == 'DUT-LF':
                self.fss[idx] = self.fss[idx][80:]
            elif self.dataset_name == 'HFUT':
                self.fss[idx] = self.fss[idx][78:]
            elif self.dataset_name == 'LFSD':
                self.fss[idx] = self.fss[idx][78:]
            elif self.dataset_name == 'Lytro':
                self.fss[idx] = self.fss[idx][79:]
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self, set_name):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if set_name == 'DUT-LF':
            fss = self.fs_loader(self.images[self.index][:76], self.images[self.index][81:], self.fss)
        elif set_name == 'HFUT':
            fss = self.fs_loader(self.images[self.index][:74], self.images[self.index][79:], self.fss)
        elif set_name == 'LFSD':
            fss = self.fs_loader(self.images[self.index][:74], self.images[self.index][79:], self.fss)
        elif set_name == 'Lytro':
            fss = self.fs_loader(self.images[self.index][:75], self.images[self.index][80:], self.fss)
        for i in range(12):
            fss[i] = self.transform(fss[i]).unsqueeze(0)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, fss, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def fs_loader(self, te_pth, item, fs_list):
        fs_pth = te_pth + '/FS/'
        m_list = []
        for fs in fs_list:
            temp = fs.split('_')
            m_list.append(temp[0])
        idxs = [i for i, x in enumerate(m_list) if x == item[:-4]]
        trg_fs_list = []
        for idx in range(len(idxs)):
            trg_fs_list.append(fs_list[idxs[idx]])
        temp_count_1 = int(12 / len(trg_fs_list))
        temp_count_2 = 12 % len(trg_fs_list)
        new_trg_fs_list = temp_count_1 * trg_fs_list + trg_fs_list[:temp_count_2]
        imgs = []
        for idx in range(12):
            img_pth = fs_pth + new_trg_fs_list[idx]
            with open(img_pth, 'rb') as f:
                img = Image.open(f)
                img.convert('RGB')
            imgs.append(img)

        return imgs

    def __len__(self):
        return self.size
