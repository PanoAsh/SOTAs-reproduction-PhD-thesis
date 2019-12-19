import numpy as np
from PIL import Image

import re, os, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import random

import settings
import pano_utils as utils
import PANOISOD_analysis

def PanoISOD_e2c(e_img, face_w, mode='bilinear', cube_format='dice'):
    '''
    e_img:  ndarray in shape of [H, W, *]
    face_w: int, the length of each face of the cubemap
    '''
    assert len(e_img.shape) == 3
    h, w = e_img.shape[:2]
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    xyz = utils.xyzcube(face_w)
    uv = utils.xyz2uv(xyz)
    coor_xy = utils.uv2coor(uv, h, w)

    cubemap = np.stack([
        utils.sample_equirec(e_img[..., i], coor_xy, order=order)
        for i in range(e_img.shape[2])
    ], axis=-1)

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = utils.cube_h2list(cubemap)
    elif cube_format == 'dict':
        cubemap = utils.cube_h2dict(cubemap)
    elif cube_format == 'dice':
        cubemap = utils.cube_h2dice(cubemap)
    else:
        raise NotImplementedError()

    return cubemap

def PanoISOD_c2e(cubemap, h, w, mode='bilinear', cube_format='list'):
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = utils.cube_list2h(cubemap)
    elif cube_format == 'dict':
        cubemap = utils.cube_dict2h(cubemap)
    elif cube_format == 'dice':
        cubemap = utils.cube_dice2h(cubemap)
    else:
        raise NotImplementedError('unknown cube_format')
    assert len(cubemap.shape) == 3
    assert cubemap.shape[0] * 6 == cubemap.shape[1]
    assert w % 8 == 0
    face_w = cubemap.shape[0]

    uv = utils.equirect_uvgrid(h, w)
    u, v = np.split(uv, 2, axis=-1)
    u = u[..., 0]
    v = v[..., 0]
    cube_faces = np.stack(np.split(cubemap, 6, 1), 0)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = utils.equirect_facetype(h, w)
    coor_x = np.zeros((h, w))
    coor_y = np.zeros((h, w))

    for i in range(4):
        mask = (tp == i)
        coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
        coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

    mask = (tp == 4)
    c = 0.5 * np.tan(np.pi / 2 - v[mask])
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = c * np.cos(u[mask])

    mask = (tp == 5)
    c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = -c * np.cos(u[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    equirec = np.stack([
        utils.sample_cubefaces(cube_faces[..., i], tp, coor_y, coor_x, order=order)
        for i in range(cube_faces.shape[3])
    ], axis=-1)

    return equirec

def uv_rotate(uv):
    max_lai, max_log = debug_uv_rot(uv)
    uv_size = uv.shape

    to_right = max_lai / 3
    to_top = max_log / 3

    uv[:, :, 0] = uv[:, :, 0] + to_right
    uv[:, :, 1] = uv[:, :, 1] + to_top

    for r in range(uv_size[0]):
        for c in range(uv_size[1]):
            if uv[r, c, 0] > max_lai:
                uv[r, c, 0] = -2 * max_lai + uv[r, c, 0]
            if uv[r, c, 1] > max_log:
                uv[r, c, 0] = -2 * max_log + uv[r, c, 0]

    return uv

def debug_uv_rot(uv):
    debug = []
    debug.append(np.max(uv[:, :, 0]))
   # debug.append(np.min(uv[:, :, 0]))
    debug.append(np.max(uv[:, :, 1]))
    #debug.append(np.min(uv[:, :, 1]))

    return debug

def data_MultiCrop(scale, img, Height, Width):
    h = Height / scale # the height of each block
    w = Width / scale # the width of each block
    imgs = []

    for i in range(scale):
        for j in range(scale):
            y1 = i * h
            x1 = j * w
            y2 = y1 + h
            x2 = x1 + w
            imgs.append(img.crop((x1, y1, x2, y2)))

    return imgs

def debug_show(img_list, scale, num_block):
    fig = plt.figure()

    for idx in range(num_block):
        fig.add_subplot(scale, scale, idx+1)
        plt.imshow(img_list[idx],cmap='gray')

    plt.show()
    print()

def norm_entropy(blk, P=settings.width_360ISOD * settings.height_360ISOD):
    # the img must be the type of PIL.Image, 360ISOD default resolution is 1024*2048
    blk = np.array(blk)
    hist_blk = np.histogram(blk, bins=255)
    blk_norm = hist_blk[0] / P
    entropy_list = []

    for idx in range(255):
        p_idx = blk_norm[idx]
        if p_idx != 0:
            entropy_list.append(p_idx * np.log2(p_idx))
        else:
            entropy_list.append(0)

    entropy = -1 * np.sum(entropy_list)

    return entropy

def file_generater(file, mode):
    if mode == 'train':
        f = open(settings.TRAIN_TXT_PATH, 'w')
    if mode == 'test':
        f = open(settings.TEST_TXT_PATH, 'w')
    if mode == 'easy':
        f = open(settings.EASY_TXT_PATH, 'w')
    if mode == 'medium':
        f = open(settings.MEDIUM_TXT_PATH, 'w')
    if mode == 'hard':
        f = open(settings.HARD_TXT_PATH, 'w')

    count = 1
    for item in file:
        line = format(str(item), '0>3s') + '.png' + '\n'
        f.write(line)
        count += 1
    f.close()

def dataset_sort(data, h2l= False):
    # sort the files
    index = sorted(range(len(data)), key=lambda k: data[k], reverse=h2l)

    return index

def entropy_save():
    pano_pp = PanoISOD_PP()

    entropy_list = []
    lvl_c = 1
    for lvl in range(32):
        complex_data = pano_pp.complexity_stt(p_max=(lvl + 1))
        entropy_list.append(complex_data)
        print("The {} level processed".format(lvl_c))
        lvl_c += 1
        if lvl % 3 == 0:
            print('saving entropy...')
            file_generater(entropy_list)
    print('data saved !')

def dataset_split(level_data=3, level_ent=20):
    # the muiti-level entropy is set to be 20 for the 360ISOD dataset;
    # the dataset is divided into three levels according to the entropy-based complexity analysis

    pano_pp = PanoISOD_PP()
    entropy_data = pano_pp.complexity_stt(p_max=(level_ent))
    entropy_index = dataset_sort(entropy_data)
    entropy_index = [x + 1 for x in entropy_index] # the image index should be in the range of [1-107]

    if level_data == 3:
        index_easy = entropy_index[:35]
        index_medium = entropy_index[35:70]
        index_hard = entropy_index[70:]

        random.shuffle(index_easy)
        random.shuffle(index_medium)
        random.shuffle(index_hard)

        train_index = index_easy[:31] + index_medium[:31] + index_hard[:33]
        test_index = index_easy[31:] + index_medium[31:] + index_hard[33:]

        file_generater(index_easy, 'easy')
        file_generater(index_medium, 'medium')
        file_generater(index_hard, 'hard')
        file_generater(train_index, 'train')
        file_generater(test_index, 'test')
        print('All done !')

def to_train():
    img_idx = open(settings.TRAIN_TXT_PATH)
    img_T_idx = open(settings.TEST_TXT_PATH)

    count = 1
    for id in img_idx:
        src = os.path.join(os.path.abspath(settings.PANOISOD_IMG_PATH), id[:-1])
        src2 = os.path.join(os.path.abspath(settings.PANOISOD_MSK_PATH), id[:-1])
        dst = os.path.join(os.path.abspath(settings.PANOISOD_IMG_TRAIN_PATH), id[:-1])
        dst2 = os.path.join(os.path.abspath(settings.PANOISOD_MSK_TRAIN_PATH), id[:-1])
        try:
            os.rename(src, dst)
            os.rename(src2, dst2)
        except:
            continue
        print(" {} images processed".format(count))
        count += 1

    count = 1
    for id in img_T_idx:
        src = os.path.join(os.path.abspath(settings.PANOISOD_IMG_PATH), id[:-1])
        src2 = os.path.join(os.path.abspath(settings.PANOISOD_MSK_PATH), id[:-1])
        dst = os.path.join(os.path.abspath(settings.PANOISOD_IMG_TEST_PATH), id[:-1])
        dst2 = os.path.join(os.path.abspath(settings.PANOISOD_MSK_TEST_PATH), id[:-1])
        try:
            os.rename(src, dst)
            os.rename(src2, dst2)
        except:
            continue
        print(" {} images processed".format(count))
        count += 1

    print('All done !')


class PanoISOD_PP():
    def __init__(self):
        self.cub_path = settings.CMP_PATH
        self.sti_path = settings.ERP_PATH

    def erp2cmp(self):
        erplist = os.listdir(self.sti_path)
        erplist.sort(key=lambda x: x[:-4])

        count = 1
        for index in erplist:
            if index.endswith('.png'):
                erp_path = os.path.join(os.path.abspath(self.sti_path), index)
                cmp_path = os.path.join(os.path.abspath(self.cub_path), index)
                erp = cv2.imread(erp_path)
                cmp = PanoISOD_e2c(erp, 512) # the face_width should be epr_width / 4
                cv2.imwrite(cmp_path, cmp)
                print(" {} images processed".format(count))
                count += 1

    def cmp2erp(self):
        cmplist = os.listdir(self.cub_path)
        cmplist.sort(key=lambda x: x[:-4])

        test_patches = []
        for index in cmplist:
            if index.endswith('.jpg'):
                if int(index[4]) == 1 and int(index[6]) == 1:
                    test_patches.append(index)

        count = 1
        num_test = int(len(test_patches) / 6)
        for idx in range(num_test):
            patch_list = []
            for fac in range(6):
                patch_path = os.path.join(os.path.abspath(self.cub_path), test_patches[6*idx+fac])
                patch_list.append(cv2.imread(patch_path))

            patch_list[1] = cv2.flip(patch_list[1], 1)
            patch_list[2] = cv2.flip(patch_list[2], 1)
            patch_list[4] = cv2.flip(patch_list[4], 0)

            erp = PanoISOD_c2e(patch_list, settings.height_360ISOD, settings.width_360ISOD)
            erp_name = test_patches[idx*6]
            erp_name = erp_name[0:3] + '.png'
            erp_path = os.path.join(os.path.abspath(self.sti_path), erp_name)
            cv2.imwrite(erp_path, erp)
            print(" {} images processed".format(count))
            count += 1

    def resize_eval(self):
        filelist = os.listdir(os.getcwd() + '/resize')

        count = 1
        for item in filelist:
            img_path = os.path.join(os.path.abspath(os.getcwd() + '/resize'), item)
            if item.endswith('.png'):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (settings.width_360ISOD, settings.height_360ISOD))
                cv2.imwrite(item, img)
                print(" {} images processed".format(count))
                count += 1

    def complexity_stt(self, p_max): # multi-level entropy
        sallist = os.listdir(self.sal_path)
        sallist.sort(key=lambda x: x[:-4])

        complexity_imgs = []
        count = 1
        for idx in sallist:
            if idx.endswith('.png'):
                sal_path = os.path.join(os.path.abspath(self.sal_path), idx)
                sal_map = Image.open(sal_path).convert('L')

                # apply muiti-crop on the salient map
                entropy_multi_level = []
                for p in range(p_max):
                    scale = p+1
                    num_blk = scale * scale
                    blocks = data_MultiCrop(scale, sal_map, 1024, 2048)

                    # debug_show(blocks, scale, num_blk) # show the image blocks

                    # compute the entropy of each block at the current level
                    entropy_multi_block = []
                    for ent in range(num_blk):
                        ent_blk = norm_entropy(blocks[ent])
                        entropy_multi_block.append(ent_blk)
                    entropy_multi_level.append(np.sum(entropy_multi_block))

                complexity_imgs.append(1 / p_max * np.sum(entropy_multi_level))
                print(" {} images processed".format(count))
                count += 1

        return complexity_imgs


class FixPos_PP():
    def __init__(self):
        self.fixpos_l_path = settings.L_PATH_TGT
        self.fixpos_r_path = settings.R_PATH_TGT

    def load_raw(self):
        pathIL = self.fixpos_l_path
        pathIR = self.fixpos_r_path

        fixlist = os.listdir(pathIL)
        fixlist.sort(key=lambda x: x[:-4])
        count = 1
        crds_list = []
        starts_list = []
        for item in fixlist:
            fixposL_path = os.path.join(os.path.abspath(pathIL), item)
            fixposR_path = os.path.join(os.path.abspath(pathIR), item)
            fixposL = np.loadtxt(fixposL_path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
            fixposR = np.loadtxt(fixposR_path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
            starts_fixposL = get_starts(fixposL)
            starts_fixposR = get_starts(fixposR)

            # combine all the fixations on the current image
            fix_crdL = fixposL[:, 1:3]
            fix_crdL.T[[0, 1]] = fix_crdL.T[[1, 0]]
            fix_crdR = fixposR[:, 1:3]
            fix_crdR.T[[0, 1]] = fix_crdR.T[[1, 0]]
            fix_coords = np.concatenate((fix_crdL, fix_crdR), axis=0)
            starts_fixposR = starts_fixposR + len(fixposL)
            fix_starts = np.concatenate((starts_fixposL, starts_fixposR), axis=0)

            crds_list.append(fix_coords)
            starts_list.append(fix_starts)
            num_runs = len(fix_starts)
            print("There are {} runs for the {} image.".format(num_runs, count))
            count += 1

        if count != len(fixlist) + 1:
            print('Fail to load all data; Please check...')
        else:
            print('All data loaded !')

        return crds_list, starts_list


def get_fixpos(fixationList, startPositions, scanpathIdx=0):
    if scanpathIdx >= startPositions.shape[0]-1:
        range_ = np.arange(startPositions[scanpathIdx], fixationList.shape[0])
    else:
        range_ = np.arange(startPositions[scanpathIdx], startPositions[scanpathIdx+1])

    #print(range_, startPositions[scanpathIdx])
    return fixationList[range_, :].copy()

def get_starts(fix_list):

    return np.where(fix_list[:, 0] == 0)[0]

def fix2heat(path, reg=0.25): # to keep the 25% of the images based on the ranking of saliency
    fixlist = os.listdir(path)
    fixlist.sort(key=lambda x: x[:-4])

    for item in fixlist:
        fix_path = os.path.join(os.path.abspath(path), item)
        fix_map = Image.open(fix_path).convert('L')
        fix_map = np.array(fix_map)
        fix_map, thr = PANOISOD_analysis.adaptive_threshold(fix_map, reg)

        for r in range(settings.height_360ISOD):
            for c in range(settings.width_360ISOD):
                if fix_map[r, c] < thr:
                    fix_map[r, c] = 0

        heat_map = cv2.applyColorMap(fix_map, cv2.COLORMAP_HOT)
        cv2.imwrite(item, heat_map)
        print(item + ' ' + 'has been processed.')

def dataset_split_IOC(level_data=3):
    # the dataset is divided into three levels according to the IOC-based complexity analysis

    ioc_data = np.loadtxt(settings.IOC_LOAD_PATH)
    ioc_index = dataset_sort(ioc_data, h2l=True) # A higher IOC means better convergence
    ioc_index = [x + 1 for x in ioc_index] # the image index should be in the range of [1-107]

    if level_data == 3:
        index_easy = ioc_index[:35] # 35
        index_medium = ioc_index[35:70] # 35
        index_hard = ioc_index[70:] # 37

        random.shuffle(index_easy)
        random.shuffle(index_medium)
        random.shuffle(index_hard)

        train_index = index_easy[:31] + index_medium[:31] + index_hard[:33] # 95
        test_index = index_easy[31:] + index_medium[31:] + index_hard[33:] # 12

        file_generater(index_easy, 'easy')
        file_generater(index_medium, 'medium')
        file_generater(index_hard, 'hard')
        file_generater(train_index, 'train')
        file_generater(test_index, 'test')
        print('All done !')

def dataset_split_IOC_2(mode='nantes'):
    # the dataset is divided into three levels according to the IOC-based complexity analysis
    # Due to the difference of E-tracking observers in Nantes_85 and Stanford_22, re-split the dataset for better analysis

    if mode == 'nantes':
        ioc_data = np.loadtxt(settings.IOC_NTS_LOAD_PATH)
        ioc_index = dataset_sort(ioc_data, h2l=True)  # A higher IOC means better convergence
        ioc_index = [x + 1 for x in ioc_index]  # the image index should be in the range of [1-85]

        index_easy = ioc_index[:28] # 28
        index_medium = ioc_index[28:56] # 28
        index_hard = ioc_index[56:] # 29

        random.shuffle(index_easy)
        random.shuffle(index_medium)
        random.shuffle(index_hard)

        train_index = index_easy[:22] + index_medium[:22] + index_hard[:24] # 68
        test_index = index_easy[22:] + index_medium[22:] + index_hard[24:] # 17

        file_generater(index_easy, 'easy')
        file_generater(index_medium, 'medium')
        file_generater(index_hard, 'hard')
        file_generater(train_index, 'train')
        file_generater(test_index, 'test')
        print('All done !')

    elif mode == 'stanford':
        ioc_data = np.loadtxt(settings.IOC_STF_LOAD_PATH)
        ioc_index = dataset_sort(ioc_data, h2l=True)  # A higher IOC means better convergence
        ioc_index = [x + 1 for x in ioc_index]  # the image index should be in the range of [1-22]

        index_easy = ioc_index[:7] # 7
        index_medium = ioc_index[7:14] # 7
        index_hard = ioc_index[14:] # 8

        file_generater(index_easy, 'easy')
        file_generater(index_medium, 'medium')
        file_generater(index_hard, 'hard')
        print('All done !')

    else:
        print('No processing done, please check your input parameters.')


if __name__ == '__main__':
    print('waiting...')
    #to_train()
    #dataset_split_IOC_2('stanford')
    pp = PanoISOD_PP()
    #pp.erp2cmp()
    #pp.cmp2erp()
    pp.resize_eval()
    # pp = FixPos_PP()
    #pp.load_raw()
    #fix2heat(settings.SALIENCY_PATH)
