import numpy as np
from PIL import Image

import re, os, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

import utils_360ISOD as utils

from PIL import Image

complexity_path = os.getcwd() + '/analysis.txt'

def PanoISOD_e2c(e_img, face_w=256, mode='bilinear', cube_format='dice'):
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

def norm_entropy(blk, P=2097152): # the img must be the type of PIL.Image, 360ISOD default resolution is 1024*2048
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

def file_generater(complexity_imgs_normal):
    f = open(complexity_path, 'w')
    count = 1
    for item in complexity_imgs_normal:
        line = "The ranking of index is:".format(count) + ' ' + str(item) + '\n'
        f.write(line)
        count += 1
    f.close()

def dataset_sort(data):
    # sort the files
    index = sorted(range(len(data)), key=lambda k: data[k])

    return index

def dataset_split():
    pano_pp = PanoISOD_PP()

    index_list = []
    lvl_c = 1
    for lvl in range(4):
        complex_data = pano_pp.complexity_stt(p_max=(lvl + 1))
        index_list.append(dataset_sort(complex_data))
        print("The {} level processed".format(lvl_c))
        if lvl == 3:
           index_dataset = index_list[lvl]
           entropy_fixation = complex_data
        lvl_c += 1

    print()

class PanoISOD_PP():
    def __init__(self):
        self.img_path = os.getcwd() + '/data/stimuli/'
        self.msk_path = os.getcwd() + '/data/mask_obj/'
        self.imgC_path = os.getcwd() + '/data_convert/stimuli_c/'
        self.mskC_path = os.getcwd() + '/data_convert/mask_obj_c/'
        self.sal_path = os.getcwd() + '/saliency/'

    def epr2cmp(self):
        eprlist = os.listdir(self.img_path)
       # eprlist = os.listdir(self.msk_path)
        eprlist.sort(key=lambda x: x[:-4])

        count = 1
        for index in eprlist:
            if index.endswith('.png'):
                epr_path = os.path.join(os.path.abspath(self.img_path), index)
               # epr_path = os.path.join(os.path.abspath(self.msk_path), index)
                epr =  cv2.imread(epr_path)
                cmp = PanoISOD_e2c(epr, 512) # the face_width should be epr_width / 4
                cv2.imwrite(index, cmp)
                print(" {} images processed".format(count))
                count += 1

    def complexity_stt(self, p_max):
        sallist = os.listdir(self.sal_path)
        sallist.sort(key=lambda x: x[:-4])

        complexity_imgs = []
        complexity_imgs_normal = []
        complexity_imgs_dff = []
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


if __name__ == '__main__':
    print(dataset_split())