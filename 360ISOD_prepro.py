import numpy as np
from PIL import Image

import re, os, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

import utils_360ISOD as utils

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

class PanoISOD_PP():
    def __init__(self):
        self.img_path = os.getcwd() + '/data/stimuli/'
        self.msk_path = os.getcwd() + '/data/mask_obj/'
        self.imgC_path = os.getcwd() + '/data_convert/stimuli_c/'
        self.mskC_path = os.getcwd() + '/data_convert/mask_obj_c/'

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


if __name__ == '__main__':
    pano_pp = PanoISOD_PP()
    pano_pp.epr2cmp()