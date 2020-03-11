import numpy as np
import pano_utils as utils
import os
import xml.etree.ElementTree as ET
from PIL import Image


def e2c(e_img, face_w, mode='bilinear', cube_format='list'):
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

def c2e(cubemap, h, w, mode='bilinear', cube_format='list'):
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

def bbox2sod(name):
    crop_path = os.getcwd() + '/SOD_crop/' + name
    path = os.getcwd() + '/xml_files/' + name
    raw_path = os.getcwd() + '/jpg_frames/' + name
    xml_ids = os.listdir(path)

    # process every frame; get the perspective image with only one salient object
    xml_count = 0
    for xml_id in xml_ids:
        xml_path = path + '/' + xml_id
        xml_tree = ET.parse(xml_path)
        xml_root  = xml_tree.getroot()
        bbox_list = []
        obj_list = []

        for obj in xml_root.iter('object'):
            obj_list.append(obj.find('name').text)
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            bbox_list.append([xmin, ymin, xmax, ymax])

        filename = xml_root.find('filename').text
        frm_path = raw_path + '/' + filename[:-4] + '.jpg'
        frm = Image.open(frm_path)

        num_sod = len(obj_list)
        for idx in range(num_sod):
            frm_crop = frm.crop(bbox_list[idx])
            frm_crop.save(crop_path + '/' + name + '_' + format(filename[:-4], '0>4s') + '_' + obj_list[idx] + '.png')

        xml_count += 1
        print("{} frames processed.".format(xml_count))
