from pprint import pprint
from glob import glob
import copy, os
import shlex, subprocess

import matplotlib
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

import cv2

import numpy as np
import scipy
import pandas as pd
pd.set_eng_float_format(accuracy=3, use_eng_prefix=True)

from scipy import ndimage

import settings
import _pickle as pck

width_360ISOD = settings.width_360ISOD
height_360ISOD = settings.height_360ISOD

from PIL import Image
import scipy.stats

import PANOISDO_prepro


def salmap_from_norm_coords(norm_coords, sigma, height_width):
    '''
    Base function to compute general saliency maps, given the normalized (from 0 to 1)
    fixation coordinates, the sigma of the gaussian blur, and the height and
    width of the saliency map in pixels.
    '''
    img_coords = np.mod(np.round(norm_coords * np.array(height_width)), np.array(height_width) - 1.0).astype(int)

    gaze_counts = np.zeros((height_width[0], height_width[1]))
    for coord in img_coords:
        gaze_counts[coord[0], coord[1]] += 1.0

    gaze_counts[0, 0] = 0.0

    sigma_y = sigma
    salmap = ndimage.filters.gaussian_filter1d(gaze_counts, sigma=sigma_y, mode='wrap', axis=0)

    # In x-direction, we scale the radius of the gaussian kernel the closer we get to the pole
    for row in range(salmap.shape[0]):
        angle = (row / float(salmap.shape[0]) - 0.5) * np.pi
        sigma_x = sigma_y / (np.cos(angle) + 1e-3)
        salmap[row, :] = ndimage.filters.gaussian_filter1d(salmap[row, :], sigma=sigma_x, mode='wrap')

    salmap /= float(np.sum(salmap))
    return salmap

def get_gaze_salmap(list_of_runs, sigma_deg=1.0, height_width=(height_360ISOD, width_360ISOD)):
    '''Computes gaze saliency maps.'''
    fixation_coords = []

    for run in list_of_runs:
        relevant_fixations = run['gaze_fixations']

        if len(relevant_fixations.shape) > 1:
            _, unique_idcs = np.unique(relevant_fixations[:, 0], return_index=True)
            all_fixations = relevant_fixations[unique_idcs]
            fixation_coords.append(all_fixations)

    norm_coords = np.vstack(fixation_coords)[:, ::-1]

    return salmap_from_norm_coords(norm_coords, sigma_deg * height_width[1] / 360.0, height_width)

def IOC_func(pck_files):
    num_pck = len(pck_files)
    IOC_imgs = []
    for pck_idx in range(num_pck):
        print("---- Show the info of the {} pck ----".format(pck_idx+1))
        ioc_path = settings.IOC_PATH + format(str(pck_idx), '0>2s') + '.txt'

        # process the current pck file and compute the ioc of it
        IOC_list = load_one_out_logfile(pck_files[pck_idx], pck_idx)
        file_generater(IOC_list, ioc_path)
        IOC_imgs.append(np.mean(IOC_list))

    file_generater(IOC_imgs, settings.IOC_PATH_TT)
    print('All done !')

def file_generater(list, path):
    f = open(path, 'w')

    count = 1
    for item in list:
        line = str(item) + '\n'
        f.write(line)
        count += 1
    f.close()

def print_logfile_stats(log):
    print("Total of %d runs." % len(log))
    for i in range(4):
        print("Viewpoint %d: %d" % (i, len(log[log['viewpoint_idx'] == i])))

def load_logfile(path):
    with open(path, 'rb') as log_file:
        log = pck.load(log_file, encoding='latin1')
    print("Loaded %s." % path)
    print_logfile_stats(log)
    return log

def show_map_self(map, name, binary):
    if binary == 'False':
        salmap = cv2.normalize(map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imwrite(settings.MAP_PATH + name + '.png', salmap)
        print('Map saved !')

    elif binary == 'True':
        salmap, threshold = adaptive_threshold(map, 0.25)
        print("Threshold: {}; Map index: {}".format(threshold, name))

        # generate the binary map according to the adaptive threshold
        for r in range(height_360ISOD):
            for c in range(width_360ISOD):
                if salmap[r, c] >= threshold:
                    salmap[r, c] = 255
                else:
                    salmap[r, c] = 0

        cv2.imwrite(settings.MAP_PATH + name + '.png', salmap)
        print('Map saved !')

    else:
        print('No processing; Please check your input parameters...')
        return map

    return salmap

def adaptive_threshold(map, region_kept):
    # find the adaptive threshold of intensity to keep the 25% of image regions
    salmap = cv2.normalize(map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    histmap = np.histogram(salmap, bins=255)
    hist_pxl = histmap[0]

    pxl_count = 0
    abs_list = []
    for int_lvl in range(255):
        pxl_count += hist_pxl[int_lvl]
        ratio = pxl_count / (width_360ISOD * height_360ISOD)
        #print("Ratio of {} with the pixel intensity lower than {}".format(ratio, (int_lvl + 1)))
        abs_list.append(np.abs(ratio - 1 + region_kept)) # region_kept is empirical value, 25% in the IOC paper
    threshold = abs_list.index(min(abs_list)) + 1

    return salmap, threshold

def get_gaze_point(list_of_runs, height_width=(height_360ISOD, width_360ISOD)):
    '''Computes gaze saliency maps.'''
    fixation_coords = []

    for run in list_of_runs:
        relevant_fixations = run['gaze_fixations']

        if len(relevant_fixations.shape) > 1:
            _, unique_idcs = np.unique(relevant_fixations[:, 0], return_index=True)
            all_fixations = relevant_fixations[unique_idcs]
            fixation_coords.append(all_fixations)

    norm_coords = np.vstack(fixation_coords)[:, ::-1]

    return salpoint_from_norm_coords(norm_coords, height_width)

def salpoint_from_norm_coords(norm_coords, height_width):
    img_coords = np.mod(np.round(norm_coords * np.array(height_width)), np.array(height_width) - 1.0).astype(int)

    gaze_counts = np.zeros((height_width[0], height_width[1]))
    for coord in img_coords:
        gaze_counts[coord[0], coord[1]] += 1.0 # Here is the normalized map, resize to 255 for visualization

    gaze_counts[0, 0] = 0.0

    return gaze_counts

def load_one_out_logfile(path, img_idx):
    with open(path, 'rb') as log_file:
        log = pck.load(log_file, encoding='latin1')
    print("Loaded %s." % path)

    total_runs = len(log) # total E-observers participated for the current image
    list_ioc = []
    for run_idx in range(total_runs):
        log_one = log.iloc[[run_idx]]
        log_rest = log.drop([run_idx], axis=0)
        print("---- Show the info of IOC process of the {} runs ----".format(run_idx))
        print('---------------- The total info ----------------')
        print_logfile_stats(log)
        print('---------------- The one info ----------------')
        print_logfile_stats(log_one)
        print('---------------- The rest info ----------------')
        print_logfile_stats(log_rest)

        #compute the ioc of the current observer on the current image
        map_ori = get_gaze_salmap(log['data'])
        show_map_self(map_ori, 'total_' + format(str(img_idx), '0>2s'), 'False')

        map_rest = get_gaze_salmap(log_rest['data'])
        show_map_self(map_rest, 'rest_' + format(str(img_idx), '0>2s') + '_' + format(str(run_idx), '0>2s'), 'False')
        rest_fix_reg = show_map_self(map_rest, 'rest_binary_' + format(str(img_idx), '0>2s') + '_'
                                        + format(str(run_idx), '0>2s'), 'True')

        map_one = get_gaze_point(log_one['data'])
        one_fix_pos = show_map_self(map_one, 'one_' + format(str(img_idx), '0>2s') + '_' + format(str(run_idx), '0>2s'),
                                'False')

        count_in = 0
        count_out = 0
        for r in range(height_360ISOD):
            for c in range(width_360ISOD):
                if one_fix_pos[r, c] == 255:
                    if rest_fix_reg[r, c] == 255:
                        count_in += 1
                    else:
                        count_out += 1
        list_ioc.append(count_in / (count_in + count_out))

    return list_ioc

def norm_entropy(salmap, P= height_360ISOD * width_360ISOD):
    salmap = np.array(salmap)
    hist_map = np.histogram(salmap, bins=255)
    map_norm = hist_map[0] / P
    entropy_list = []

    for idx in range(255):
        p_idx = map_norm[idx]
        if p_idx != 0:
            entropy_list.append(p_idx * np.log2(p_idx))
        else:
            entropy_list.append(0)

    entropy = -1 * np.sum(entropy_list)

    return entropy

def shannon_entropy(salmap, P= height_360ISOD * width_360ISOD):
    salmap = np.array(salmap)
    salmap = salmap / 255

    entropy = 0
    for r in range(height_360ISOD):
        for c in range(width_360ISOD):
            if salmap[r, c] != 0:
                entropy = entropy + salmap[r, c] * salmap[r, c] * np.log2(salmap[r, c] * salmap[r, c])

    return -1 * entropy

def entropy_func():
    salmaps = os.listdir(settings.SALIENCY_PATH)
    salmaps.sort(key=lambda x: x[:-4])

    entropy_list = []
    for idx in salmaps:
        if idx.endswith('.png'):
            sal_path = os.path.join(os.path.abspath(settings.SALIENCY_PATH), idx)
            sal_map = Image.open(sal_path).convert('L')
            entropy_list.append(norm_entropy(sal_map))
            #entropy_list.append(shannon_entropy(sal_map))
            print(" {} images processed".format(idx))

    file_generater(entropy_list, settings.ENTROPY_PATH)

def IOC_func_2():
    data_loader = PANOISDO_prepro.FixPos_PP()
    crds, starts = data_loader.load_raw()

    num_imgs = len(crds)
    IOC_imgs = []
    for idx in range(num_imgs):
        print("---- Show the info of the {} image ----".format(idx + 1))
        ioc_path = settings.IOC_2_PATH + format(str(idx+1), '0>2s') + '.txt'

        IOC_list = load_one_out_txt(crds[idx], starts[idx], idx)
        file_generater(IOC_list, ioc_path)
        IOC_imgs.append(np.mean(IOC_list))

    file_generater(IOC_imgs, settings.IOC_2_PATH_TT)
    print('All done !')

def load_one_out_txt(fixations, starts_idx, img_idx):
    # save the fixation map
    map_ori = salmap_from_norm_coords(fixations, 1.0 * width_360ISOD / 360.0, (height_360ISOD, width_360ISOD))
    map_ori = cv2.normalize(map_ori, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imwrite(format(str(img_idx+1), '0>3s') + '.png', map_ori)

    total_runs = len(starts_idx)
    print("---- There are {} runs in total ----".format(total_runs))
    list_ioc = []
    run_list = []
    for run_idx in range(total_runs):
        run_list.append(PANOISDO_prepro.get_fixpos(fixations, starts_idx, run_idx))

    for pro_idx in range(total_runs):
        run_one = run_list[pro_idx]

        if pro_idx == 0:
            run_rest = np.concatenate((run_list[1:]), axis=0)
        elif pro_idx == total_runs - 1:
            run_rest = np.concatenate((run_list[:-1]), axis=0)
        else:
            run_rest_s1 = np.concatenate((run_list[:pro_idx]), axis=0)
            run_rest_s2 = np.concatenate((run_list[(pro_idx+1):]), axis=0)
            run_rest = np.concatenate((run_rest_s1, run_rest_s2), axis=0)

        #compute the ioc of the current run on the current image
        map_one = salpoint_from_norm_coords(run_one, (height_360ISOD, width_360ISOD))
        map_one = cv2.normalize(map_one, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imwrite(format(str(img_idx + 1), '0>3s') + '_run_one_' + format(str(pro_idx), '0>3s') + '.png', map_one)

        map_rest = salmap_from_norm_coords(run_rest, 1.0 * width_360ISOD / 360.0, (height_360ISOD, width_360ISOD))
        map_rest, threshold = adaptive_threshold(map_rest, 0.25)
        print("Threshold: {}; Map index: {}".format(threshold, pro_idx))

        for r in range(height_360ISOD):
            for c in range(width_360ISOD):
                if map_rest[r, c] >= threshold:
                    map_rest[r, c] = 255
                else:
                    map_rest[r, c] = 0

        cv2.imwrite(format(str(img_idx + 1), '0>3s') + '_run_rest_' + format(str(pro_idx), '0>3s') + '.png', map_rest)

        count_in = 0
        count_out = 0
        for r in range(height_360ISOD):
            for c in range(width_360ISOD):
                if map_one[r, c] == 255:
                    if map_rest[r, c] == 255:
                        count_in += 1
                    else:
                        count_out += 1
        list_ioc.append(count_in / (count_in + count_out))

    return list_ioc

if __name__ == '__main__':
    #all_files = sorted(glob(os.path.join(settings.DATASET_PATH_VR, '*.pck')))
    #IOC_func(all_files)
    #entropy_func()
    IOC_func_2()
