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

# Helper functions
def cond_mkdir(path):
    '''Helper function to create a directory if it doesn't exist already.'''
    if not os.path.exists(path):
        os.makedirs(path)


def gnomonic2lat_lon(x_y_coords, fov_vert_hor, center_lat_lon):
    '''
    Converts gnomonic (x, y) coordinates to (latitude, longitude) coordinates.

    x_y_coords: numpy array of floats of shape (num_coords, 2)
    fov_vert_hor: tuple of vertical, horizontal field of view in degree
    center_lat_lon: The (lat, lon) coordinates of the center of the viewport that the x_y_coords are referencing.
    '''
    sphere_radius_lon = 1. / (2.0 * np.tan(np.radians(fov_vert_hor[1] / 2.0)))
    sphere_radius_lat = 1. / (2.0 * np.tan(np.radians(fov_vert_hor[0] / 2.0)))

    x, y = x_y_coords[:, 0], x_y_coords[:, 1]

    x_y_hom = np.column_stack([x.ravel(), y.ravel(), np.ones(len(x.ravel()))])

    K_inv = np.zeros((3, 3))
    K_inv[0, 0] = 1.0 / sphere_radius_lon
    K_inv[1, 1] = 1.0 / sphere_radius_lat
    K_inv[0, 2] = -1. / (2.0 * sphere_radius_lon)
    K_inv[1, 2] = -1. / (2.0 * sphere_radius_lat)
    K_inv[2, 2] = 1.0

    R_lat = np.zeros((3, 3))
    R_lat[0, 0] = 1.0
    R_lat[1, 1] = np.cos(np.radians(-center_lat_lon[0]))
    R_lat[2, 2] = R_lat[1, 1]
    R_lat[1, 2] = -1.0 * np.sin(np.radians(-center_lat_lon[0]))
    R_lat[2, 1] = -1.0 * R_lat[1, 2]

    R_lon = np.zeros((3, 3))
    R_lon[2, 2] = 1.0
    R_lon[0, 0] = np.cos(np.radians(-center_lat_lon[1]))
    R_lon[1, 1] = R_lon[0, 0]
    R_lon[0, 1] = - np.sin(np.radians(-center_lat_lon[1]))
    R_lon[1, 0] = - R_lon[0, 1]

    R_full = np.matmul(R_lon, R_lat)

    dot_prod = np.sum(np.matmul(R_full, K_inv).reshape(1, 3, 3) * x_y_hom.reshape(-1, 1, 3), axis=2)

    sphere_points = dot_prod / np.linalg.norm(dot_prod, axis=1, keepdims=True)

    lat = np.degrees(np.arccos(sphere_points[:, 2]))
    lon = np.degrees(np.arctan2(sphere_points[:, 0], sphere_points[:, 1]))

    lat_lon = np.column_stack([lat, lon])
    lat_lon = np.mod(lat_lon, np.array([180.0, 360.0]))

    return lat_lon


def angle2img(lat_lon_array, img_height_width):
    '''
    Convertes an array of latitude, longitude coordinates to image coordinates with range (0, height) x (0, width)
    '''
    return lat_lon_array / np.array([180., 360.]).reshape(1, 2) * np.array(img_height_width).reshape(1, 2)


def stitch2video(video_name, frame_dir, fps=30., print_output=False):
    '''
    Uses ffmpeg to stitch together a bunch of frames to a video. frame_dir has to be an absolute path.
    '''
    framename_format_string = os.path.join(frame_dir, "%06d.png")
    ffmpeg_cmd = "ffmpeg -r %d -f image2 -s 1920x1080 -i %s -vcodec libx264 -crf 25  %s -y" % \
                 (fps, framename_format_string, video_name)
    args = shlex.split(ffmpeg_cmd)

    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()

    if print_output:
        pprint(output)
        pprint(err)


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


def get_head_salmap(list_of_runs, height_width=(720, 1440)):
    '''Computes head saliency maps.'''
    thresh = 37.196
    all_head_velos = []
    all_head_lat_lons = []

    for run in list_of_runs:
        all_head_velos.append(run['ang_head_velo'])
        all_head_lat_lons.append(run['headLatLon'])

    head_velos = np.vstack(all_head_velos)
    head_lat_lons = np.vstack(all_head_lat_lons)

    fixation_idcs = head_velos[:, 1] < thresh
    fix_lat_lons = head_lat_lons[fixation_idcs]
    norm_fix_coords = fix_lat_lons / np.array([180, 360])

    # Get this cubemap's gaze salmap
    salmap = salmap_from_norm_coords(norm_fix_coords, sigma=19.0, height_width=height_width)

    return salmap


def overlay_image_salmap(img_path, salmap):
    '''Overlays an image with a saliency map.'''
    image = matplotlib.image.imread(img_path).astype(float)[:, :, :3]
    salmap_resized = cv2.resize(salmap, (image.shape[1], image.shape[0])).astype(float)

    fig, ax = plt.subplots(frameon=False)
    fig.set_size_inches(16, 8)

    ax.imshow(image)
    ax.imshow(salmap_resized, cmap=plt.cm.jet, alpha=0.4)

    ax.axis('tight')
    ax.axis('off')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    return fig, ax


def filter_starting_point(list_of_runs, threshold_deg=20.0):
    '''Filters all measurements of a run that are at the
    beginning and in a vicinity of threshold_deg around the starting coordinates.'''
    cleaned_runs = []
    for run in list_of_runs:
        init_starting_point_lon = run['gaze_lat_lon'][0, 1]
        outside_bool = np.absolute(init_starting_point_lon - run['gaze_lat_lon'][:, 1]) > threshold_deg
        if np.any(outside_bool):
            first_left = np.amin(np.where(outside_bool)[0])
            cleaned_runs.append({key: value[first_left:] for key, value in run.iteritems()})
        else:
            cleaned_runs.append(run)
    return cleaned_runs


def get_data_matrix(list_of_runs, column_set='vr'):
    '''Joins specified columns of a list of runs to a single data matrix
    gaze_lat_lon_offset_lon'''
    if column_set == 'vr':
        columns = ['gaze_lat_lon_offset', 'gaze_fixations_single', 'ang_gaze_offset_velo', 'ang_head_velo',
                   'ang_gaze_velo', 'headTilt', 'gaze_fixations_ind']
    else:
        columns = ['gaze_lat_lon_offset', 'gaze_fixations_single', 'ang_gaze_offset_velo', 'ang_head_velo',
                   'ang_gaze_velo', 'gaze_fixations_ind']

    data = []
    column_names = []
    for column in columns:
        array_list = []

        for run in list_of_runs:
            array_list.append(np.squeeze(np.array(run[column])))

        if len(array_list[0].shape) == 1:
            joined_array = np.concatenate(array_list)

            data.append(joined_array)
            column_names.append(column)
        else:
            joined_array = np.vstack(array_list)

            data.append(joined_array[:, 0])
            data.append(joined_array[:, 1])
            column_names.append(column + '_lat')
            column_names.append(column + '_lon')

    return np.column_stack(data), column_names


def reject_gauss_outliers(data, column_names, m=2, take_absolute=True):
    '''Function to reject the outliers visible in above plot. Assumes a fundamentally normal distribution,
    which is close to what we see in the data.'''
    if take_absolute:
        data_ = np.absolute(data)
    else:
        data_ = np.copy(data)

    data_ = np.ma.masked_invalid(data_)

    col_means = np.nanmean(data_, axis=0, keepdims=True)
    centered_data = data_ - col_means
    column_stds = np.nanstd(data_, axis=0)

    bad_values_bool = np.absolute(centered_data) > m * column_stds
    bad_values = np.where(bad_values_bool)

    cleaned_data = np.copy(data)
    cleaned_data[bad_values] = np.take(col_means, bad_values[1])

    print("Rejection percentages:")
    for i in range(bad_values_bool.shape[1]):
        print("Column %s: %0.4f" % (column_names[i],
                                    float(np.sum(bad_values_bool[:, i], axis=0)) / bad_values_bool.shape[0]))

    return cleaned_data, col_means, column_stds


def plot_with_viewport(img, viewport_coords, out_path):
    viewport_coords_resh = viewport_coords.reshape(800, 800, 2)
    upper_line = viewport_coords_resh[0, :, :]
    lower_line = viewport_coords_resh[-1, :, :]
    right_line = viewport_coords_resh[:, -1, :]
    left_line = viewport_coords_resh[:, 0, :]

    lines = [upper_line, lower_line, right_line, left_line]

    split_lines = []
    for line in lines:
        diff = np.diff(line, axis=0)
        wrap_idcs = np.where(np.abs(diff) > 10)[0]
        if not len(wrap_idcs):
            split_lines.append(line)
        else:
            split_lines.append(line[:wrap_idcs[0] + 1])
            split_lines.append(line[wrap_idcs[0] + 1:])

    fig, ax = plt.subplots(frameon=False)
    fig.set_size_inches(48, 24)
    ax.imshow(img)

    for line in split_lines:
        ax.plot(line[:, 1], line[:, 0], color='black', linewidth=10)

    for line in split_lines:
        ax.plot(line[:, 1], line[:, 0], color='white', linewidth=8)

    ax.axis('tight')
    ax.axis('off')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    fig.clf()
    plt.close()


def get_gnomonic_hom(center_lat_lon, origin_image, height_width, fov_vert_hor=(60.0, 60.0)):
    '''Extracts a gnomonic viewport with height_width from origin_image
    at center_lat_lon with field of view fov_vert_hor.
    '''
    org_height_width, _ = origin_image.shape[:2], origin_image.shape[-1]
    height, width = height_width

    if len(origin_image.shape) == 3:
        result_image = np.zeros((height, width, 3))
    else:
        result_image = np.zeros((height, width))

    sphere_radius_lon = width / (2.0 * np.tan(np.radians(fov_vert_hor[1] / 2.0)))
    sphere_radius_lat = height / (2.0 * np.tan(np.radians(fov_vert_hor[0] / 2.0)))

    y, x = np.mgrid[0:height, 0:width]
    x_y_hom = np.column_stack([x.ravel(), y.ravel(), np.ones(len(x.ravel()))])

    K_inv = np.zeros((3, 3))
    K_inv[0, 0] = 1.0 / sphere_radius_lon
    K_inv[1, 1] = 1.0 / sphere_radius_lat
    K_inv[0, 2] = -width / (2.0 * sphere_radius_lon)
    K_inv[1, 2] = -height / (2.0 * sphere_radius_lat)
    K_inv[2, 2] = 1.0

    R_lat = np.zeros((3, 3))
    R_lat[0, 0] = 1.0
    R_lat[1, 1] = np.cos(np.radians(-center_lat_lon[0]))
    R_lat[2, 2] = R_lat[1, 1]
    R_lat[1, 2] = -1.0 * np.sin(np.radians(-center_lat_lon[0]))
    R_lat[2, 1] = -1.0 * R_lat[1, 2]

    R_lon = np.zeros((3, 3))
    R_lon[2, 2] = 1.0
    R_lon[0, 0] = np.cos(np.radians(-center_lat_lon[1]))
    R_lon[1, 1] = R_lon[0, 0]
    R_lon[0, 1] = - np.sin(np.radians(-center_lat_lon[1]))
    R_lon[1, 0] = - R_lon[0, 1]

    R_full = np.matmul(R_lon, R_lat)

    dot_prod = np.sum(np.matmul(R_full, K_inv).reshape(1, 3, 3) * x_y_hom.reshape(-1, 1, 3), axis=2)

    sphere_points = dot_prod / np.linalg.norm(dot_prod, axis=1, keepdims=True)

    lat = np.degrees(np.arccos(sphere_points[:, 2]))
    lon = np.degrees(np.arctan2(sphere_points[:, 0], sphere_points[:, 1]))

    lat_lon = np.column_stack([lat, lon])
    lat_lon = np.mod(lat_lon, np.array([180.0, 360.0]))

    org_img_y_x = lat_lon / np.array([180.0, 360.0]) * np.array(org_height_width)
    org_img_y_x = np.clip(org_img_y_x, 0.0, np.array(org_height_width).reshape(1, 2) - 1.0).astype(int)
    org_img_y_x = org_img_y_x.astype(int)

    if len(origin_image.shape) == 3:
        result_image[x_y_hom[:, 1].astype(int), x_y_hom[:, 0].astype(int), :] = origin_image[org_img_y_x[:, 0],
                                                                                org_img_y_x[:, 1], :]
    else:
        result_image[x_y_hom[:, 1].astype(int), x_y_hom[:, 0].astype(int)] = origin_image[org_img_y_x[:, 0],
                                                                                          org_img_y_x[:, 1]]
    return result_image.astype(float), org_img_y_x


def get_pano_no(pano_no, undersample=3):
    '''Helper function to load the panorama oc scene pano_no and downsample it by a factor of undersample.'''
    path = os.path.join(settings.IMG_PATH, 'cubemap_%04d.png' % pano_no)
    pano = mplimg.imread(path)
    pano = pano[::undersample, ::undersample, :3]
    return pano


def extract_vid_frames(vid_path, target_dir, fps=15, print_output=False):
    '''Extract frames from a video, wrapping ffmpeg.'''
    vid_name = os.path.basename(vid_path)
    vid_name_no_ext = os.path.splitext(vid_name)[0]

    ffmpeg_template = 'ffmpeg -i {} -vf fps={} {}'

    ffmpeg_cmd = ffmpeg_template.format(vid_path, fps, os.path.join(target_dir, '%06d.png'))
    args = shlex.split(ffmpeg_cmd)

    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()

    if print_output:
        pprint(output)
        pprint(err)


def interpolate_nan_rows(array, bad_rows_bool):
    '''Interpolates nan rows in an array.'''
    if True not in bad_rows_bool.astype(bool): return array

    good_rows = np.where(np.logical_not(bad_rows_bool))[0]

    # Since extrapolation is a bad idea, we identify the low-confidence indices that are outside the interpolatable range.
    low_non_nan, high_non_nan = np.amin(good_rows), np.amax(good_rows)
    interp_idcs = np.copy(bad_rows_bool)
    interp_idcs[high_non_nan:] = False
    interp_idcs[:low_non_nan] = False

    extra_idcs = np.logical_and(bad_rows_bool,
                                np.logical_or(np.arange(len(bad_rows_bool)) <= low_non_nan,
                                              np.arange(len(bad_rows_bool)) >= high_non_nan))

    interp_func = scipy.interpolate.interp1d(good_rows,
                                             np.take(array, good_rows, axis=0),
                                             kind='linear',
                                             axis=0)
    array[interp_idcs] = interp_func(np.where(interp_idcs)[0])
    # The indices outside the interpolatable range are set to the mean of the series.
    array[extra_idcs] = np.mean(array[good_rows], axis=0)

    return array


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
      #  print("Ratio of {} with the pixel intensity lower than {}".format(ratio, (int_lvl + 1)))
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
        gaze_counts[coord[0], coord[1]] += 1.0

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

if __name__ == '__main__':
    all_files = sorted(glob(os.path.join(settings.DATASET_PATH_VR, '*.pck')))
    #IOC_func(all_files)
    entropy_func()
