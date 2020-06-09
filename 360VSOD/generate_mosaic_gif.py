"""
Generates an animated gif of the Holopix50k mosaic, where images are randomly replaced.

Example usage:
 - python3 generate_mosaic_gif.py holopix_logo_icon.png ~/Holopix50k/train/left/
"""

import os, random
import cv2
import numpy as np
import argparse
import tqdm

import imageio

from joblib import Memory


def get_average_color(image):
    # Note: resizing here is faster. This gets us an approximation of the average color.
    image = cv2.resize(image, (20, 20))
    return tuple(np.average(image.reshape(image.shape[0] * image.shape[1], image.shape[2]), axis=0))


memory = Memory(location='/tmp', verbose=0)


@memory.cache
def get_average_color_for_file(filename: str):
    image = cv2.imread(filename)
    return get_average_color(image)


@memory.cache
def get_average_color_list(filepaths):
    average_color = np.zeros((len(filepaths), 3), dtype=np.uint8)
    for i in tqdm.trange(len(filepaths)):
        average_color[i, :] = get_average_color_for_file(filepaths[i])
    return average_color


def get_best_match_index(target_color, color_list):
    """
    Given a target color and a list of colors, returns the index of the list that most closely matches.
    """
    min_index = 0
    min_dist = float("inf")
    target_color_expanded = [target_color] * len(color_list)
    diff = color_list - target_color_expanded
    diff2 = abs(diff[:, 0]) + abs(diff[:, 1]) + abs(diff[:, 2])
    for i in range(len(color_list)):
        if diff2[i] < min_dist:
            min_dist = diff2[i]
            min_index = i
    return min_index


def get_mosaic(target_image, input_image_dir, rows: int, columns: int):
    input_image_filenames = os.listdir(input_image_dir)
    input_image_filepaths = [os.path.abspath(os.path.join(input_image_dir, input_image_filenames[i])) for i in range(len(input_image_filenames))]

    w = 12
    h = 12
    W = w * columns
    H = h * rows
    target_image = cv2.resize(target_image, (W, H), cv2.INTER_AREA)

    mosaic_image = np.zeros((H, W, 3), dtype=np.uint8)

    print('calculating color averages...')
    input_images_averages = get_average_color_list(input_image_filepaths)

    # Iterate over the data in random order.
    coords = []
    for x in range(rows):
        for y in range(columns):
            coords.append((x, y))
    random.shuffle(coords)

    print('Populating the base image...')
    for (i, j) in tqdm.tqdm(coords):
        target_subimage = target_image[i*h:(i+1)*h, j*w:(j+1)*w, :]

        target_average = get_average_color(target_subimage)

        match_index = get_best_match_index(target_average, input_images_averages)
        source_image = cv2.imread(input_image_filepaths[match_index])

        # Blur before downsampling to reduce aliasing
        source_image = cv2.resize(source_image, (w*5, h*5), cv2.INTER_AREA)
        source_image = cv2.GaussianBlur(source_image, (5, 5), 0)
        source_image = cv2.resize(source_image, (w, h), cv2.INTER_AREA)

        mosaic_image[i*w:(i+1)*w, j*h:(j+1)*h] = source_image

        input_images_averages = np.delete(input_images_averages, obj=match_index, axis=0)
        del input_image_filepaths[match_index]

        cv2.imshow('out', mosaic_image)
        cv2.waitKey(1)

    print('Adding frames to the gif...')
    gif_frames = []
    for iteration, (i, j) in enumerate(tqdm.tqdm(coords[:150])):
        target_subimage = target_image[i*h:(i+1)*h, j*w:(j+1)*w, :]

        target_average = get_average_color(target_subimage)

        match_index = get_best_match_index(target_average, input_images_averages)
        source_image = cv2.imread(input_image_filepaths[match_index])

        # Blur before downsampling to reduce aliasing
        source_image = cv2.resize(source_image, (w*5, h*5), cv2.INTER_AREA)
        source_image = cv2.GaussianBlur(source_image, (5, 5), 0)
        source_image = cv2.resize(source_image, (w, h), cv2.INTER_AREA)

        mosaic_image[i*w:(i+1)*w, j*h:(j+1)*h] = source_image

        input_images_averages = np.delete(input_images_averages, obj=match_index, axis=0)
        del input_image_filepaths[match_index]

        cv2.imshow('out', mosaic_image)
        cv2.waitKey(1)

        if iteration % 6 == 0:
            gif_frame = cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2BGR)
            gif_frames.append(gif_frame)

    print(len(gif_frames))

    # Append a reversed animation, so that the loop appears to be seamless.
    gif_frames = gif_frames + gif_frames[::-1][1:-1]

    print(len(gif_frames))

    print('Encoding gif...')
    imageio.mimsave('out.gif', gif_frames)
    print('Saved to out.gif')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Target image")
    parser.add_argument("data_dir", help="Dataset directory")
    args = parser.parse_args()

    target_image = cv2.imread(args.target)

    print('starting photomosaic creation...')

    # Reduce this factor to speed up development
    scale_factor = 1.0

    get_mosaic(target_image, args.data_dir, rows=int(25*scale_factor), columns=int(45*scale_factor))
