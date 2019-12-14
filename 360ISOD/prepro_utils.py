import re, os, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

import settings


def file_rename(mode='left'):
    if mode == 'left':
        pathI = settings.L_PATH_RAW
        pathO = settings.L_PATH_TGT
    elif mode == 'right':
        pathI = settings.R_PATH_RAW
        pathO = settings.R_PATH_TGT
    else:
        print('No processing; Please check your input parameters.')

    filelist = os.listdir(pathI)
    filelist.sort(key=lambda x: x[:-4])

    count = 1
    for item in filelist:
        if item.endswith('.txt'):
            src = os.path.join(os.path.abspath(pathI), item)
            dst = os.path.join(os.path.abspath(pathO), format(str(count), '0>3s') + '.txt')
            try:
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
            except:
                continue
        print(" {} images processed".format(count))
        count += 1

    print('Naming process done !')

class common_prepro():
    def __init__(self):
        self.path_L_raw = os.getcwd() + '/scanpath_nante/L/'
        self.path_R_raw = os.getcwd() + '/scanpath_nante/R/'
        self.path_L_prd = os.getcwd() + '/fixations/L/'
        self.path_R_prd = os.getcwd() + '/fixations/R/'

    def readbin(self):
        filelist = os.listdir(self.path)

        #  Possible float precision of bin files
        dtypes = {16: np.float16,
                  32: np.float32,
                  64: np.float64}

        count = 1
        for item in filelist:
            get_file_info = re.compile("(\w+_\d{1,2})_(\d+)x(\d+)_(\d+)b")
            info = get_file_info.findall(item.split(os.sep)[-1])

            name, width, height, dtype = info[0]
            width, height, dtype = int(width), int(height), int(dtype)

            #  Open file to read as binary
            img_path = os.path.join(os.path.abspath(self.path), item)
            with open(img_path, "rb") as f:
                #  Read from file the content of one frame
                data = np.fromfile(f, count=width * height, dtype=dtypes[dtype])

                # threshold the saliencies
                maxSal = np.max(data)
                for i in range(width*height):
                    if data[i] <= 0.2 * maxSal: # 0.2 is a empirical value
                        data[i] = 0

                # Reshape flattened data to 2D image
                data = data.reshape([height, width])
                data = cv2.normalize(data, None, alpha=0, beta=255,
                                     norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_8UC1)

                # save the image
                heatmap = cv2.applyColorMap(data, cv2.COLORMAP_HOT)
                cv2.imwrite(name + '.png', heatmap)
                print(" {} images processed".format(count))
                count += 1

    def resize(self):
        filelist = os.listdir(self.path)

        count = 1
        for item in filelist:
            img_path = os.path.join(os.path.abspath(self.path), item)
            if item.endswith('.png'):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (512, 256))
                cv2.imwrite(item[:-3] + 'png', img)
                print(" {} images processed".format(count))
                count += 1

    def imgfuse(self):
        filelist = os.listdir(settings.ERP_PATH)
        filelist.sort(key=lambda x: x[:-4])

        count = 1
        for item in filelist:
            img1_path = os.path.join(os.path.abspath(settings.ERP_PATH), item)
            img1 = cv2.imread(img1_path)
            img2_path = os.path.join(os.path.abspath(settings.HEAT_PATH), item)
            img2 = cv2.imread(img2_path)

            img3 = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
            cv2.imwrite(item, img3)
            print(" {} images processed".format(count))
            count += 1

    def VideoToImg(self):
        filelist = os.listdir(self.pathv)

        count = 1
        for item in filelist:
            if item.endswith('.mp4'):
                video_path = os.path.join(os.path.abspath(self.pathv), item)
                frame_path = os.path.join(os.path.abspath(self.pathf), item)
                cap = cv2.VideoCapture(video_path)
                frames_num = int(cap.get(7))
                countF = 1
                for i in range(frames_num):
                    ret, frame = cap.read()
                    frame = cv2.resize(frame, (3840, 2048))
                    cv2.imwrite(frame_path[:-4] + '_' +
                                format(str(countF), '0>4s') + '.png',
                                frame)
                    print(" {} frames processed".format(countF))
                    countF += 1
                print(" {} videos processed".format(count))
                count += 1

    def video_bin(self, scale_frame):
        filelist = os.listdir(self.path)

        #  Possible float precision of bin files
        dtypes = {16: np.float16,
                  32: np.float32,
                  64: np.float64}

        count = 1
        for item in filelist:
            get_file_info = re.compile("(\d+_\w+)_(\d+)x(\d+)x(\d+)_(\d+)b")
            info = get_file_info.findall(item.split(os.sep)[-1])

            name, width, height, Frames, dtype = info[0]
            width, height, Frames, dtype = int(width), int(height), \
                                           int(Frames), int(dtype)

            #  Open file to read as binary
            img_path = os.path.join(os.path.abspath(self.path), item)
            with open(img_path, "rb") as f:
                for Nframe in range(Frames):
                    if (Nframe + 1) % scale_frame == 0:
                        # Position read pointer right before target frame
                        f.seek(width * height * Nframe * (dtype // 8))

                        #  Read from file the content of one frame
                        data = np.fromfile(f, count=width * height,
                                        dtype=dtypes[dtype])

                        # threshold the saliencies
                        # maxSal = np.max(data)
                        # for i in range(width * height):
                        #     if data[i] <= 0.2 * maxSal:
                        #         data[i] = 0

                        # Reshape flattened data to 2D image
                        data = data.reshape([height, width])
                        data = cv2.normalize(data, None, alpha=0, beta=255,
                                             norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)

                        # save the image
                        heatmap = cv2.applyColorMap(data, cv2.COLORMAP_HOT)
                        cv2.imwrite(name + '_' + format(str(Nframe+1), '0>4s') +
                                    '.png', heatmap)
                        print("{}, frame #{}".format(name, Nframe))
                print(" {} videos processed".format(count))
                count += 1

    def overlay_video(self):
        fl = os.listdir(self.path3)
        fl.sort(key=lambda x: x[:-4])
        video = cv2.VideoWriter(self.pathv, 0, 25, (2048, 1024))

        count = 1
        for item in fl:
            frame_path = os.path.join(os.path.abspath(self.path3), item)
            video.write(cv2.imread(frame_path))
            print("{} wrote".format(count))
            count += 1

        cv2.destroyAllWindows()
        video.release()


if __name__ == '__main__':
   print('waiting...')
   #cpp = common_prepro()
   #cpp.imgfuse()