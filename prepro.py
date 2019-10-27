import re, os, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

threshold = 0.5
multi = 3 / threshold / 2
width_st = 2048
height_st = 1024

class preprocessing():
    def __init__(self):
        self.path = os.getcwd() + '/bins'
        self.path1 = os.getcwd() + '/'
        self.path2 = os.getcwd() + '/new'
        self.pathv = os.getcwd() + '/videos'

    def thresholding(self):
        filelist = os.listdir(self.path)

        count = 1
        for item in filelist:
            if item.endswith('.png'):
                img_path = os.path.join(os.path.abspath(self.path), item)
                img = cv2.imread(img_path) # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB
                H = img.shape[0]
                W = img.shape[1]

                # threshold
                for i in range(H):
                    for j in range(W):
                        if img[i, j, 0] <= 255 * threshold: # red
                           img[i, j, 0] = 0
                        img[i, j, 1] = 0 # green
                        if img[i, j, 2] >= 255 * threshold: # blue
                           img[i, j, 2] = 0

                # debug_show(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                for m in range(H): # lighting
                    for n in range(W):
                        img[m, n] *= multi
                        if img[m, n] >= 255:
                            img[m, n] = 255
                cv2.imwrite(item, img)
                print(" {} images processed".format(count))
                count += 1

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
                    if data[i] <= 0.2 * maxSal:
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
            if item.endswith('.jpg'):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (width_st, height_st))
                cv2.imwrite(item[:-3] + 'png', img)
                print(" {} images processed".format(count))
                count += 1

    def rename(self):
        filelist = os.listdir(self.path)

        count = 1
        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path1), item)
                dst = os.path.join(os.path.abspath(self.path2),
                                   '01_00' + item[2:4] + '.png')
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                except:
                    continue
            print(" {} images processed".format(count))
            count += 1

    def imgfuse(self):
        filelist = os.listdir(self.path)

        count = 1
        for item in filelist:
            img_path = os.path.join(os.path.abspath(self.path), item)
            img = cv2.imread(img_path)
            img1_path = os.path.join(os.path.abspath(self.path1), item)
            img1 = cv2.imread(img1_path)
            img2_path = os.path.join(os.path.abspath(self.path2), item)

            img2 = cv2.addWeighted(img, 0.6, img1, 0.4, 0)
            cv2.imwrite(img2_path, img2)
            print(" {} images processed".format(count))
            count += 1

    def VideoToImg(self):
        filelist = os.listdir(self.pathv)

        count = 1
        for item in filelist:
            if item.endswith('.mp4'):
                video_path = os.path.join(os.path.abspath(self.pathv), item)
                frame_path = os.path.join(os.path.abspath(self.path2), item)
                cap = cv2.VideoCapture(video_path)
                frames_num = int(cap.get(7))
                countF = 1
                for i in range(frames_num):
                    ret, frame = cap.read()
                    frame = cv2.resize(frame, (width_st, height_st))
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
                        # heatmap = cv2.applyColorMap(data, cv2.COLORMAP_HOT)
                        cv2.imwrite(name + '_' + "{}".format(Nframe + 1) +
                                    '.png', data)
                        print("{}, frame #{}".format(name, Nframe))
                print(" {} videos processed".format(count))
                count += 1

def debug_show(img):
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.subplot(1, 4, 2)
    plt.imshow(img[:, :, 0])
    plt.subplot(1, 4, 3)
    plt.imshow(img[:, :, 1])
    plt.subplot(1, 4, 4)
    plt.imshow(img[:, :, 2])
    plt.show()

if __name__ == '__main__':
    pp = preprocessing()
    pp.video_bin(10) # get the iamge every 10 frames