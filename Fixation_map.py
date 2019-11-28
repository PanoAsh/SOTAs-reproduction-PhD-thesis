from data import *
import matplotlib.pyplot as plt

class VideoFixationMap():
    def __init__(self):
        self.path = os.getcwd() + '/dataset'
        self.path_debug = os.getcwd() + '/data_debug'
        self.path_frame = os.getcwd() + '/frames'
        self.video = os.getcwd() + '/overlay_video.avi'

    def video_overlay(self):
        fl = os.listdir(self.path_debug)

        count = 1
        for item in fl:
            if item.endswith('.jpg'):
                item_map = item[:-4] + '_gt.npy'
                img_path = os.path.join(os.path.abspath(self.path_debug), item)
                map_path = os.path.join(os.path.abspath(self.path_debug), item_map)

                img = cv2.imread(img_path)
                height = img.shape[0]
                width = img.shape[1]

                map_load = np.load(map_path)
                map = map_load[0]
                map = cv2.resize(map, (width, height))
                map = cv2.normalize(map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                map = cv2.applyColorMap(map, cv2.COLORMAP_HOT)

              #  show_img(img, map)
                overlay = cv2.addWeighted(img, 0.5, map, 0.5, 0)
               # show_overlay(overlay)
                cv2.imwrite(item[:-4] + '_overlay' + '.png', overlay)

                print("{} have been processed".format(count))
                count += 1

    def overlay_video(self):
        fl = os.listdir(self.path_frame)
        fl.sort(key=lambda x: x[:-4])
        video = cv2.VideoWriter(self.video, 0, 20, (3840, 1920))

        count = 1
        for item in fl:
            frame_path = os.path.join(os.path.abspath(self.path_frame), item)
            video.write(cv2.imread(frame_path))
            print("{} wrote".format(count))
            count += 1

        cv2.destroyAllWindows()
        video.release()

    def fixation_reproduce(self):
        para_default = 1 # just default, the real value will be equal to the corresponding image
        vr_saliency = VRSaliency(root=self.path, frame_h=para_default, frame_w=para_default)
       # vr_saliency.cache_map()
        vr_saliency.fixation_self()

def show_img(debug_img, debug_map):
        plt.subplot(1, 2, 1)
        plt.imshow(debug_img)
        plt.subplot(1, 2, 2)
        plt.imshow(debug_map, cmap='gray')
        plt.show()
        print()

def show_overlay(overlay):
    plt.figure()
    plt.imshow(overlay)
    plt.show()
    print()


if __name__ == '__main__':
    s_video = VideoFixationMap()
   # s_video.video_overlay()
    s_video.overlay_video()

  #  vr_obj = VideoFixationMap()
   # vr_obj.fixation_reproduce()
