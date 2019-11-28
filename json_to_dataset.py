import re, os, sys
from os import listdir
import cv2

class file_arrangement():
    def __init__(self):
        self.path = os.getcwd()
        self.path_labels = os.getcwd() + '/labels'
        self.path_overlays = os.getcwd() + '/overlays'
        self.path_txt = os.getcwd() + '/texts'
        self.path_yaml = os.getcwd() + '/yamls'
        self.path_instance = os.getcwd() + '/salient_instance'
        self.path_object = os.getcwd() + '/salient_object'
        self.path_stimuli = os.getcwd() + '/stimuli'

    def arrange_labels(self):
        for item1 in listdir(self.path):
            if item1.endswith('_json'):
                label_path = os.path.join(os.path.abspath(self.path), item1)

                for item0 in listdir(label_path):
                    if item0.endswith('img.png'):
                        src = os.path.join(os.path.abspath(label_path), item0)
                        dst = os.path.join(os.path.abspath(self.path_stimuli),
                                           item1[0:3] + '.png')
                        try:
                            os.rename(src, dst)
                            print('converting %s to %s ...' % (src, dst))
                        except:
                            continue

                for item2 in listdir(label_path):
                    if item2.endswith('label.png'):
                        src = os.path.join(os.path.abspath(label_path), item2)
                        dst = os.path.join(os.path.abspath(self.path_labels),
                                           item1[0:3] + '.png')
                        try:
                            os.rename(src, dst)
                            print('converting %s to %s ...' % (src, dst))
                        except:
                            continue

                for item3 in listdir(label_path):
                    if item3.endswith('viz.png'):
                        src = os.path.join(os.path.abspath(label_path), item3)
                        dst = os.path.join(os.path.abspath(self.path_overlays),
                                           item1[0:3] + '.png')
                        try:
                            os.rename(src, dst)
                            print('converting %s to %s ...' % (src, dst))
                        except:
                            continue

                for item4 in listdir(label_path):
                    if item4.endswith('.txt'):
                        src = os.path.join(os.path.abspath(label_path), item4)
                        dst = os.path.join(os.path.abspath(self.path_txt),
                                           item1[0:3] + '.txt')
                        try:
                            os.rename(src, dst)
                            print('converting %s to %s ...' % (src, dst))
                        except:
                            continue

                for item5 in listdir(label_path):
                    if item5.endswith('.yaml'):
                        src = os.path.join(os.path.abspath(label_path), item5)
                        dst = os.path.join(os.path.abspath(self.path_yaml),
                                           item1[0:3] + '.yaml')
                        try:
                            os.rename(src, dst)
                            print('converting %s to %s ...' % (src, dst))
                        except:
                            continue

    def instance_to_object(self):
        count = 1

        for item in listdir(self.path_instance):
            img_path = os.path.join(os.path.abspath(self.path_instance), item)
            obj_path = os.path.join(os.path.abspath(self.path_object), item)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_h = len(img[:, 0])
            img_w = len(img[0, :])

            for i in range(img_h):
                for j in range(img_w):
                    if img[i, j] > 0:
                        img[i, j] = 255
            cv2.imwrite(obj_path, img)

            print(" {} imgs processed".format(count))
            count += 1

    def rename(self):
        filelist = os.listdir(self.path)
        filelist.sort(key=lambda x: x[:-5])

        count = 1
        for item in filelist:
            if item.endswith('.json'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), format(str(count), '0>3s')+'.json')
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                except:
                    continue
                print(" {} images processed".format(count))
                count += 1

if __name__ == '__main__':
    file_arranger = file_arrangement()
   # file_arranger.rename()
   # file_arranger.arrange_labels()
    file_arranger.instance_to_object()