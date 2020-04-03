import os
import cv2
import numpy as np
from PIL import Image

pixel_shift = 512

class ProcessingTool():
    def __init__(self):
        self.ori_path = os.getcwd() + '/frm_ori/'
        self.tar_path = os.getcwd() + '/frm_shift/'
        self.sft_path = os.getcwd() + '/msk_shift/'
        self.fin_path = os.getcwd() + '/msk_fin/'
        self.pts_path = os.getcwd() + '/parts/'

    def split2whole(self):
        ori_list = os.listdir(self.ori_path)
        count = 0
        for item in ori_list:
            item_path = self.ori_path + item
            img = cv2.imread(item_path)
            img_shift = img.copy()
            img_shift = np.roll(img_shift, -1 * pixel_shift, axis=1)
            cv2.imwrite(self.tar_path + item, img_shift)
            count += 1
            print(" {} frames have been processed.".format(count))
        print(' Done ! ')

    def shiftRecover(self):
        sft_list = os.listdir(self.sft_path)
        count = 0
        for item in sft_list:
            item_path = self.sft_path + item
            msk = cv2.imread(item_path)
            msk_shift = msk.copy()
            msk_shift = np.roll(msk_shift, pixel_shift, axis=1)
            msk_name = 'frame_' + format(item[:-4], '0>6s') + '.png'
            cv2.imwrite(self.fin_path + msk_name, msk_shift)
            count += 1
            print(" {} masks have been processed.".format(count))
        print(' Done ! ')

    def frm2vid(self):
        vid_name = '_-Uy5LTocHmoA_1'
        vid_H = 2048
        vid_W = 3840
        vid_fps = 30

        frm_list = os.listdir(os.getcwd() + '/' + vid_name + '/')
        frm_list.sort(key=lambda x: x[:-4])
        vid = cv2.VideoWriter(os.getcwd() + '/' + vid_name + '.avi', 0, vid_fps, (vid_W, vid_H))

        count = 0
        for frm in frm_list:
            frm_path = os.getcwd() + '/' + vid_name + '/' + frm
            vid.write(cv2.imread(frm_path))
            count += 1
            print(" {} frames added.".format(count))
        vid.release()
        print(' Done !')

    def ist2obj(self):
        ists_path = os.getcwd() + '/_-Uy5LTocHmoA_1/'
        objs_path = os.getcwd() + '/obj/'

        count = 1
        for item in os.listdir(ists_path):
            ist_path = os.path.join(os.path.abspath(ists_path), item)
            obj_path = os.path.join(os.path.abspath(objs_path), item)
            ist = cv2.imread(ist_path)
            ist = cv2.cvtColor(ist, cv2.COLOR_RGB2GRAY)
            ret, obj = cv2.threshold(ist, 0, 255, cv2.THRESH_BINARY)
            cv2.imwrite(obj_path, obj)
            print(" {} frames processed".format(count))
            count += 1
        print(' Done !')

    def ist_merge(self):
        main_list = os.listdir(self.pts_path + '/part1/')
        main_list.sort(key=lambda x: x[:-4])
        sub_list_1 = os.listdir(self.pts_path + '/part2/')
        sub_list_1.sort(key=lambda x: x[:-4])
        sub_list_2 = os.listdir(self.pts_path + '/part3/')
        sub_list_2.sort(key=lambda x: x[:-4])

        count = 0
        for item in main_list:
            msk1 = cv2.imread(self.pts_path + '/part1/' + item)
            if item in sub_list_1:
                msk2 = cv2.imread(self.pts_path + '/part2/' + item)
                msk1 = msk1 + msk2
            if item in sub_list_2:
                msk3 = cv2.imread(self.pts_path + '/part3/' + item)
                msk1 = msk1 + msk3

            cv2.imwrite(item, msk1)
            count += 1
            print(" {} frames processed".format(count))



if __name__ == '__main__':
    PT = ProcessingTool()
    #PT.split2whole()
    #PT.shiftRecover()
    #PT.frm2vid()
    #PT.ist2obj()
    PT.ist_merge()