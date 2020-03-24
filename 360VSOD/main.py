import os
import cv2
import numpy as np


class ProcessingTool():
    def __init__(self):
        self.ori_path = os.getcwd() + '/frm_ori/'
        self.tar_path = os.getcwd() + '/frm_shift/'
        self.sft_path = os.getcwd() + '/msk_shift/'
        self.fin_path = os.getcwd() + '/msk_fin/'

    def split2whole(self):
        ori_list = os.listdir(self.ori_path)
        count = 0
        for item in ori_list:
            item_path = self.ori_path + item
            img = cv2.imread(item_path)
            img_shift = img.copy()
            img_shift = np.roll(img_shift, -1 * 512, axis=1)
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
            msk_shift = np.roll(msk_shift, 512, axis=1)
            msk_name = 'frame_' + format(item[:-4], '0>6s') + '.png'
            cv2.imwrite(self.fin_path + msk_name, msk_shift)
            count += 1
            print(" {} masks have been processed.".format(count))
        print(' Done ! ')


if __name__ == '__main__':
    PT = ProcessingTool()
    #PT.split2whole()
    PT.shiftRecover()