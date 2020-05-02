import os
import cv2
import numpy as np
from PIL import Image
from time import sleep
import sys

pixel_shift = 1920
interval_shift = 6

def bar_show(nFr):

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write('--------------------------------------------------------------------' + '\n')
        sys.stdout.write('360VSOD-10K  ||  ' + str(nFr) + ' / 10000  ||  ' +
                         "[%-20s] %d%%" % ('=' * int(nFr / 10000 * 20), nFr / 10000 * 100) + '\n')
        sys.stdout.write('--------------------------------------------------------------------' + '\n')
        sys.stdout.flush()
        sleep(0.25)


class ProcessingTool():
    def __init__(self):
        self.ori_path = os.getcwd() + '/frm_ori/'
        self.tar_path = os.getcwd() + '/frm_shift/'
        self.sft_path = os.getcwd() + '/msk_shift/'
        self.fin_path = os.getcwd() + '/msk_fin/'
        self.pts_path = os.getcwd() + '/parts/'
        self.src_path = os.getcwd() + '/source_videos/'


    def split2whole(self):
        ori_list = os.listdir(self.ori_path)
        count = 0
        for item in ori_list:
            int_item = int(item[:-4])
            if int_item % interval_shift == 0:
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
        vid_name = '_-6QUCaLvQ_3I'
        vid_H = 2048
        vid_W = 3840
        vid_fps = 5

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
        ists_path = os.getcwd() + '/_-Uy5LTocHmoA_2/'
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

    def getKeyFrm(self):
        rawFrm_path = os.getcwd() + '/_-Uy5LTocHmoA_2/'
        rawFrm_list = os.listdir(rawFrm_path)
        rawFrm_list.sort(key=lambda x: x[:-4])

        for frm in rawFrm_list:
            frm_list = frm.split('_')
            frm_list_2 = frm_list[1].split('.')
            frm_idx = int(frm_list_2[0])
            if frm_idx % 6 == 0:
                KeyFrm_path = rawFrm_path + frm
                new_path = os.getcwd() + '/' + frm
                os.rename(KeyFrm_path, new_path)

        print('Done !')

    def numFrm(self):
        file_path = os.getcwd() + '/mask_instance/'
        file_list = os.listdir(file_path)

        numFrm = 0
        count = 0
        for file in file_list:
            file_cur_path = file_path + file
            file_cur_list = os.listdir(file_cur_path)
            numFrm += len(file_cur_list)
            count += 1
            print(" {} done. ".format(count))

        return numFrm

    def frmStt(self):
        seq_list = os.listdir(self.src_path)

        f = open(os.getcwd() + '/360VSOD_stt.txt', 'w')
        frames_num = []
        duration_num = []
        count = 0
        for seq in seq_list:
            if seq.endswith('.mp4'):
                seq_path = os.path.join(os.path.abspath(self.src_path), seq)
                cap = cv2.VideoCapture(seq_path)
                frames_num.append(int(cap.get(7)))
                duration_num.append(int(cap.get(7) / (cap.get(5))))
                line = str(count + 1) + '    ' + seq[:-4] + '    ' + str(frames_num[count]) + '    ' + \
                       str(duration_num[count]) + '    ' + '\n'
                f.write(line)
                count += 1
                print(" {} videos processed".format(count))

        total_frames = np.sum(frames_num)
        total_duration = np.sum(duration_num)
        f.write('There are ' + str(total_frames) + ' frames.' + '\n')
        f.write('The total duration is ' + str(total_duration) + ' s.' + '\n')
        f.write('The average duration is ' + str(total_duration / count) + ' s.')
        f.close()

        return total_frames

    def srcApl(self):
        seq_list = os.listdir('mask_instance')
        src_list = os.listdir('source')

        count = 0
        for seq in seq_list:
            seq_id = seq + '.mp4'
            if seq_id in src_list:
                src_path = os.getcwd() + '/source/' + seq_id
                new_path = os.getcwd() + '/source_videos/' + seq_id
                os.rename(src_path, new_path)
                count += 1
                print(" {} videos transfered.".format(count))


if __name__ == '__main__':
    PT = ProcessingTool()
    #PT.split2whole()
    #PT.shiftRecover()
    #PT.frm2vid()
    #PT.ist2obj()
    #PT.ist_merge()
    #PT.getKeyFrm()
    #print('There are: ' + str(PT.numFrm()) + ' key frames.')
    bar_show(PT.numFrm())
    PT.frmStt()