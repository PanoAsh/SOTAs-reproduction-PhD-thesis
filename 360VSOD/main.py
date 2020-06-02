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
            #int_item = int(item[:-4])
            #if int_item % interval_shift == 0:
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
            item_list = item.split('_')
            item_path = self.sft_path + item
            msk = cv2.imread(item_path)
            msk_shift = msk.copy()
            msk_shift = np.roll(msk_shift, pixel_shift, axis=1)
            #msk_name = 'frame_' + format(item[:-4], '0>6s') + '.png'
            msk_name = 'frame_' + item_list[3]
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
        rawFrm_path = os.getcwd() + '/_-ZuXCMpVR24I/'
        rawFrm_list = os.listdir(rawFrm_path)
        rawFrm_list.sort(key=lambda x: x[:-4])

        for frm in rawFrm_list:
            frm_list = frm.split('_')
            frm_list_2 = frm_list[2].split('.')
            frm_idx = int(frm_list_2[0])
            if frm_idx % 6 == 0:
                if frm_idx % 12 != 0:
                    KeyFrm_path = rawFrm_path + frm
                    new_path = os.getcwd() + '/frames_part2/' + 'frame_' + frm[-10:]
                    os.rename(KeyFrm_path, new_path)

        print('Done !')

    def numFrm(self):
        file_path = '/home/yzhang1/PythonProjects/360vSOD/data/mask_instance/'
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
        seq_list = os.listdir('/home/yzhang1/PythonProjects/360vSOD/data/source_videos/')
        #frm_list = os.listdir('/home/yzhang1/PythonProjects/360vSOD/data/frames/')
        #msk_list = os.listdir('/home/yzhang1/PythonProjects/360vSOD/data/mask_instance/')

        f = open(os.getcwd() + '/360VSOD_stt.txt', 'w')
        frames_num = []
        duration_num = []
        fps_num = []
        count = 0
        for seq in seq_list:
            if seq.endswith('.mp4'):
                seq_path = os.path.join('/home/yzhang1/PythonProjects/360vSOD/data/source_videos/', seq)
                cap = cv2.VideoCapture(seq_path)
                msk_path = os.path.join('/home/yzhang1/PythonProjects/360vSOD/data/mask_instance/', seq[:-4])
                frm_path = os.path.join('/home/yzhang1/PythonProjects/360vSOD/data/frames/',  seq[:-4])
                if len(os.listdir(msk_path)) != len(os.listdir(frm_path)):
                    print('Please Check !!!')
                    break
                frames_num.append(len(os.listdir(msk_path)))
                fps_num.append(cap.get(5))
                duration_num.append((len(os.listdir(msk_path)) - 1) * 6 / (int(cap.get(5)) + 1))
                line = format(str(count + 1), '0>2s') + '    ' + 'ID: ' + seq[:-4] + '    ' + str(frames_num[count]) \
                       + ' key frames' + '    ' + str(duration_num[count]) + ' s' + '    ' + str(fps_num[count]) + \
                       ' fps' + '\n'
                f.write(line)
                count += 1
                print(" {} videos processed".format(count))

        total_frames = np.sum(frames_num)
        total_duration = np.sum(duration_num)
        f.write('Key frames with instance-level annotations: ' + str(total_frames) + '\n')
        f.write('Total duration: ' + str(total_duration) + ' s' + '\n')
        f.write('Average duration: ' + str(total_duration / count) + ' s')
        f.close()

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

    def demoMsk(self):
        seq_list = os.listdir('/home/yzhang1/PythonProjects/360vSOD/data/mask_instance/')
        demo_w = 3840
        demo_h = 1920
        frmShow = 10
        vid_oly = cv2.VideoWriter(os.getcwd() + '/demo_instance_overlay.avi', 0, 3, (demo_w, demo_h))

        count = 1
        for seq in seq_list:
            msk_list = os.listdir('/home/yzhang1/PythonProjects/360vSOD/data/mask_instance/' + seq)
            msk_list.sort(key=lambda x: x[:-4])
            demo_itv = int(len(msk_list) / frmShow)

            for idx in range(frmShow):
                msk_path = os.path.join('/home/yzhang1/PythonProjects/360vSOD/data/mask_instance/', seq,
                                        msk_list[idx * demo_itv])
                msk = cv2.imread(msk_path)
                frm_id = msk_list[idx * demo_itv].split('_')[1]
                frm_path = os.path.join('/home/yzhang1/PythonProjects/360vSOD/data/frames/', seq, (seq + '_' + frm_id))
                frm = cv2.imread(frm_path)
                oly = cv2.addWeighted(frm, 0.8, msk, 1, 0)
                oly = cv2.resize(oly, (demo_w, demo_h))
                vid_oly.write(oly)
            print("{} videos processed.".format(count))
            count += 1

        vid_oly.release()

    def seq2frm(self):
        seq_list = os.listdir(self.src_path)

        count = 1
        for seq in seq_list:
            if seq.endswith('.mp4'):
                seq_path = os.path.join(os.path.abspath(self.src_path), seq)
                new_path = os.path.join(os.path.abspath('frames'), seq[:-4])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                frm_path = os.path.join(os.path.abspath(new_path), seq)
                cap = cv2.VideoCapture(seq_path)
                frames_num = int(cap.get(7))
                countF = 0
                for i in range(frames_num):
                    ret, frame = cap.read()
                    # frame = cv2.resize(frame, (Width, Height))
                    if countF % 6 == 0:
                        cv2.imwrite(frm_path[:-4] + '_' + format(str(countF), '0>6s') + '.png', frame)
                        print(" {} frames are extracted.".format(countF))
                    countF += 1
                print(" {} videos processed".format(count))
                count += 1

    def mskRename(self):
        msk_list = os.listdir(os.getcwd() + '/_-gSueCRQO_5g/')
        msk_list.sort(key=lambda x: x[:-4])

        for msk in msk_list:
            old_path = os.getcwd() + '/_-gSueCRQO_5g/' + msk
            msk_list = msk.split('_')
            msk_idx = int(msk_list[3][:-4])
            new_path = os.getcwd() + '/msk_may/' + 'frame_' + format(str(msk_idx), '0>6s') + '.png'
            os.rename(old_path, new_path)

        print('done !')

    def GTResize(self):
        src_path = os.getcwd() + '/_-MFVmxoXgeNQ/'
        tgt_path = os.getcwd() + '/resized/'
        src_list = os.listdir(src_path)
        src_list.sort(key=lambda x: x[:-4])

        count = 1
        for frm in src_list:
            frm_path = src_path + frm
            img = cv2.imread(frm_path)
            img = cv2.resize(img, (3840, 1920))
            #ret, img = cv2.threshold(img, 0, 128, cv2.THRESH_BINARY)
            new_path = tgt_path + frm
            cv2.imwrite(new_path, img)
            print(" {} frames processed.".format(count))
            count += 1

    def mskRGB(self):
        rgb_1 = [128, 0, 0]
        rgb_2 = [0, 128, 0]
        rgb_3 = [128, 128, 0]
        rgb_4 = [0, 0, 128]

        listOri = os.listdir(os.getcwd() + '/listori/')
        listOri.sort(key=lambda x: x[:-4])
        count = 1
        for msk_idx in listOri:
            msk_path = os.getcwd() + '/listori/' + msk_idx
            msk = cv2.imread(msk_path)
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
            mskH, mskW, RGB = msk.shape
            for i in range(mskH):
                for j in range(mskW):
                    if msk[i, j, :].tolist() == rgb_2:
                        msk[i, j, 0] = 128
                    elif msk[i, j, :].tolist() == rgb_3:
                        msk[i, j, 1] = 0
                    elif msk[i, j, :].tolist() == rgb_4:
                        msk[i, j, 1] = 128
                        msk[i, j, 2] = 0

            listFin = os.getcwd() + '/listfin/'
            msk_nPath = listFin + msk_idx
            msk = cv2.cvtColor(msk, cv2.COLOR_RGB2BGR)
            cv2.imwrite(msk_nPath, msk)
            print(" {} frames processed.".format(count))
            count += 1

    def mskEdit(self):
        rgb_1 = [128, 0, 0]
        rgb_2 = [0, 128, 0]
        rgb_3 = [128, 128, 0]
        rgb_4 = [0, 0, 128]

        oriPath = os.getcwd() + '/_-Bvu9m__ZX60/'
        mskList = os.listdir(oriPath)
        mskList.sort(key=lambda x: x[:-4])
        count = 1
        for msk in mskList:
            idx = int(msk.split('_')[1][:-4])
            if idx % 12 == 0:
                mskPath = oriPath + msk
                mskImg = cv2.imread(mskPath)
                mskImg = cv2.cvtColor(mskImg, cv2.COLOR_BGR2RGB)
                mskH, mskW, RGB = mskImg.shape
                for i in range(mskH):
                    for j in range(mskW):
                        if mskImg[i, j, :].tolist() == rgb_4:
                            mskImg[i, j, :] = 0
                mskPath_next = oriPath + 'frame_' + format(str(idx + 6), '0>6s') + '.png'
                mskImgNext = cv2.imread(mskPath_next)
                mskImgNext = cv2.cvtColor(mskImgNext, cv2.COLOR_BGR2RGB)
                for i in range(mskH):
                    for j in range(mskW):
                        if mskImgNext[i, j, :].tolist() == rgb_4:
                            mskImg[i, j, 2] = 128
                mskPath_new = os.getcwd() + '/edit/' + msk
                mskImg = cv2.cvtColor(mskImg, cv2.COLOR_RGB2BGR)
                cv2.imwrite(mskPath_new, mskImg)
                print(" {} key frames processed.".format(2 * count))
                count += 1

    def objStt(self):
        rgb_1 = (128, 0, 0)
        rgb_2 = (0, 128, 0)
        rgb_3 = (128, 128, 0)
        rgb_4 = (0, 0, 128)
        rgb_5 = (128, 0, 128)
        rgb_6 = (0, 128, 128)
        rgb_7 = (128, 128, 128)
        rgb_8 = (64, 0, 0)
        rgb_9 = (192, 0, 0)

        f = open(os.getcwd() + '/360VSOD_obj_stt.txt', 'w')

        seq_list = os.listdir('/home/yzhang1/PythonProjects/360vSOD/data/mask_instance/')
        seq_count = 0
        num_obj_sum = 0
        for seq in seq_list:
            msk_list = os.listdir(os.path.join('/home/yzhang1/PythonProjects/360vSOD/data/mask_instance/', seq))
            msk_list.sort(key=lambda x: x[:-4])
            msk_count = 0
            num_obj = 0
            for msk_idx in msk_list:
                msk_path = os.path.join('/home/yzhang1/PythonProjects/360vSOD/data/mask_instance/', seq, msk_idx)
                msk = Image.open(msk_path)
                obj_count = 0
                obj_bool = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                for pix in msk.getdata():
                    if pix == (0, 0, 0):
                        continue
                    elif pix == rgb_1:
                        if obj_bool[0] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[0] = 1
                    elif pix == rgb_2:
                        if obj_bool[1] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[1] = 1
                    elif pix == rgb_3:
                        if obj_bool[2] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[2] = 1
                    elif pix == rgb_4:
                        if obj_bool[3] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[3] = 1
                    elif pix == rgb_5:
                        if obj_bool[4] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[4] = 1
                    elif pix == rgb_6:
                        if obj_bool[5] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[5] = 1
                    elif pix == rgb_7:
                        if obj_bool[6] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[6] = 1
                    elif pix == rgb_8:
                        if obj_bool[7] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[7] = 1
                    elif pix == rgb_9:
                        if obj_bool[8] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[8] = 1
                f_line = seq + '    ' + msk_idx + '    ' + str(obj_count) + ' objs' + '\n'
                f.write(f_line)
                num_obj = num_obj + obj_count
                msk_count += 1
                print(" {} key frames processed.".format(msk_count))
            f_line2 = str(num_obj) + ' objects in ' + seq + '\n'
            f.write(f_line2)
            num_obj_sum = num_obj_sum + num_obj
            seq_count += 1
            print(" {} videos processed.".format(seq_count))
        f_line3 = str(num_obj_sum) + ' objects in total.'
        f.write(f_line3)
        f.close()
        print('All done !')

    def listPrint(self):
        startpath = os.getcwd() + '/360vsod/'
        f = open(os.getcwd() + '/360VSOD_categories.txt', 'w')

        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            f.write('{}{}/'.format(indent, os.path.basename(root)) + '\n')
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))
                f.write('{}{}'.format(subindent, f) + '\n')

        f.close()
        print('Done!')


if __name__ == '__main__':
    PT = ProcessingTool()
    #PT.split2whole()
    #PT.shiftRecover()
    #PT.frm2vid()
    #PT.ist2obj()
    #PT.ist_merge()
    #PT.getKeyFrm()
    #print('There are: ' + str(PT.numFrm()) + ' key frames.')
    #PT.frmStt()
    bar_show(PT.numFrm())
    #PT.demoMsk()
    #PT.mskRename()
    #PT.GTResize()
    #PT.seq2frm()
    #PT.mskRGB()
    #PT.mskEdit()
    #PT.objStt()
    PT.listPrint()