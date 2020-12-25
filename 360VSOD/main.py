import os
import cv2
import numpy as np
from PIL import Image
from time import sleep
import sys
import xml.etree.ElementTree as ET

pixel_shift = 640
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
                img_shift = np.roll(img_shift, pixel_shift, axis=1)
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
            msk_shift = np.roll(msk_shift, -1 * pixel_shift, axis=1)
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
        oriPath = os.getcwd() + '/mask_new/'
        trgPath = os.getcwd() + '/obj_new/'
        orilist = os.listdir(oriPath)
        count_video = 1
        for item in orilist:
            if item == '.DS_Store': continue
            ists_path = os.path.join(oriPath, item)
            objs_path = os.path.join(trgPath, item)
            if not os.path.exists(objs_path):
                os.makedirs(objs_path)

            count = 1
            for item in os.listdir(ists_path):
                if item == '.DS_Store': continue
                ist_path = os.path.join(os.path.abspath(ists_path), item)
                obj_path = os.path.join(os.path.abspath(objs_path), item)
                ist = cv2.imread(ist_path)
                ist = cv2.cvtColor(ist, cv2.COLOR_RGB2GRAY)
                ret, obj = cv2.threshold(ist, 0, 255, cv2.THRESH_BINARY)
                cv2.imwrite(obj_path, obj)
                print(" {} frames processed".format(count))
                count += 1
            print(" {} videos processed".format(count_video))
            count_video += 1

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
        src_path = os.getcwd() + '/_-eqmjLZGZ36k/'
        tgt_path = os.getcwd() + '/resized/'
        src_list = os.listdir(src_path)
        src_list.sort(key=lambda x: x[:-4])

        count = 1
        for frm in src_list:
            frm_path = src_path + frm
            img = cv2.imread(frm_path)
            img = cv2.resize(img, (3840, 1920))
            ret, img = cv2.threshold(img, 0, 128, cv2.THRESH_BINARY)
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
                    if j < 1980:
                        if msk[i, j, :].tolist() == rgb_3:
                            msk[i, j, 0] = 0
                            msk[i, j, 1] = 0
              #      if msk[i, j, :].tolist() == rgb_2:
               #         msk[i, j, 0] = 128
                #    elif msk[i, j, :].tolist() == rgb_3:
                 #       msk[i, j, 1] = 0
                  #  elif msk[i, j, :].tolist() == rgb_4:
                   #     msk[i, j, 1] = 128
                    #    msk[i, j, 2] = 0

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
        rgb_0 = (0, 0, 0)
        rgb_1 = (128, 0, 0)
        rgb_2 = (0, 128, 0)
        rgb_3 = (128, 128, 0)
        rgb_4 = (0, 0, 128)
        rgb_5 = (128, 0, 128)
        rgb_6 = (0, 128, 128)
        rgb_7 = (128, 128, 128)
        rgb_8 = (64, 0, 0)
        rgb_9 = (192, 0, 0)
        rgb_10 = (0, 64, 64)
        rgb_11 = (0, 192, 0)
        rgb_12 = (192, 192, 0)
        rgb_13 = (0, 0, 192)
        rgb_14 = (192, 192, 192)

        f = open(os.getcwd() + '/av360.txt', 'w')
        f2 = open(os.getcwd() + '/av360_obj_size.txt', 'w')

        seq_list = os.listdir(os.getcwd() + '/mask_new/')
        seq_count = 0
        num_obj_sum = 0
        num_frm_sum = 0
        for seq in seq_list:
            if seq == '.DS_Store': continue
            msk_list = os.listdir(os.path.join(os.getcwd() + '/mask_new/', seq))
            msk_list.sort(key=lambda x: x[:-4])
            msk_count = 0
            num_obj = 0
            size_ins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            num_ins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for msk_idx in msk_list:
                if msk_idx == '.DS_Store': continue
                msk_path = os.path.join(os.getcwd() + '/mask_new/', seq, msk_idx)
                msk = Image.open(msk_path)
                obj_count = 0
                obj_bool = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                rgb_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for pix in msk.getdata():
                    if pix == rgb_0:
                        rgb_count[0] += 1
                        continue
                    elif pix == rgb_1:
                        rgb_count[1] += 1
                        if obj_bool[0] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[0] = 1
                    elif pix == rgb_2:
                        rgb_count[2] += 1
                        if obj_bool[1] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[1] = 1
                    elif pix == rgb_3:
                        rgb_count[3] += 1
                        if obj_bool[2] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[2] = 1
                    elif pix == rgb_4:
                        rgb_count[4] += 1
                        if obj_bool[3] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[3] = 1
                    elif pix == rgb_5:
                        rgb_count[5] += 1
                        if obj_bool[4] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[4] = 1
                    elif pix == rgb_6:
                        rgb_count[6] += 1
                        if obj_bool[5] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[5] = 1
                    elif pix == rgb_7:
                        rgb_count[7] += 1
                        if obj_bool[6] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[6] = 1
                    elif pix == rgb_8:
                        rgb_count[8] += 1
                        if obj_bool[7] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[7] = 1
                    elif pix == rgb_9:
                        rgb_count[9] += 1
                        if obj_bool[8] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[8] = 1
                    elif pix == rgb_10:
                        rgb_count[10] += 1
                        if obj_bool[9] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[9] = 1
                    elif pix == rgb_11:
                        rgb_count[11] += 1
                        if obj_bool[10] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[10] = 1
                    elif pix == rgb_12:
                        rgb_count[12] += 1
                        if obj_bool[11] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[11] = 1
                    elif pix == rgb_13:
                        rgb_count[13] += 1
                        if obj_bool[12] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[12] = 1
                    elif pix == rgb_14:
                        rgb_count[14] += 1
                        if obj_bool[13] == 1:
                            continue
                        else:
                            obj_count += 1
                            obj_bool[13] = 1
                rgb_ratio = rgb_count / np.sum(rgb_count)
                for idx in range(15):
                    if rgb_ratio[idx] != 0:
                        num_ins[idx] += 1
                        size_ins[idx] += rgb_ratio[idx]
                f_line = seq + '    ' + msk_idx + '    ' + str(obj_count) + ' objs' + '\n'
                f.write(f_line)
                f_line = seq + '    ' + msk_idx + '    ' + str(rgb_ratio) + '\n'
                f2.write(f_line)
                num_obj = num_obj + obj_count
                msk_count += 1
                print(" {} key frames processed.".format(msk_count))
            f_line2 = str(num_obj) + ' objects in ' + seq + '    ' + str(msk_count) + ' key frames in ' + seq + '\n'
            f.write(f_line2)
            for idx in range(15):
                if num_ins[idx] != 0:
                    size_ins[idx] = size_ins[idx] / num_ins[idx]
            f_line2_1 = 'object size infomation: ' + str(size_ins) + '\n'
            f.write(f_line2_1)
            num_obj_sum = num_obj_sum + num_obj
            num_frm_sum = num_frm_sum + msk_count
            seq_count += 1
            print(" {} videos processed.".format(seq_count))
        f_line3 = str(num_obj_sum) + ' objects in total;' + '     ' + str(num_frm_sum) + ' key frames in total.'
        f.write(f_line3)
        f.close()
        f2.close()
        print('All done !')

    def listPrint(self):
        startpath = os.getcwd() + '/dataset_split/'
        f = open(os.getcwd() + '/360vsod_split.txt', 'w')

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

    def bbox2reg(self):
        obj_path = os.getcwd() + '/obj_reg/'
        xml_path = os.getcwd() + '/_-ZuXCMpVR24I_part2.xml'
        frm_path = os.getcwd() + '/_-ZuXCMpVR24I/'
        f = open(os.getcwd() + '/_-ZuXCMpVR24I_part2.txt', 'w')
        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()

        bbox_list = []
        obj_list = []

        for frm in xml_root.iter('image'):
            for obj in frm.iter('box'):
                obj_list.append([frm.attrib['name'], obj.attrib['label']])
                xmin = int(float(obj.attrib['xtl']))
                ymin = int(float(obj.attrib['ytl']))
                xmax = int(float(obj.attrib['xbr']))
                ymax = int(float(obj.attrib['ybr']))
                bbox_list.append([xmin, ymin, xmax, ymax])

        regShow(obj_list, bbox_list, frm_path, obj_path, f)
        f.close()
        print('done !')

    def sobjCount(self):
        f = open(os.getcwd() + '/sound_obj.txt', 'w')
        pathMain = '/home/yzhang1/PythonProjects/360vSOD/data/bbox_sound_object/'
        listMain = os.listdir(pathMain)
        countObj = 0
        count = 0
        for file in listMain:
            fileImg = file + '_img'
            pathSub = os.path.join(pathMain, file, fileImg)
            listFrm = os.listdir(pathSub)
            #listFrm.sort(key=lambda x: x[:-4])
            f.write(file + '  ' + str(len(listFrm)) + '\n')
            countObj += len(listFrm)
            count += 1
            print(" {} sequences  processed.".format(count))
        print(str(countObj) + '  sounding objects in total.')
        f.write(str(countObj) + '  sounding objects in total.')
        f.close()

    def fixation_match(self):
        fix_w_sound_path = os.getcwd() + '/fixation_with_sound/'
        fix_wo_sound_path = os.getcwd() + '/fixation_without_sound/'
        img_path = os.getcwd() + '/img/'

        count = 0
        img_list = os.listdir(img_path)
        for idx in img_list:
            print(idx)

            img = cv2.imread(os.path.join(img_path, idx))
            img = cv2.resize(img, (600, 300))

            fix_w_s = cv2.imread(os.path.join(fix_w_sound_path, idx[-9:]))
            fix_w_s = cv2.GaussianBlur(fix_w_s, (45, 45), 10)
            fix_w_s = cv2.normalize(fix_w_s, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8UC1)
            fix_w_s[:, :, 0] = 0
            #fix_w_s = cv2.applyColorMap(fix_w_s, cv2.COLORMAP_HOT)

            fix_wo_s = cv2.imread(os.path.join(fix_wo_sound_path, idx[-9:]))
            fix_wo_s = cv2.GaussianBlur(fix_wo_s, (45, 45), 10)
            fix_wo_s = cv2.normalize(fix_wo_s, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8UC1)
            fix_wo_s[:, :, 0] = 0
            #fix_wo_s = cv2.applyColorMap(fix_wo_s, cv2.COLORMAP_HOT)

            heatmap = cv2.addWeighted(fix_w_s, 1, fix_wo_s, 1, 0)
            overlay = cv2.addWeighted(img, 0.6, heatmap, 1.2, 0)
            cv2.imwrite(os.getcwd() + '/overlay/' + idx, overlay)

            count += 1
            print(str(count) + ' images processed.')

    def instanceOverlay(self):
        img_pth = os.getcwd() + '/img/'
        ins_pth = os.getcwd() + '/instance/'
        count = 0
        ins_list = os.listdir(ins_pth)
        for idx in ins_list:
            print(idx)
            img = cv2.imread(os.path.join(img_pth, idx))
            ins = cv2.imread(os.path.join(ins_pth, idx))
            overlay = cv2.addWeighted(img, 0.8, ins, 1.6, 0)
            overlay = cv2.resize(overlay, (600, 300))
            cv2.imwrite(idx, overlay)
            print(count + 1)
            count += 1

    def figShow(self):
        obj_pth = os.getcwd() + '/object/'
        img_pth = os.getcwd() + '/img/'
        ins_pth = os.getcwd() + '/instance/'
        bbox_pth = os.getcwd() + '/img_sphe_bbox/'
        bboxSphe_pth = os.getcwd() + '/img_sphere/'

        obj_list = os.listdir(obj_pth)
        count = 0
        for idx in obj_list:
            frm = np.zeros((256, 2600, 3))
            frm.fill(255)

            img = cv2.imread(os.path.join(img_pth, idx))
            img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
            ins = cv2.imread(os.path.join(ins_pth, idx))
            ins = cv2.resize(ins, (512, 256), interpolation=cv2.INTER_AREA)
            obj = cv2.imread(os.path.join(obj_pth, idx))
            obj = cv2.resize(obj, (512, 256), interpolation=cv2.INTER_AREA)
            bbox = cv2.imread(os.path.join(bbox_pth, idx))
            bbox = cv2.resize(bbox, (512, 256), interpolation=cv2.INTER_AREA)
            bboxSphe = cv2.imread(os.path.join(bboxSphe_pth, idx))
            bboxSphe = cv2.resize(bboxSphe, (512, 256), interpolation=cv2.INTER_AREA)

            frm[:, :512, :] = img
            frm[:, 522:1034, :] = ins
            frm[:, 1044:1556, :] = obj
            frm[:, 1566:2078, :] = bbox
            frm[:, 2088:2600, :] = bboxSphe
            cv2.imwrite(idx, frm)
            print(count+1)
            count += 1

    def wholeShow_1(self):
        fix_pth = os.getcwd() + '/fixation_overlay/'
        ins_pth = os.getcwd() + '/instance_overlay/'
        fix_list = os.listdir(fix_pth)
        fig = np.zeros((300, 1210, 3))
        fig.fill(255)

        for idx in fix_list:
            print(idx)
            fig[:, :600, :] = cv2.imread(os.path.join(fix_pth, idx))
            fig[:, 610:, :] = cv2.imread(os.path.join(ins_pth, idx))
            cv2.imwrite(idx, fig)

    def wholeShow_2(self):
        pth = os.getcwd() + '/sub_3/'
        sub_list = os.listdir(pth)
        fig = np.zeros((2160, 3650, 3))
        fig.fill(255)

        for i in range(7):
            for j in range(3):
                if i != 0 and j != 0:
                    fig[(300*i+10*i):(300*i+10*i+300), (1210*j+10*j):(1210*j+10*j+1210), :] = cv2.imread(
                        os.path.join(pth,
                                     sub_list[3*i+j]))
                elif i != 0 and j == 0:
                    fig[(300*i+10*i):(300*i+10*i+300), :1210, :] = cv2.imread(
                        os.path.join(pth,
                                     sub_list[3 * i + j]))
                elif i == 0 and j != 0:
                    fig[:300, (1210*j+10*j):(1210*j+10*j+1210), :] = cv2.imread(
                        os.path.join(pth,
                                     sub_list[3 * i + j]))
                else:
                    fig[:300, :1210, :] = cv2.imread(
                        os.path.join(pth,
                                     sub_list[3 * i + j]))
        cv2.imwrite('sub_3.png', fig)

    def qlt_show(self):
        num_model = 21
        sample_list = os.listdir(os.getcwd() + '/GT_fig_quantity/')

        pth_gt = '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/GTs/test/GT'
        pth_img = '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/GTs/test' \
                  '/HSAV360_test/Img_test'
        pth_sal_img = [pth_img,
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_ASNet',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_AADFNet',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_poolnet',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_cpd-r',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_basnet',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_egnet',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_scrn',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_u2net',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_RAS',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_f3net',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_gcpanet',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_scribblesod',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_minet',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_ldf',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_CSNet',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_CSFRes2Net',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_MGA_raft_sintel',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_rcrnet',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_cosnet',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_SSAV',
                       '/home/yzhang1/PythonProjects/360vSOD/experiments/weights_results/results_test/Sal_test_DDS',
                       pth_gt]

        fig_img = np.zeros((256 * (num_model + 2) + 10 * (num_model + 1), 512 * 8 + 70, 3)) # eight samples
        fig_img.fill(255)

        count = 0
        for item in sample_list:
            for idx in range(num_model + 2):
                print(idx)
                pthCurr = os.path.join(pth_sal_img[idx], item)
                imgCurr = cv2.imread(pthCurr)
                if idx == 0 and count == 0:
                    fig_img[:256, :512, :] = imgCurr
                elif idx == 0 and count != 0:
                    fig_img[:256, count * (10 + 512): count * (10 + 512) + 512, :] = imgCurr
                elif idx != 0 and count == 0:
                    fig_img[idx * (10 + 256): idx * (10 + 256) + 256, :512, :] = imgCurr
                else:
                    fig_img[idx * (10 + 256): idx * (10 + 256) + 256,
                    count * (10 + 512): count * (10 + 512) + 512, :] = imgCurr

            count += 1

        cv2.imwrite('fig_test_imgSOD.png', fig_img)

    def qlt_show2(self):
        num_model_img = 11
        num_model_vid = 1
        sample_list = os.listdir(os.getcwd() + '/GT/')

        pth_gt = '/home/yzhang1/PythonProjects/omnivsod/results_test/GTs/GT/'
        pth_img = '/home/yzhang1/PythonProjects/omnivsod/results_test/GTs/Img/'
        pth_sal_img = [pth_img,
                       os.getcwd() + '/ft_basnet_e4/HSAV360/',
                       os.getcwd() + '/ft_cpd_e7/HSAV360/',
                       os.getcwd() + '/ft_poolnet_e8/HSAV360/',
                       os.getcwd() + '/ft_egnet_e8/HSAV360/',
                       os.getcwd() + '/ft_scrn_e8/HSAV360/',
                       os.getcwd() + '/ft_ras_e6/HSAV360/',
                       os.getcwd() + '/ft_gcpanet_e7/HSAV360/',
                       os.getcwd() + '/ft_f3net_e2/HSAV360/',
                       os.getcwd() + '/ft_minet_e9/HSAV360/',
                       os.getcwd() + '/ft_csnet_e10/HSAV360/',
                       os.getcwd() + '/ft_csf_e10/HSAV360/',
                       pth_gt]
        pth_sal_vid = [pth_img,
                       os.getcwd() + '/ft_rcrnet_e7/HSAV360/',
                       pth_gt]

        fig_img = np.zeros((256 * (num_model_img + 2) + 10 * (num_model_img + 1), 512 * 8 + 70, 3))
        fig_img.fill(255)
        fig_vid = np.zeros((256 * (num_model_vid + 2) + 10 * (num_model_vid + 1), 512 * 8 + 70, 3))
        fig_vid.fill(255)

        count = 0
        for item in sample_list:
            for idx in range(num_model_img + 2):
                pthCurr = os.path.join(pth_sal_img[idx], item)
                imgCurr = cv2.imread(pthCurr)
                if idx == 0 and count == 0:
                    fig_img[:256, :512, :] = imgCurr
                elif idx == 0 and count != 0:
                    fig_img[:256, count * (10 + 512): count * (10 + 512) + 512, :] = imgCurr
                elif idx != 0 and count == 0:
                    fig_img[idx * (10 + 256): idx * (10 + 256) + 256, :512, :] = imgCurr
                else:
                    fig_img[idx * (10 + 256): idx * (10 + 256) + 256,
                    count * (10 + 512): count * (10 + 512) + 512, :] = imgCurr

            for idx in range(num_model_vid + 2):
                pthCurr = os.path.join(pth_sal_vid[idx], item)
                imgCurr = cv2.imread(pthCurr)
                if idx == 0 and count == 0:
                    fig_vid[:256, :512, :] = imgCurr
                elif idx == 0 and count != 0:
                    fig_vid[:256, count * (10 + 512): count * (10 + 512) + 512, :] = imgCurr
                elif idx != 0 and count == 0:
                    fig_vid[idx * (10 + 256): idx * (10 + 256) + 256, :512, :] = imgCurr
                else:
                    fig_vid[idx * (10 + 256): idx * (10 + 256) + 256,
                    count * (10 + 512): count * (10 + 512) + 512, :] = imgCurr

            count += 1

        cv2.imwrite('fig_ft_imgSOD.png', fig_img)
        cv2.imwrite('fig_ft_vidSOD.png', fig_vid)

    def file_rename(self):
        with open(os.getcwd() + '/test_img.lst', 'r') as f:
            img_list = [x.strip() for x in f.readlines()]
        asnet = []
        dds = []
        for item in img_list:
            new_item = '/home/yzhang1/PythonProjects/360vSOD/code/ASNet_output/' + item[52:]
            asnet.append(new_item)
        for item in img_list:
            new_item = '/home/yzhang1/PythonProjects/360vSOD/code/DDS_output/' + item[52:]
            dds.append(new_item)

       # for item in asnet:
        #    name_list = item[55:].split('/')
         #   frm_name = name_list[0] + '-' + name_list[1] + '-' + name_list[2]
          #  new_item = '/home/yzhang1/PythonProjects/360vSOD/code/ASNet/' + frm_name
           # os.rename(item, new_item)
        for item in dds:
            name_list = item[53:].split('/')
            frm_name = name_list[0] + '-' + name_list[1] + '-' + name_list[2]
            new_item = '/home/yzhang1/PythonProjects/360vSOD/code/DDS/' + frm_name
            os.rename(item, new_item)

    def omniCAP_show(self):
        fig_img = np.zeros((256 * 2 + 10, 512 * 7 + 60, 3))
        fig_img.fill(255)

        line1_list = os.listdir(os.getcwd() + '/overlay_p3/')
        line1_list.sort(key=lambda x: x[:-4])
        line2_list = os.listdir(os.getcwd() + '/objs/')
        line2_list.sort(key=lambda x: x[:-4])
        for idx in range(7):
            oly = cv2.imread(os.path.join(os.getcwd() + '/overlay_p3', line1_list[idx]))
            oly = cv2.resize(oly, (512, 256), interpolation=cv2.INTER_AREA)
            obj = cv2.imread(os.path.join(os.getcwd() + '/objs/', line2_list[idx]))
            obj = cv2.resize(obj, (512, 256), interpolation=cv2.INTER_AREA)
            if idx == 0:
                fig_img[:256, :512, :] = oly
                fig_img[266:, :512, :] = obj
            else:
                fig_img[:256, (idx * (512 + 10)):(idx * (512 + 10) + 512), :] = oly
                fig_img[266:, (idx * (512 + 10)):(idx * (512 + 10) + 512), :] = obj

        cv2.imwrite('fig_360cap_3.png', fig_img)

    def sound_map_overlay(self):
        img_pth = os.getcwd() + '/davpnet_img.png'
        sound_pth = os.getcwd() + '/0015.jpg'

        img = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)

        sound = cv2.imread(sound_pth, cv2.IMREAD_GRAYSCALE)
        sound = cv2.resize(sound, (512, 256), interpolation=cv2.INTER_AREA)

        # threshold
       # mean_sound = np.mean(sound)
        #for i in range(256):
        #    for j in range(512):
        #        if sound[i, j] < 4 * mean_sound: sound[i,j] = 0

        # shift
        sound = np.roll(sound, 0, axis=1)
        sound = np.roll(sound, 20, axis=0)

        mean_value = np.mean(sound)
        mean_map = np.empty((256, 512), dtype=np.float)
        mean_map[:] = mean_value

        overlay = cv2.addWeighted(img, 1, sound, 1, 0)
        cv2.imwrite('1.png', overlay)
        cv2.imwrite('sound.png', sound)
       # cv2.imwrite('mean_map.png', mean_map)
        print()

    def sound_map_prepro(self):
        maps_ori_pth = os.getcwd() + '/sound_map_ori/'
        maps_ori_list = os.listdir(maps_ori_pth)
        maps_fin_pth = os.getcwd() + '/sound_map/'

        count = 0
        for item in maps_ori_list:
            if not os.path.exists(maps_fin_pth + item): os.mkdir(maps_fin_pth + item)
            frms_pth = os.path.join(maps_ori_pth, item)
            frms_list = os.listdir(frms_pth)
            frms_list.sort(key=lambda x: x[:-4])
            for idx in range(len(frms_list)):
                if idx % 6 == 0:
                    img_pth = os.path.join(frms_pth, frms_list[idx])
                    img = cv2.imread(img_pth)
                    img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
                    img_save_pth = os.path.join(maps_fin_pth, item, 'frame_' + format(str(idx), '0>6s') + '.png')
                    cv2.imwrite(img_save_pth, img)
            count += 1
            print(str(count) + ' videos processed.')

    def rgb_exchange(self):
        img_pth = os.getcwd() + '/_-SdGCX2H-_Uk_000084.png'
        img = cv2.imread(img_pth)  # BGR
        for i in range(1920):
            for j in range(3840):
                c1 = img[i,j,0]
                c2 = img[i, j, 1]
                c3 = img[i, j, 2]
                if c1 == 0 and c2 == 128 and c3 == 0:
                    img[i, j, 0] = 0
                    img[i, j, 1] = 0
                    img[i, j, 2] = 128
                elif c1 == 0 and c2 == 0 and c3 == 128:
                    img[i, j, 0] = 0
                    img[i, j, 1] = 128
                    img[i, j, 2] = 0
        cv2.imwrite('_-SdGCX2H-_Uk_000084_rgb.png', img)


def regShow(obj_list, bbox_list, ori_path, save_path, txt):
    count = 0
    for obj in obj_list:
        frm = Image.open(ori_path + obj[0])
        frm_crop = frm.crop(bbox_list[count])
        frm_crop.save(save_path + obj[0][:-4] + '_' + obj[1] + '.png')

        # for those splitted
        #bbox_list[count][0] -= pixel_shift
        #bbox_list[count][2] -= pixel_shift

        txt.write(obj[0][:-4] + '  ' + obj[1] + '  ' + str(bbox_list[count]) + '\n')
        count += 1
        print(" {} sounding objects counted.".format(count))

def rgbd_test_prepar():
    ori_pth = os.getcwd() + '/RGBD_for_test/'
    fin_pth = os.getcwd() + '/FS/'
    set_list = os.listdir(ori_pth)
    count = 0
    for set in set_list:
        set_pth = os.path.join(ori_pth, set, 'FS')
       # set_pth = os.path.join(ori_pth, set)
        img_list = os.listdir(set_pth)
        img_list.sort(key=lambda x: x[:-4])
        for img_idx in img_list:
            img_pth = os.path.join(set_pth, img_idx)
            new_img_idx = set + '_' + img_idx
            new_img_pth = os.path.join(fin_pth, new_img_idx)
            os.rename(img_pth, new_img_pth)
        count += 1
        print(" {} datasets processed.".format(count))

def LFSOD_split():
    txt_file = open(os.getcwd() + '/DUT-LF_test.txt')
    items = txt_file.readlines()
    count = 1
    for item in items:
        depth_o = os.getcwd() + '/DUT/depth/' + item[:4] + '.png'
        depth_f = os.getcwd() + '/DUT-LF_test/depth/' + item[:4] + '.png'
        os.rename(depth_o, depth_f)
        gt_o = os.getcwd() + '/DUT/gt/' + item[:4] + '.png'
        gt_f = os.getcwd() + '/DUT-LF_test/gt/' + item[:4] + '.png'
        os.rename(gt_o, gt_f)
        rgb_o = os.getcwd() + '/DUT/rgb/' + item[:4] + '.jpg'
        rgb_f = os.getcwd() + '/DUT-LF_test/rgb/' + item[:4] + '.jpg'
        os.rename(rgb_o, rgb_f)
        print(count)
        count += 1

def FileRename():
    dep_pth = os.getcwd() + '/HFUT/depth/'
    gt_pth = os.getcwd() + '/HFUT/GT/'
    rgb_pth = os.getcwd() + '/HFUT/RGB/'

    ori_pth = rgb_pth
    list_ori = os.listdir(ori_pth)
    list_ori.sort(key=lambda x: (float(x[:-4]), len(x[:-4])))

    count = 1
    for item in list_ori:
        item_ori_pth = os.path.join(ori_pth, item)
        item_new = format(str(count), '0>5s') + item[-4:]
        item_fin_pth = os.path.join(ori_pth, item_new)
        os.rename(item_ori_pth, item_fin_pth)
        print(count)
        count += 1

def HFUT_FS_rename():
    ori_pth = os.getcwd() + '/HFUT/'
    fin_pth = os.getcwd() + '/rename/'
    list_ori = os.listdir(ori_pth)
    list_ori.sort(key=lambda x: (float(x[:-16]), len(x[:-16])))
    count = 1
    curr_idx = '2'
    for item in list_ori:
        if item == '2__refocus_03.jpg': new_item = '00001__refocus_03.jpg'
        else:
            item_list = item.split('_')
            if item_list[0] != curr_idx:
                curr_idx = item_list[0]
                count += 1
            new_item = format(str(count), '0>5s') + item[-16:]
        os.rename(ori_pth + item, fin_pth + new_item)


def LFSOD_FS_Split():
    fs_pth = os.getcwd() +'/HFUT/'
    test_pth = os.getcwd() + '/test/'
    list_fs = os.listdir(fs_pth)
    list_fs.sort(key=lambda x: x[:-4])
    txt_file = open(os.getcwd() + '/HFUT_test.txt')
    text_idxs = txt_file.readlines()
    for i in range(len(text_idxs)):
        text_idxs[i] = text_idxs[i][:-1]
    count = 1
    for item in list_fs:
        item_ori_pth = os.path.join(fs_pth, item)
        item_list = item.split('_')
        if item_list[0] in text_idxs:
            os.rename(item_ori_pth, test_pth + item)
        print(count)
        count += 1

def Lytro_FS_rename():
    ori_pth = os.getcwd() + '/Lytro/'
    new_pth = os.getcwd()  + '/new/'
    list_ori = os.listdir(ori_pth)
    for item in list_ori:
        new_item = item[4:]
        os.rename(ori_pth + item, new_pth + new_item)

def fs_for_test():
    ori_pth = os.getcwd() + '/FS/'
    fss_list = os.listdir(ori_pth)
    count = 0
    for fs in fss_list:
        fs_list = fs.split('_')
        fs_dir = os.path.join(os.getcwd(), fs_list[0])
        if not os.path.exists(fs_dir): os.makedirs(fs_dir)
        fs_name = fs[len(fs_list[0]) + 1:]
        new_pth = os.path.join(fs_dir, fs_name)
        old_pth = os.path.join(ori_pth, fs)
        os.rename(old_pth, new_pth)
        count += 1
        print(count)

def listTest():
    img_pth = os.getcwd() + '/test/'
    img_list = os.listdir(img_pth)
    img_list.sort(key=lambda x: x[:-4])
    f = open(os.getcwd() + '/test.txt', 'w')

    for item in img_list:
        f.write(item[:-4]  + '\n')
    f.close()

def ToTestLFSOD():
    print()
    total_list = os.listdir(os.getcwd() + '/Results/')
    total_list.sort(key=lambda x: x[:-4])
    for item in total_list:
        item_list = item.split('_')
        new_dir = os.path.join(os.getcwd(), item_list[0])
        new_name = item_list[1]
        if not os.path.exists(new_dir):  os.makedirs(new_dir)
        old_pth = os.path.join(os.getcwd() + '/Results/', item)
        new_pth = os.path.join(new_dir, new_name)
        os.rename(old_pth, new_pth)


def dataList():
    MSK_pth = '/home/yzhang1/PythonProjects/AV360/DATA/test/'
    Img_pth = '/home/yzhang1/PythonProjects/AV360/frame_key/'
    MSK_list = os.listdir(MSK_pth)
    f1 = open(os.getcwd() + '/test_img.txt', 'w')
    f2 = open(os.getcwd() + '/test_msk.txt', 'w')

    for Seq in MSK_list:
        Seq_pth = os.path.join(MSK_pth, Seq)
        Seq_list = os.listdir(Seq_pth)
        Seq_list.sort(key=lambda x: x[:-4])
        for frm in Seq_list:
            msk_pth = os.path.join(Seq_pth, frm)
            img_pth = os.path.join(Img_pth, Seq, frm)
            f1.write(img_pth + '\n')
            f2.write(msk_pth + '\n')

    f1.close()
    f2.close()

def dataList_2():
    ORI_pth = '/home/yzhang1/PythonProjects/AV360/obj_new/test/'
    ORI_list = os.listdir(ORI_pth)
    ORI_list.sort(key=lambda x: x[-1:])
    f = open(os.getcwd() + '/test_img_ca.txt', 'w')
    for CA in ORI_list:
        CA_pth = os.path.join(ORI_pth, CA)
        CA_list = os.listdir(CA_pth)
        for Seq in CA_list:
            Seq_pth = os.path.join(CA_pth, Seq)
            Seq_list = os.listdir(Seq_pth)
            Seq_list.sort(key=lambda x: x[:-4])
            for frm in Seq_list:
                msk_pth = os.path.join(Seq_pth, frm)
                f.write(msk_pth + '\n')
    f.close()

if __name__ == '__main__':
    dataList_2()
    #dataList()
    #ToTestLFSOD()
    #listTest()
    #fs_for_test()
    #Lytro_FS_rename()
    #HFUT_FS_rename()
    #LFSOD_FS_Split()
    #PT = ProcessingTool()
    #LFSOD_split()
    #FileRename()
    #rgbd_test_prepar()
    #PT.rgb_exchange()
    #bar_show(PT.numFrm())
    #PT.bbox2reg()
    #PT.sobjCount()
    #PT.split2whole()
    #PT.shiftRecover()
    #PT.frm2vid()
    #PT.ist2obj()
    #PT.ist_merge()
    #PT.getKeyFrm()
    #print('There are: ' + str(PT.numFrm()) + ' key frames.')
    #PT.frmStt()
    #PT.demoMsk()
    #PT.mskRename()
    #PT.GTResize()
    #PT.seq2frm()
    #PT.mskRGB()
    #PT.mskEdit()
    #PT.objStt()
    #PT.listPrint()
    #PT.fixation_match()
    #PT.instanceOverlay()
    #PT.figShow()
    #PT.wholeShow_2()
    #PT.qlt_show()
    #PT.qlt_show2()
    #PT.file_rename()
    #PT.omniCAP_show()
    #PT.sound_map_overlay()
    #PT.sound_map_prepro()
