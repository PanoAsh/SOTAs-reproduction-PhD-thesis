import numpy as np
import cv2
import os
from PIL import  Image

RGB_background = tuple([0, 0, 0])

def KeySelect():
    videos_pth = os.getcwd() + '/saliency_re_order/'
    video_list = os.listdir(videos_pth)
    for vid in video_list:
        if vid == '.DS_Store': continue
        vid_pth = os.path.join(videos_pth, vid)
        frm_list = os.listdir(vid_pth)
        frm_list.sort(key=lambda x: x[:-4])
        for frm in frm_list:
            if frm == '.DS_Store': continue
            if int(frm[:-4]) % 6 == 0:
                frm_pth = os.path.join(vid_pth, frm)
                new_pth = os.path.join(os.getcwd() + '/saliency_key_frame/', vid)
                if not os.path.exists(new_pth): os.makedirs(new_pth)
                new_pth = os.path.join(new_pth, frm)
                os.rename(frm_pth, new_pth)
        print(vid + 'done.')

def ReOrder():
    pth = os.getcwd() + '/_-1LM84FSzW0g_1/'
    frame_list = os.listdir(pth)
    frame_list.sort(key=lambda x: x[:-4])
    for frm in frame_list:
        if frm == '.DS_Store': continue
        old_pth = os.path.join(pth, frm)
        order = format(str(int(frm[:-4]) - 300), '0>5s')
        new_pth = os.getcwd() + '/key/' + order + '.png'
        os.rename(old_pth, new_pth)

def AnnotationPrep():
    msks_pth = os.getcwd() + '/mask_instance/'
    fixs_pth = os.getcwd() + '/saliency_key_frame/'
    vids_list = os.listdir(fixs_pth)
    vids_list.sort(key=lambda x: x[:-4])
    count = 0
    count_frm_total = 0
    for vid in vids_list:
        if vid == '.DS_Store': continue
        vid_pth = os.path.join(fixs_pth, vid)
        fix_list = os.listdir(vid_pth)
        fix_list.sort(key=lambda x: x[:-4])
        count_frm = 0
        for fix in fix_list:
            if fix == '.DS_Store': continue

            fix_pth = os.path.join(vid_pth, fix)
            fix_map = cv2.imread(fix_pth)
            msk_idx = 'frame_' + format(str(int(fix[:-4])), '0>6s') + '.png'
            msk_pth = os.path.join(msks_pth, vid, msk_idx)
            msk = cv2.imread(msk_pth)
            obj = cv2.cvtColor(msk, cv2.COLOR_RGB2GRAY)
            ret, obj = cv2.threshold(obj, 0, 255, cv2.THRESH_BINARY)

            # producing thresholding fixation map and save it to new dir
            ret, trs_fix_map = cv2.threshold(fix_map, 180, 255, cv2.THRESH_BINARY)
            trs_fix_map_mark = np.copy(trs_fix_map)
            trs_fix_map_mark[:, :, :-1] = 0
            fix_new_dir = os.path.join(os.getcwd() + '/saliency_threshold/', vid)
            if not os.path.exists(fix_new_dir): os.makedirs(fix_new_dir)
            fix_new_pth = os.path.join(fix_new_dir, fix)
            cv2.imwrite(fix_new_pth, trs_fix_map)

            # producing edge map and overlay it to fixation and save it to new dir
            msk_edge = cv2.Canny(obj, 0, 255)
            msk_edge = cv2.cvtColor(msk_edge, cv2.COLOR_GRAY2RGB)
            trs_fix_map_mark = cv2.resize(trs_fix_map_mark, (np.shape(msk)[1], np.shape(msk)[0]))
            oly_edge = cv2.addWeighted(trs_fix_map_mark, 1, msk_edge, 1, 0)
            oly_new_dir = os.path.join(os.getcwd() + '/overlay_edge/', vid)
            if not os.path.exists(oly_new_dir): os.makedirs(oly_new_dir)
            oly_new_pth = os.path.join(oly_new_dir, fix)
            cv2.imwrite(oly_new_pth, oly_edge)

            # overlay instance with fixation and save it to new dir
            trs_fix_map = cv2.resize(trs_fix_map, (np.shape(msk)[1], np.shape(msk)[0]))
            oly_ins = cv2.addWeighted(trs_fix_map, 1, msk, 1, 0)
            oly_new_dir = os.path.join(os.getcwd() + '/overlay_instance/', vid)
            if not os.path.exists(oly_new_dir): os.makedirs(oly_new_dir)
            oly_new_pth = os.path.join(oly_new_dir, fix)
            cv2.imwrite(oly_new_pth, oly_ins)

            # overlay edge map with original fixation and save it to new dir
            fix_map_mark = np.copy(fix_map)
            fix_map_mark[:, :, :-1] = 0
            fix_map_mark = cv2.resize(fix_map_mark, (np.shape(msk)[1], np.shape(msk)[0]))
            oly_ori = cv2.addWeighted(fix_map_mark, 1, msk_edge, 1, 0)
            oly_new_dir = os.path.join(os.getcwd() + '/overlay_ori/', vid)
            if not os.path.exists(oly_new_dir): os.makedirs(oly_new_dir)
            oly_new_pth = os.path.join(oly_new_dir, fix)
            cv2.imwrite(oly_new_pth, oly_ori)

            # producing alternative threshold maps and overlay it with edge map and save them
            ret, trs_fix_map_a1 = cv2.threshold(fix_map, 120, 255, cv2.THRESH_BINARY)
            trs_fix_map_a1_mark = np.copy(trs_fix_map_a1)
            trs_fix_map_a1_mark[:, :, :-1] = 0
            trs_fix_map_a1_mark = cv2.resize(trs_fix_map_a1_mark, (np.shape(msk)[1], np.shape(msk)[0]))
            oly_edge = cv2.addWeighted(trs_fix_map_a1_mark, 1, msk_edge, 1, 0)
            oly_new_dir = os.path.join(os.getcwd() + '/overlay_edge_120/', vid)
            if not os.path.exists(oly_new_dir): os.makedirs(oly_new_dir)
            oly_new_pth = os.path.join(oly_new_dir, fix)
            cv2.imwrite(oly_new_pth, oly_edge)

            ret, trs_fix_map_a2 = cv2.threshold(fix_map, 60, 255, cv2.THRESH_BINARY)
            trs_fix_map_a2_mark = np.copy(trs_fix_map_a2)
            trs_fix_map_a2_mark[:, :, :-1] = 0
            trs_fix_map_a2_mark = cv2.resize(trs_fix_map_a2_mark, (np.shape(msk)[1], np.shape(msk)[0]))
            oly_edge = cv2.addWeighted(trs_fix_map_a2_mark, 1, msk_edge, 1, 0)
            oly_new_dir = os.path.join(os.getcwd() + '/overlay_edge_60/', vid)
            if not os.path.exists(oly_new_dir): os.makedirs(oly_new_dir)
            oly_new_pth = os.path.join(oly_new_dir, fix)
            cv2.imwrite(oly_new_pth, oly_edge)

            fix_new_dir = os.path.join(os.getcwd() + '/saliency_threshold_60/', vid)
            if not os.path.exists(fix_new_dir): os.makedirs(fix_new_dir)
            fix_new_pth = os.path.join(fix_new_dir, fix)
            cv2.imwrite(fix_new_pth, trs_fix_map_a2)
            fix_new_dir = os.path.join(os.getcwd() + '/saliency_threshold_120/', vid)
            if not os.path.exists(fix_new_dir): os.makedirs(fix_new_dir)
            fix_new_pth = os.path.join(fix_new_dir, fix)
            cv2.imwrite(fix_new_pth, trs_fix_map_a1)

            count_frm += 1
            print(" {} frames have been processed.".format(count_frm))

        count += 1
        print(" {} videos have been processed.".format(count))
        count_frm_total += count_frm
        print(" {} total frames have been processed.".format(count_frm_total))

def AnnotationPrep2():
    msks_pth = os.getcwd() + '/mask_ori/'
    fixs_pth = os.getcwd() + '/saliency/'
    vids_list = os.listdir(fixs_pth)
    vids_list.sort(key=lambda x: x[:-4])
    count = 0
    count_frm_total = 0
    for vid in vids_list:
        if vid == '.DS_Store': continue
        vid_pth = os.path.join(fixs_pth, vid)
        fix_list = os.listdir(vid_pth)
        fix_list.sort(key=lambda x: x[:-4])
        count_frm = 0
        for fix in fix_list:
            if fix == '.DS_Store': continue
            if int(fix[:-4]) % 6 != 0: continue

            fix_pth = os.path.join(vid_pth, fix)
            fix_map = cv2.imread(fix_pth)
            msk_idx = 'frame_' + format(str(int(fix[:-4])), '0>6s') + '.png'
            msk_pth = os.path.join(msks_pth, vid, msk_idx)
            msk = cv2.imread(msk_pth)
            obj = cv2.cvtColor(msk, cv2.COLOR_RGB2GRAY)
            ret, obj = cv2.threshold(obj, 0, 255, cv2.THRESH_BINARY)

            # producing thresholding fixation map and save it to new dir
            ret, trs_fix_map = cv2.threshold(fix_map, 127, 255, cv2.THRESH_BINARY)
            trs_fix_map_mark = np.copy(trs_fix_map)
            trs_fix_map_mark[:, :, :-1] = 0
            fix_new_dir = os.path.join(os.getcwd() + '/saliency_threshold/', vid)
            if not os.path.exists(fix_new_dir): os.makedirs(fix_new_dir)
            fix_new_pth = os.path.join(fix_new_dir, fix)
            cv2.imwrite(fix_new_pth, trs_fix_map)

            # producing edge map and overlay it to fixation and save it to new dir
            msk_edge = cv2.Canny(obj, 0, 255)
            msk_edge = cv2.cvtColor(msk_edge, cv2.COLOR_GRAY2RGB)
            trs_fix_map_mark = cv2.resize(trs_fix_map_mark, (np.shape(msk)[1], np.shape(msk)[0]))
            oly_edge = cv2.addWeighted(trs_fix_map_mark, 1, msk_edge, 1, 0)
            oly_new_dir = os.path.join(os.getcwd() + '/overlay_edge/', vid)
            if not os.path.exists(oly_new_dir): os.makedirs(oly_new_dir)
            oly_new_pth = os.path.join(oly_new_dir, fix)
            cv2.imwrite(oly_new_pth, oly_edge)

            # producing new instance map and save it to new dir
            sal_pix = np.where(trs_fix_map_mark[:, :, 2] != 0)
            msk_sal_RGB = msk[sal_pix[0], sal_pix[1], :]
            msk_sal_RGB = np.unique(msk_sal_RGB, axis=0)
            num_RGB = np.shape(msk_sal_RGB)[0]
            keep_pix = []  # keep the pixels in salient instance regions
            for idx in range(num_RGB):
                curr_RGB = tuple(msk_sal_RGB[idx])
                if curr_RGB == RGB_background: continue
                keep_pix.append(np.where((msk[:, :, 0] == msk_sal_RGB[idx][0]) &
                                         (msk[:, :, 1] == msk_sal_RGB[idx][1]) &
                                         (msk[:, :, 2] == msk_sal_RGB[idx][2])))

            new_msk = np.zeros((np.shape(msk)[0], np.shape(msk)[1], 3))
            if keep_pix != []:
                num_ins = len(keep_pix)
                for idx in range(num_ins):
                    curr_pix = keep_pix[idx]
                    new_msk[curr_pix[0], curr_pix[1], :] = msk[curr_pix[0], curr_pix[1], :]

            ins_new_dir = os.path.join(os.getcwd() + '/new_mask_instance/', vid)
            if not os.path.exists(ins_new_dir): os.makedirs(ins_new_dir)
            ins_new_pth = os.path.join(ins_new_dir, fix)
            cv2.imwrite(ins_new_pth, new_msk)

            count_frm += 1
            print(" {} frames have been processed.".format(count_frm))

        count += 1
        print(" {} videos have been processed.".format(count))
        count_frm_total += count_frm
        print(" {} total frames have been processed.".format(count_frm_total))

def demo():
    vid_name = 'demo'
    vid_H = 300
    vid_W = 1210
    vid_fps = 1
    frm_list = os.listdir(os.getcwd() + '/' + 'demo_edge' + '/')
    frm_list.sort(key=lambda x: x[:-4])
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    vid = cv2.VideoWriter(os.getcwd() + '/' + vid_name + '.avi', fourcc, vid_fps, (vid_W, vid_H))

    count = 0
    for frm in frm_list:
        frm_path = os.getcwd() + '/' + 'demo_edge' + '/' + frm
        frm_path2 = os.getcwd() + '/' + 'demo_instance' + '/' + frm
        frm_edge = cv2.imread(frm_path)
        frm_edge = cv2.resize(frm_edge, (600, 300))
        frm_instance = cv2.imread(frm_path2)
        frm_instance = cv2.resize(frm_instance, (600, 300))
        frm_show = np.zeros((vid_H, vid_W, 3))
        frm_show[:, :600, :] = frm_edge
        frm_show[:, 610:, :] = frm_instance
       # cv2.imwrite(frm, frm_show)
        vid.write(frm_show)
        count += 1
        print(" {} frames added.".format(count))
    vid.release()
    print(' Done !')

def av_annotation():
    Imgs_pth = os.getcwd() + '/Img/'
    Sals_pth = os.getcwd() + '/Sal/'
    video_list = os.listdir(Imgs_pth)
    count = 0
    for vid in video_list:
        if vid == '.DS_Store': continue
        vid_pth = os.path.join(Imgs_pth, vid)
        frm_list = os.listdir(vid_pth)
        frm_list.sort(key=lambda x: x[:-4])
        for img in frm_list:
            if img == '.DS_Store': continue
            img_pth = os.path.join(vid_pth, img)
            sal_pth = os.path.join(Sals_pth, vid, img)
            img_curr = cv2.imread(img_pth)
            img_curr = cv2.resize(img_curr, (600, 300))
            sal_curr = cv2.imread(sal_pth)
            ret, sal_curr = cv2.threshold(sal_curr, 127, 255, cv2.THRESH_BINARY)
            sal_curr[:, :, :-1] = 0
            oly_curr = cv2.addWeighted(img_curr, 0.4, sal_curr, 1, 0)

            oly_dir = os.path.join(os.getcwd() + '/Overlaid/', vid)
            if not os.path.exists(oly_dir): os.makedirs(oly_dir)
            oly_pth = os.path.join(oly_dir, img)
            cv2.imwrite(oly_pth, oly_curr)
        count += 1
        print(count)

def key_frame_extraction():
    olys_pth = os.getcwd() + '/Overlaid_pre/'
    olys_list = os.listdir(olys_pth)
    count = 0
    for oly in olys_list:
        if oly == '.DS_Store': continue
        frm_pth = os.path.join(olys_pth, oly)
        frm_list = os.listdir(frm_pth)
        frm_list.sort(key=lambda x: x[:-4])
        for frm in frm_list:
            if frm == '.DS_Store': continue
            if int(frm[:-4]) % 6 == 0:
                ori_pth = os.path.join(frm_pth, frm)
                new_dir = os.path.join(os.getcwd() + '/Overlaid_key/', oly)
                if not os.path.exists(new_dir): os.makedirs(new_dir)
                new_pth = os.path.join(new_dir, frm)
                os.rename(ori_pth, new_pth)
        count += 1
        print(count)

def av_new_mask():
    ori_pth = os.getcwd() + '/mask_new/'
    sup_pth = os.getcwd() + '/mask_1217/'
    list_dir = os.listdir(sup_pth)
    for new_item in list_dir:
        if new_item == '.DS_Store': continue
        msk_1_pth = os.path.join(ori_pth, new_item)
        msk_2_pth = os.path.join(sup_pth, new_item)
        new_pth = os.path.join(os.getcwd() + '/new/', new_item)
        if not os.path.exists(new_pth): os.makedirs(new_pth)
        msk_list = os.listdir(msk_2_pth)
        msk_list.sort(key=lambda x: x[:-4])
        ori_list = os.listdir(msk_1_pth)
        ori_list.sort(key=lambda x: x[:-4])
        count = 1
        for item in ori_list:
            if item == '.DS_Store': continue
            base_pth = os.path.join(msk_1_pth, item)
            ori = cv2.imread(base_pth)
            if item in msk_list:
                add_pth = os.path.join(msk_2_pth, item)
                add = cv2.imread(add_pth)
                for i in range(add.shape[0]):
                    for j in range(add.shape[1]):
                        if add[i, j, 0] != 0 or add[i, j, 1] != 0 or add[i, j, 2] != 0:
                            if ori[i, j, 0] != 0 or ori[i, j, 1] != 0 or ori[i, j, 2] != 0:
                                ori[i, j, :] = 0
                ori = ori + add
            cv2.imwrite(os.path.join(new_pth, item), ori)
            print(count)
            count += 1


def frm_rename():
    ori_pth = os.getcwd() + '/ori_frm/'
    fin_pth = os.getcwd() + '/fin_frm/'
    frm_list = os.listdir(ori_pth)
    frm_list.sort(key=lambda x: x[:-4])
    for frm in frm_list:
        idx = int(frm[-10:-4])
        frm_ori_pth = os.path.join(ori_pth, frm)
        frm_fin_pth = os.path.join(fin_pth, format(str(idx), '0>5s') + '.png')
        os.rename(frm_ori_pth, frm_fin_pth)

def check_new_av():
    msks_new_pth = os.getcwd() + '/mask_new/'
    msks_list = os.listdir(msks_new_pth)
    check_new_pth = os.getcwd() + '/check_new_AV360/'
    count = 0
    for msks in msks_list:
        if msks == '.DS_Store': continue
        msk_list = os.listdir(os.path.join(msks_new_pth, msks))
        for item in msk_list:
            if item == '.DS_Store': continue
            item_pth = os.path.join(msks_new_pth, msks, item)
            sal_pth = os.path.join(os.getcwd() + '/saliency_overlaid/', msks, item)
            sal = cv2.imread(sal_pth)
            sal = cv2.resize(sal, (1000, 500))
            msk = cv2.imread(item_pth)
            msk = cv2.resize(msk, (1000, 500))
            oly = cv2.addWeighted(msk, 0.8, sal, 0.5, 0)
            SAL_pth = os.path.join(os.getcwd() + '/saliency/', msks, item)
            SAL = cv2.imread(SAL_pth)
            ret, SAL = cv2.threshold(SAL, 127, 255, cv2.THRESH_BINARY)
            SAL = cv2.resize(SAL, (1000, 500))
            oly_2 = cv2.addWeighted(oly, 0.8, SAL, 0.5, 0)
            oly_dir = os.path.join(check_new_pth, msks)
            if not os.path.exists(oly_dir): os.makedirs(oly_dir)
            oly_pth = os.path.join(oly_dir, item)
            cv2.imwrite(oly_pth, oly_2)
        count += 1
        print(count)

def demoMsk():
        seq_list = os.listdir(os.getcwd() + '/check_new_AV360/')
        demo_w = 600
        demo_h = 300
        frmShow = 10
        vid_oly = cv2.VideoWriter(os.getcwd() + '/demo_instance_overlay.avi', 0, 3, (demo_w, demo_h))

        count = 1
        for seq in seq_list:
            if seq == '.DS_Store': continue
            msk_list = os.listdir(os.getcwd() + '/check_new_AV360/' + seq)
            msk_list.sort(key=lambda x: x[:-4])
            demo_itv = int(len(msk_list) / frmShow)

            for idx in range(frmShow):
                if idx == '.DS_Store': continue
                msk_path = os.path.join(os.getcwd() + '/check_new_AV360/', seq,
                                        msk_list[idx * demo_itv])
                msk = cv2.imread(msk_path)
                vid_oly.write(msk)
            print("{} videos processed.".format(count))
            count += 1

        vid_oly.release()

seqs_pth = os.getcwd() + '/source/'
frms_pth = os.getcwd() + '/frm/'

def VideoToImg():
    seq_list = os.listdir(seqs_pth)

    count = 1
    for seq in seq_list:
        if seq.endswith('.mp4'):
            seq_path = os.path.join(seqs_pth, seq)
            frm_path = os.path.join(frms_pth, seq)
            frm_path = frm_path[:-4]
            if not os.path.exists(frm_path): os.makedirs(frm_path)
            cap = cv2.VideoCapture(seq_path)
            frames_num = int(cap.get(7))
            countF = 0
            for i in range(frames_num):
                ret, frame = cap.read()
                if frame is None: continue
                #frame = cv2.resize(frame, (512, 256))
                if countF % 6 == 0:
                    cv2.imwrite(os.path.join(frm_path, format(str(countF), '0>5s') + '.png'), frame)
                countF += 1
                print(" {} frames processed".format(countF))
            print(" {} videos processed".format(count))
            count += 1

def delete_mask():
    msk_redundant_pth = os.getcwd() + '/msk_redundant/'
    msk_new_pth = os.getcwd() + '/msk_new/'
    msk_list = os.listdir(msk_redundant_pth)
    count = 0
    for name in msk_list:
        if name == '.DS_Store': continue
        msk_pth = os.path.join(msk_redundant_pth, name)
        msk = cv2.imread(msk_pth)
        for i in range(msk.shape[0]):
            for j in range(msk.shape[1]):
                if msk[i, j, 0] == 0 and msk[i, j, 1] == 0 and msk[i, j, 2] == 64:
                    msk[i, j, 0] = 0
                    msk[i, j, 1] = 0
                    msk[i, j, 2] = 0
                #elif msk[i, j, 0] == 0 and msk[i, j, 1] == 0 and msk[i, j, 2] == 64:
                 #   msk[i, j, 0] = 0
                  #  msk[i, j, 1] = 0
                   # msk[i, j, 2] = 0
        cv2.imwrite(os.path.join(msk_new_pth, name), msk)
        count += 1
        print(count)


def objStt():
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


def ToTestLFSOD():
    total_list = os.listdir(os.getcwd() + '/Results/')
    total_list.sort(key=lambda x: x[:-4])
    for item in total_list:
        if item == '.DS_Store': continue
        item_list = item.split('_')
        new_dir = os.path.join(os.getcwd(), item_list[0])
        new_name = item_list[1]
        if not os.path.exists(new_dir):  os.makedirs(new_dir)
        old_pth = os.path.join(os.getcwd() + '/Results/', item)
        new_pth = os.path.join(new_dir, new_name)
        os.rename(old_pth, new_pth)

def LFSOD_fig_supp():
    num_model = 7
    sample_list = os.listdir(os.getcwd() + '/LFSOD_sample/LFSD/')

    pth_gt = os.getcwd() + '/fig_supp/GT/LFSD/'
    pth_img = os.getcwd() + '/fig_supp/RGB/LFSD/'
    pth_sal_img = [pth_img,
                   pth_gt,
                   os.getcwd() + '/fig_supp/ours/LFSD/',
                   os.getcwd() + '/fig_supp/ERNetT/LFSD/',
                   os.getcwd() + '/fig_supp/ERNetS/LFSD/',
                   os.getcwd() + '/fig_supp/S2MA/LFSD/',
                   os.getcwd() + '/fig_supp/D3Net/LFSD',
                   os.getcwd() + '/fig_supp/LFS/LFSD/',
                   os.getcwd() + '/fig_supp/DILF/LFSD/'
                   ]

    fig_img = np.zeros((300 * (num_model + 2) + 10 * (num_model + 1), 300 * 8 + 70, 3))  # eight samples
    fig_img.fill(255)

    count = 0
    for item in sample_list:
        if item == '.DS_Store': continue
        for idx in range(num_model + 2):
            if idx == '.DS_Store': continue
            print(idx)
            pthCurr = os.path.join(pth_sal_img[idx], item)
            if idx == 0: pthCurr = pthCurr[:-4] + '.jpg'
            imgCurr = cv2.imread(pthCurr)
            imgCurr = cv2.resize(imgCurr, (300, 300), interpolation=cv2.INTER_AREA)
            if idx == 0 and count == 0:
                fig_img[:300, :300, :] = imgCurr
            elif idx == 0 and count != 0:
                fig_img[:300, count * (10 + 300): count * (10 + 300) + 300, :] = imgCurr
            elif idx != 0 and count == 0:
                fig_img[idx * (10 + 300): idx * (10 + 300) + 300, :300, :] = imgCurr
            else:
                fig_img[idx * (10 + 300): idx * (10 + 300) + 300,
                count * (10 + 300): count * (10 + 300) + 300, :] = imgCurr

        count += 1
        print('count' + '  ' + str(count))

    cv2.imwrite('fig_supp.png', fig_img)

def LFSOD_fig_qua():
    num_sample = 4
    num_model = 7
    sample_list = os.listdir(os.getcwd() + '/LFSOD_sample/DUT-LF/')
    sample_list.sort(key=lambda x: x[:-4])

    pth_gt = os.getcwd() + '/fig_qua/GT/DUT-LF/'
    pth_img = os.getcwd() + '/fig_qua/RGB/DUT-LF/'
    pth_sal_img = [pth_img,
                   pth_gt,
                   os.getcwd() + '/fig_qua/ours/DUT-LF/',
                   os.getcwd() + '/fig_qua/ERNetT/DUT-LF/',
                   os.getcwd() + '/fig_qua/ERNetS/DUT-LF/',
                   os.getcwd() + '/fig_qua/S2MA/DUT-LF/',
                   os.getcwd() + '/fig_qua/D3Net/DUT-LF/',
                   os.getcwd() + '/fig_qua/LFS/DUT-LF/',
                   os.getcwd() + '/fig_qua/DILF/DUT-LF/'
                   ]

    fig_img = np.zeros((300 * num_sample + 10 * (num_sample - 1), 450 * (num_model + 2) + 10 * (num_model + 1), 3))
    fig_img.fill(255)

    count = 0
    for item in sample_list:
        if item == '.DS_Store': continue
        for idx in range(num_model + 2):
            if idx == '.DS_Store': continue
            print(idx)
            pthCurr = os.path.join(pth_sal_img[idx], item)
            if idx == 0: pthCurr = pthCurr[:-4] + '.jpg'
            imgCurr = cv2.imread(pthCurr)
            imgCurr = cv2.resize(imgCurr, (450, 300), interpolation=cv2.INTER_AREA)
            if idx == 0 and count == 0:
                fig_img[:300, :450, :] = imgCurr
            elif idx == 0 and count != 0:
                fig_img[count * (10 + 300): count * (10 + 300) + 300, :450, :] = imgCurr
            elif idx != 0 and count == 0:
                fig_img[:300, idx * (10 + 450): idx * (10 + 450) + 450, :] = imgCurr
            else:
                fig_img[count * (10 + 300): count * (10 + 300) + 300,
                idx * (10 + 450): idx * (10 + 450) + 450,
                :] = imgCurr

        count += 1
        print('count' + '  ' + str(count))

    cv2.imwrite('fig_qua.png', fig_img)

def LFSOD_fig_ab():
    SIZE = 300
    num_sample = 3
    num_model = 4
    sample_list = os.listdir(os.getcwd() + '/LFSOD_sample/DUT-LF/')
    sample_list.sort(key=lambda x: x[:-4])

    sample_list = ['1564.png', '0119.png', '0155.png']

    pth_gt = os.getcwd() + '/fig_ab/GT/DUT-LF/'
    pth_img = os.getcwd() + '/fig_ab/RGB/DUT-LF/'
    pth_sal_img = [pth_img,
                   pth_gt,
                   os.getcwd() + '/fig_ab/no0/DUT-LF/',
                   os.getcwd() + '/fig_ab/no1/DUT-LF/',
                  # os.getcwd() + '/fig_ab/no2/DUT-LF/',
                   os.getcwd() + '/fig_ab/no3/DUT-LF/',
                  # os.getcwd() + '/fig_ab/no4/DUT-LF/',
                  # os.getcwd() + '/fig_ab/no5/DUT-LF/',
                   os.getcwd() + '/fig_ab/ours/DUT-LF/'
                   ]

    fig_img = np.zeros((300 * num_sample + 10 * (num_sample - 1), SIZE * (num_model + 2) + 10 * (num_model + 1), 3))
    fig_img.fill(255)

    count = 0
    for item in sample_list:
        if item == '.DS_Store': continue
        for idx in range(num_model + 2):
            if idx == '.DS_Store': continue
            print(idx)
            pthCurr = os.path.join(pth_sal_img[idx], item)
            if idx == 0: pthCurr = pthCurr[:-4] + '.jpg'
            imgCurr = cv2.imread(pthCurr)
            imgCurr = cv2.resize(imgCurr, (SIZE, 300), interpolation=cv2.INTER_AREA)
            if idx == 0 and count == 0:
                fig_img[:300, :SIZE, :] = imgCurr
            elif idx == 0 and count != 0:
                fig_img[count * (10 + 300): count * (10 + 300) + 300, :SIZE, :] = imgCurr
            elif idx != 0 and count == 0:
                fig_img[:300, idx * (10 + SIZE): idx * (10 + SIZE) + SIZE, :] = imgCurr
            else:
                fig_img[count * (10 + 300): count * (10 + 300) + 300,
                idx * (10 + SIZE): idx * (10 + SIZE) + SIZE,
                :] = imgCurr

        count += 1
        print('count' + '  ' + str(count))

    cv2.imwrite('fig_qua.png', fig_img)


if __name__ == '__main__':
    LFSOD_fig_ab()
    #LFSOD_fig_qua()
    #LFSOD_fig_supp()
    #objStt()
    #delete_mask()
    #VideoToImg()
    #demoMsk()
    #check_new_av()
    #frm_rename()
    #av_new_mask()
    #key_frame_extraction()
    #av_annotation()
    #demo()
    #AnnotationPrep2()  # get new instance map
    #AnnotationPrep()
    #KeySelect()
    #ReOrder()

# kernelx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
           # kernely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
           # msk_gx = cv2.filter2D(msk, cv2.CV_8U, kernelx)
           # msk_gy = cv2.filter2D(msk, cv2.CV_8U, kernely)
           # msk_edge = msk_gx + msk_gy

    #cv2.imwrite('debug_msk.png', msk)
    #cv2.imwrite('debug_obj.png', obj)
    #cv2.imwrite('debug_edge.png', msk_edge)
    #cv2.imwrite('fix.png', trs_fix_map)
    #cv2.imwrite('fix_mark.png', trs_fix_map_mark)

# palette (cv2: BGR)
# RGB_ins1 = [0, 0, 128]
# RGB_ins2 = [0, 128, 0]
# RGB_ins3 = [0, 128, 128]
# RGB_ins4 = [128, 0, 0]
# RGB_ins5 = [128, 0, 128]
# RGB_ins6 = [128, 128, 0]
# RGB_ins7 = [128, 128, 128]
# RGB_ins8 = [0, 0, 64]
# RGB_ins9 = [0, 0, 192]
# Palette_list = [RGB_ins1, RGB_ins2, RGB_ins3, RGB_ins4, RGB_ins5, RGB_ins6, RGB_ins7,
#                RGB_ins8, RGB_ins9]