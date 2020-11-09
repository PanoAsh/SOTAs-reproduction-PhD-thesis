import numpy as np
import cv2
import os

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
    msks_pth = os.getcwd() + '/mask_instance/'
    fixs_pth = os.getcwd() + '/saliency_key_frame_visual/'
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
    msk_1_pth = os.getcwd() + '/mask_av_1/_-4SilhsTuDU0'
    msk_2_pth = os.getcwd() + '/mask_av_2/_-4SilhsTuDU0'
    new_pth = os.getcwd() + '/new/'
    msk_list = os.listdir(msk_2_pth)
    msk_list.sort(key=lambda x: x[:-4])
    ori_list = os.listdir(msk_1_pth)
    ori_list.sort(key=lambda x: x[:-4])
    count = 1
    for item in ori_list:
        if item == '.DS_Store': continue
        ori_pth = os.path.join(msk_1_pth, item)
        ori = cv2.imread(ori_pth)
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
            msk = cv2.imread(item_pth)
            msk = cv2.resize(msk, (600, 300))
            oly = cv2.addWeighted(msk, 0.5, sal, 0.5, 0)
            oly_dir = os.path.join(check_new_pth, msks)
            if not os.path.exists(oly_dir): os.makedirs(oly_dir)
            oly_pth = os.path.join(oly_dir, item)
            cv2.imwrite(oly_pth, oly)
        count += 1
        print(count)


if __name__ == '__main__':
    check_new_av()
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