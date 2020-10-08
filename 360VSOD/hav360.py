import numpy as np
import cv2
import os

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
    pth = os.getcwd() + '/_-4fxKBGthpaw/'
    frame_list = os.listdir(pth)
    frame_list.sort(key=lambda x: x[:-4])
    for frm in frame_list:
        if frm == '.DS_Store': continue
        old_pth = os.path.join(pth, frm)
        order = format(str(int(frm[:-4]) - 121), '0>5s')
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


if __name__ == '__main__':
    AnnotationPrep()
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