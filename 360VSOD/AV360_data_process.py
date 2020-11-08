import cv2
import os

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

def saliency_threshold():
    print()

def re_order():
    pth = os.getcwd() + '/_-1An41lDIJ6Q/'
    frame_list = os.listdir(pth)
    frame_list.sort(key=lambda x: x[:-4])
    for frm in frame_list:
        if frm == '.DS_Store': continue
        old_pth = os.path.join(pth, frm)
        order = format(str(int(frm[:-4]) - 112), '0>5s')
        new_pth = os.getcwd() + '/key/' + order + '.png'
        os.rename(old_pth, new_pth)

def av_overlaid():
    Imgs_pth = os.getcwd() + '/frame_total/'
    Sals_pth = os.getcwd() + '/saliency/'
    video_list = os.listdir(Imgs_pth)
    count = 0
    for vid in video_list:
        if vid == '.DS_Store': continue
        vid_pth = os.path.join(Imgs_pth, vid)
        frm_list = os.listdir(vid_pth)
        frm_list.sort(key=lambda x: x[:-4])
        for img in frm_list:
            if img == '.DS_Store': continue
            if int(img[:-4]) % 6 == 0:
                img_pth = os.path.join(vid_pth, img)
                sal_pth = os.path.join(Sals_pth, vid, img)
                img_curr = cv2.imread(img_pth)
                img_curr = cv2.resize(img_curr, (600, 300))
                sal_curr = cv2.imread(sal_pth)
                ret, sal_curr = cv2.threshold(sal_curr, 127, 255, cv2.THRESH_BINARY)
                sal_curr[:, :, :-1] = 0
                oly_curr = cv2.addWeighted(img_curr, 0.8, sal_curr, 1, 0)

                oly_dir = os.path.join(os.getcwd() + '/saliency_overlaid/', vid)
                if not os.path.exists(oly_dir): os.makedirs(oly_dir)
                oly_pth = os.path.join(oly_dir, img)
                cv2.imwrite(oly_pth, oly_curr)
        count += 1
        print(count)

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

def av_new_mask():
    msk_1_pth = os.getcwd() + '/mask_av_1/_-0suxwissusc'
    msk_2_pth = os.getcwd() + '/mask_av_2/_-0suxwissusc'
    new_pth = os.getcwd() + '/new/'
    msk_list = os.listdir(msk_2_pth)
    msk_list.sort(key=lambda x: x[:-4])
    ori_list = os.listdir(msk_1_pth)
    ori_list.sort(key=lambda x: x[:-4])
    count = 1
    for item in ori_list:
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

def oly_three():
    frames_pth = os.getcwd() + '/frame_key/_-4fxKBGthpaw'
    sals_pth = os.getcwd() + '/saliency/_-4fxKBGthpaw'
    inss_pth = os.getcwd() + '/mask_av_1/_-4fxKBGthpaw'
    fin_pth = os.getcwd() + '/new/'
    frm_list = os.listdir(frames_pth)
    frm_list.sort(key=lambda x: x[:-4])
    count = 0
    for frm in frm_list:
        frm_pth = os.path.join(frames_pth, frm)
        img = cv2.imread(frm_pth)
        H = img.shape[0]
        W = img.shape[1]
        sal_pth = os.path.join(sals_pth, frm)
        sal = cv2.imread(sal_pth)
        ret, sal = cv2.threshold(sal, 127, 255, cv2.THRESH_BINARY)
        sal = cv2.resize(sal, (W, H))
        ins_pth = os.path.join(inss_pth, frm)
        ins = cv2.imread(ins_pth)
        oly = cv2.addWeighted(img, 1, sal, 0.5, 0)
        oly = cv2.addWeighted(oly, 1, ins, 0.8, 0)
        cv2.imwrite(os.path.join(fin_pth, frm), oly)
        count += 1
        print(count)


if __name__ == '__main__':
    oly_three()
    #av_new_mask()
    #frm_rename()
    #av_overlaid()
    #re_order()
    #saliency_threshold()
    #VideoToImg()
