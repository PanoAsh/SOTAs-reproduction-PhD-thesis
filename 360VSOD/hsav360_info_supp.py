import numpy as np
import cv2
import os

video_length = [
27.8,
25.8,
27.0,
16.4,
27.2,
21.2,
23.2,
22.8,
31.0,
28.838709677419356,
31.764705882352942,
24.8,
26.6,
21.12,
24.8,
22.470588235294116,
24.0,
31.2,
29.2,
28.06451612903226,
34.4,
36.5,
26.903225806451612,
24.774193548387096,
30.6,
26.903225806451612,
11.764705882352942,
15.4,
21.923076923076923,
20.4,
10.4,
21.9,
27.4,
29.76923076923077,
28.8,
32.0,
31.741935483870968,
30.4,
27.4,
29.8,
25.8,
28.615384615384617,
31.8,
24.8,
16.258064516129032,
21.7,
28.8,
25.8,
19.176470588235293,
18.24,
40.705882352941174,
28.8,
27.677419354838708,
14.9,
23.529411764705884,
20.0,
29.8,
35.76470588235294,
19.8,
29.8,
28.8,
28.615384615384617,
32.0,
29.8,
29.8,
23.307692307692307,
29.6,
34.75,
24.0
]

def video_mean_std():
    mean = np.mean(video_length)
    std = np.std(video_length)
    print(mean)
    print(std)

def fig_1():
    fixs_w_sound_pth = os.getcwd() + '/fig_1_0910/fix_w_s/'
    fixs_wo_sound_pth = os.getcwd() + '/fig_1_0910/fix_wo_s/'
    imgs_pth = os.getcwd() + '/fig_1_0910/img/'
    sound_pth = os.getcwd() + '/fig_1_0910/sound/'

    count = 0
    imgs_list = os.listdir(imgs_pth)
    imgs_list.sort(key=lambda x: x[:-4])
    fixs_w_list = os.listdir(fixs_w_sound_pth)
    fixs_w_list.sort(key=lambda x: x[:-4])
    fixs_wo_list = os.listdir(fixs_wo_sound_pth)
    fixs_wo_list.sort(key=lambda x: x[:-4])
    for name in imgs_list:
        img = cv2.imread(os.path.join(imgs_pth, name))
        img = cv2.resize(img, (600, 300))

        fix_w_s = cv2.imread(os.path.join(fixs_w_sound_pth, name))
        fix_w_s = cv2.GaussianBlur(fix_w_s, (45, 45), 10)
        fix_w_s = cv2.normalize(fix_w_s, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8UC1)
        #fix_w_s[:, :, :-1] = 0
        fix_w_s = cv2.applyColorMap(fix_w_s, cv2.COLORMAP_HOT)

        fix_wo_s = cv2.imread(os.path.join(fixs_wo_sound_pth, name))
        fix_wo_s = cv2.GaussianBlur(fix_wo_s, (45, 45), 10)
        fix_wo_s = cv2.normalize(fix_wo_s, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8UC1)
        fix_wo_s = cv2.applyColorMap(fix_wo_s, cv2.COLORMAP_OCEAN)
       # fix_wo_s[:, :, 1:] = 0

        heat = cv2.addWeighted(fix_wo_s, 1.2, fix_w_s, 1.2, 0)

        overlay = cv2.addWeighted(img, 0.4, heat, 0.6, 0)
        cv2.imwrite(os.getcwd() + '/overlay_fix/' + name, overlay)

        sound = cv2.imread(os.path.join(sound_pth, name))
        sound = cv2.resize(sound, (600, 300))
        sound = cv2.normalize(sound, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8UC1)
        sound = cv2.applyColorMap(sound, cv2.COLORMAP_PINK)

        overlay_2 = cv2.addWeighted(img, 0.4, sound, 0.6, 0)
        cv2.imwrite(os.getcwd() + '/overlay_sound/' + name, overlay_2)

        count += 1
        print(str(count) + ' images processed.')

def fig_2_step_1():
     img_pth = os.getcwd() + '/fig_2_0911/P3/imgs/'
     fix_pth = os.getcwd() + '/fig_2_0911/P3/fixations_ws/'

     count = 0
     imgs_list = os.listdir(img_pth)
     imgs_list.sort(key=lambda x: x[:-4])
     fixs_list = os.listdir(fix_pth)
     fixs_list.sort(key=lambda x: x[:-4])
     for idx in range(len(imgs_list)):
         img = cv2.imread(os.path.join(img_pth, imgs_list[idx]))
         img = cv2.resize(img, (600, 300))

         fix = cv2.imread(os.path.join(fix_pth, fixs_list[idx]))
         fix = cv2.GaussianBlur(fix, (45, 45), 10)
         fix = cv2.normalize(fix, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_8UC1)
         # fix_w_s[:, :, :-1] = 0
         fix = cv2.applyColorMap(fix, cv2.COLORMAP_HOT)
         overlay = cv2.addWeighted(img, 0.4, fix, 0.6, 0)
         cv2.imwrite(os.getcwd() + '/overlay_cap/' + imgs_list[idx], overlay)

         count += 1
         print(str(count) + ' images processed.')

def fig_2_step_2():
    fig_img = np.zeros((256 * 2 + 10, 512 * 7 + 60, 3))
    fig_img.fill(255)

    overlay_pth = os.getcwd() + '/fig_2_0911/P3/overlay_cap_p3/'
    obj_pth = os.getcwd() + '/fig_2_0911/P3/objs/'
    fig = 'fig_cap_3.png'

    line1_list = os.listdir(overlay_pth)
    line1_list.sort(key=lambda x: x[:-4])
    line2_list = os.listdir(obj_pth)
    line2_list.sort(key=lambda x: x[:-4])
    for idx in range(7):
        oly = cv2.imread(os.path.join(overlay_pth, line1_list[idx]))
        oly = cv2.resize(oly, (512, 256), interpolation=cv2.INTER_AREA)
        obj = cv2.imread(os.path.join(obj_pth, line2_list[idx]))
        obj = cv2.resize(obj, (512, 256), interpolation=cv2.INTER_AREA)
        if idx == 0:
            fig_img[:256, :512, :] = oly
            fig_img[266:, :512, :] = obj
        else:
            fig_img[:256, (idx * (512 + 10)):(idx * (512 + 10) + 512), :] = oly
            fig_img[266:, (idx * (512 + 10)):(idx * (512 + 10) + 512), :] = obj

            cv2.imwrite(fig, fig_img)

def attr():
    MP = 0
    OC = 0
    LS = 0
    OV = 0
    MB = 0
    GD = 0
    FS = 0
    # 6+5+4+3+2+1 = 21
    num_1_2 = 0
    num_1_3 = 0
    num_1_4 = 0
    num_1_5 = 0
    num_1_6 = 0
    num_1_7 = 0

    num_2_3 = 0
    num_2_4 = 0
    num_2_5 = 0
    num_2_6 = 0
    num_2_7 = 0

    num_3_4 = 0
    num_3_5 = 0
    num_3_6 = 0
    num_3_7 = 0

    num_4_5 = 0
    num_4_6 = 0
    num_4_7 = 0

    num_5_6 = 0
    num_5_7 = 0

    num_6_7 = 0

    attr_list = [[1 ,    1    , 0   ,  0 ,    0   ,  0   ,  0 ],
                [ 0,     0  ,   1  ,   1 ,    1  ,   0  ,   0],
                [ 0 ,    0   ,  1  ,   0    , 1  ,   1   ,  0 ],
                [ 0 ,    1 ,    1  ,   0,     0 ,    0 ,    0 ],
                [ 0  ,   0 ,    1  ,   0   ,  1   ,  0    , 0 ],
                [ 0     ,1 ,    1 ,    1 ,    1,     0,     0 ],
                [ 1   ,  1    , 1   ,  1     ,1    , 1     ,0 ],
                [ 1    , 0 ,    0 ,    0   ,  0 ,    1 ,    0 ],
                [ 0  ,   1 ,    0,     0    , 0    , 0    , 0 ],
                [ 0   ,  1  ,   0 ,    0 ,    0    , 1,     0 ],
                [ 0    , 0  ,   0  ,   0  ,   1  ,   1 ,    0 ],
                [ 0 ,    0   ,  0   ,  0   ,  0   ,  1  ,   0 ],
                [ 1  ,   0    , 0    , 0    , 0    , 1   ,  0 ],
                [ 0   ,  1,     1     ,0     ,1     ,0    , 0 ],
                [ 0    , 1 ,    1  ,   1,     0,     1     ,0 ],
                [ 0,     0  ,   1,     0 ,    0 ,    0,     1 ],
                [ 0 ,    1   ,  0 ,    0  ,   1  ,   1 ,    0 ],
                [ 0  ,   1    , 0  ,   0   ,  0   ,  0  ,   0 ],
                [ 0   ,  1,     1   ,  0    , 1    , 0   ,  1 ],
                [ 0     ,0 ,    0    , 0     ,0     ,1    , 0 ],
                [ 1    , 0  ,   1     ,1,     1,     1     ,0 ],
                [ 0,     1   ,  0     ,0 ,    1 ,    1,     0 ],
                [ 0 ,    0    , 1,     0  ,   1  ,   1 ,    0 ],
                [ 0  ,   0     ,1 ,    1    , 0   ,  0  ,   1 ],
                [ 0   ,  0,     1  ,   0   ,  0    , 0    , 1 ],
                [ 1    , 0 ,    0   ,  0     ,0     ,0   ,  0 ],
                [ 0  ,   1  ,   1    , 0     ,1     ,0     ,1 ],
                [ 0  ,   0   ,  0     ,1,     0     ,0,     0 ],
                [ 0  ,   1    , 0     ,0,     0,     0 ,    0 ],
                [ 0  ,   0     ,0 ,    0  ,   0 ,    0  ,   0 ],
                [ 0  ,   1,     0,     0  ,   0  ,   1   ,  0 ],
                [ 0  ,   1 ,    0 ,    1 ,    1   ,  1    , 0 ],
                [ 0  ,   0  ,   0  ,   1  ,   1    , 1     ,0 ],
                [ 0   ,  0   ,  0   ,  0   ,  0     ,1 ,    0 ],
                [ 0   ,  0    , 1    , 0    , 1,     0,     0 ],
                [ 0   ,  1     ,1     ,0     ,0 ,    0,     0 ],
                [  0   ,  1,     1,     0  ,   1 ,    0,     0 ],
                [  0   ,  0,     1 ,    1 ,    1 ,    0 ,    0 ] ,
                [ 0   ,  1 ,    1   ,  0   ,  0   ,  0   ,  0  ]  ,
                [  1   ,  0 ,    1   ,  0   ,  1  ,   0   ,  0 ]   ,
                [  0   ,  0  ,   1    , 0    , 1   ,  0    , 0 ]     ,
                [  0   ,  1   ,  0,     1,     1    , 0     ,0 ]    ,
                [  1   ,  1    , 1 ,    1    , 1     ,1,     0 ]     ,
                [  0   ,  0     ,0   ,  0,     0,     0 ,    0 ]     ,
                [  0   ,  1 ,    1    , 0,     1 ,    0  ,   0 ]     ,
                [  0   ,  0,     0     ,0 ,    0  ,   1   ,  0 ]    ,
                [  0   ,  1,     1 ,    0  ,   0   ,  0    , 0 ]     ,
                [  0   ,  1 ,    0  ,   0     ,1    , 1     ,0 ]    ,
                [  0   ,  0  ,   0   ,  0   ,  0     ,0,     0 ]    ,
                [  0   ,  1   ,  0    , 1    , 1,     0 ,    0 ]    ,
                [  0   ,  0    , 1,     0,     0 ,    0  ,   1 ]   ,
                [  0   ,  0,     0 ,    0 ,    1  ,   1   ,  0 ]   ,
                [  0    , 0 ,    0  ,   1  ,   0   ,  1    , 0 ]    ,
                [  0   ,  1  ,   0   ,  1   ,  1    , 0     ,0 ]    ,
                [  0  ,   0   ,  0    , 0    , 1     ,1,     0 ]    ,
                [  0   ,  1    , 1     ,0,     1 ,    0  ,   1 ]    ,
                [  0  ,   0     ,0,     1 ,    0 ,    1  ,   0 ]    ,
                [  1  ,   1,     0 ,    0  ,   0  ,   0   ,  0 ]    ,
                [  0  ,   0,     0  ,   0   ,  1   ,  0    , 0 ]    ,
                [  0   ,  0 ,    0   ,  0    , 0    , 0     ,0 ]    ,
                [  1  ,   1  ,   0    , 0     ,1     ,0 ,    0 ]    ,
                [  0  ,   1   ,  0,     0,     0 ,    0  ,   1 ]    ,
                [  0    , 1    , 0 ,    0 ,    0  ,   1   ,  0 ]    ,
                [  1  ,   1     ,0  ,   0  ,   0   ,  0    , 1 ]    ,
                [  1   ,  1 ,    0   ,  0   ,  0    , 0     ,0 ]    ,
                [  1   ,  1  ,   1    , 0    , 1     ,0,     0 ]    ,
                [  1   ,  1   ,  0     ,0     ,1,     0 ,    0 ]    ,
                [  0   ,  1    , 0 ,    0,     1 ,    0  ,   1 ]    ,
                [  1   ,  0     ,1  ,   0 ,    1  ,   0   ,  0 ]    ,
                ]
    for idx in range(69):
        if attr_list[idx][0] == 1:
            MP += 1
        if attr_list[idx][1] == 1:
            OC += 1
        if attr_list[idx][2] == 1:
            LS += 1
        if attr_list[idx][3] == 1:
            OV += 1
        if attr_list[idx][4] == 1:
            MB += 1
        if attr_list[idx][5] == 1:
            GD += 1
        if attr_list[idx][6] == 1:
            FS += 1


        if attr_list[idx][0] == 1 and attr_list[idx][1] == 1:
            num_1_2 += 1
        if attr_list[idx][0] == 1 and attr_list[idx][2] == 1:
            num_1_3 += 1
        if attr_list[idx][0] == 1 and attr_list[idx][3] == 1:
            num_1_4 += 1
        if attr_list[idx][0] == 1 and attr_list[idx][4] == 1:
            num_1_5 += 1
        if attr_list[idx][0] == 1 and attr_list[idx][5] == 1:
            num_1_6 += 1
        if attr_list[idx][0] == 1 and attr_list[idx][6] == 1:
            num_1_6 += 1

        if attr_list[idx][1] == 1 and attr_list[idx][2] == 1:
            num_2_3 += 1
        if attr_list[idx][1] == 1 and attr_list[idx][3] == 1:
            num_2_4 += 1
        if attr_list[idx][1] == 1 and attr_list[idx][4] == 1:
            num_2_5 += 1
        if attr_list[idx][1] == 1 and attr_list[idx][5] == 1:
            num_2_6 += 1
        if attr_list[idx][1] == 1 and attr_list[idx][6] == 1:
            num_2_7 += 1

        if attr_list[idx][2] == 1 and attr_list[idx][3] == 1:
            num_3_4 += 1
        if attr_list[idx][2] == 1 and attr_list[idx][4] == 1:
            num_3_5 += 1
        if attr_list[idx][2] == 1 and attr_list[idx][5] == 1:
            num_3_6 += 1
        if attr_list[idx][2] == 1 and attr_list[idx][6] == 1:
            num_3_7 += 1

        if attr_list[idx][3] == 1 and attr_list[idx][4] == 1:
            num_4_5 += 1
        if attr_list[idx][3] == 1 and attr_list[idx][5] == 1:
            num_4_6 += 1
        if attr_list[idx][3] == 1 and attr_list[idx][6] == 1:
            num_4_7 += 1

        if attr_list[idx][4] == 1 and attr_list[idx][5] == 1:
            num_5_6 += 1
        if attr_list[idx][4] == 1 and attr_list[idx][6] == 1:
            num_5_7 += 1

        if attr_list[idx][5] == 1 and attr_list[idx][6] == 1:
            num_6_7 += 1

    print()

def qlt_show():
    num_model = 20
    sample_list = os.listdir(os.getcwd() + '/GT_fig/')
    sample_list_ordered = [sample_list[4], sample_list[6], sample_list[3], sample_list[7],
                         sample_list[1], sample_list[5], sample_list[8], sample_list[0]]

    pth_gt = os.getcwd() + '/GT'
    pth_img = os.getcwd() + '/Img_test'
    pth_sal_img = [pth_img,
                   os.getcwd() + '/fine_tune/ft_aadfnet_e7',
                   os.getcwd() + '/fine_tune/ft_poolnet_e10',
                   os.getcwd() + '/fine_tune/ft_cpd_e7',
                   os.getcwd() + '/fine_tune/ft_basnet_e4',
                   os.getcwd() + '/fine_tune/ft_egnet_e8',
                   os.getcwd() + '/fine_tune/ft_scrn_e8',
                   os.getcwd() + '/fine_tune/ft_u2net_e8',
                   os.getcwd() + '/fine_tune/ft_ras_e6',
                   os.getcwd() + '/fine_tune/ft_f3net_e2',
                   os.getcwd() + '/fine_tune/ft_gcpanet_e7',
                   os.getcwd() + '/fine_tune/ft_scrsal_e10',
                   os.getcwd() + '/fine_tune/ft_minet_e5',
                   os.getcwd() + '/fine_tune/ft_ldf_e2',
                   os.getcwd() + '/fine_tune/ft_csnet_e10',
                   os.getcwd() + '/fine_tune/ft_csf_e10',
                   os.getcwd() + '/fine_tune/ft_rcrnet_e7',
                   os.getcwd() + '/fine_tune/ft_cosnet_e4',
                   os.getcwd() + '/fine_tune/ft_ssav_e2',
                   os.getcwd() + '/fine_tune/ft_dds_e1',
                   os.getcwd() + '/fine_tune/davpnet_e7',
                   pth_gt]

    fig_img = np.zeros((256 * (num_model + 2) + 10 * (num_model + 1), 512 * 8 + 70, 3)) # eight samples
    fig_img.fill(255)

    count = 0
    for item in sample_list_ordered:
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

    cv2.imwrite('fig_ft.png', fig_img)



if __name__ == "__main__":
    #curve_show()
    #attr()
    qlt_show()
    #fig_2_step_2()
    #fig_2_step_1()
    #fig_1()
    #video_mean_std()

    # chart.data = [
    #     {
    # from: "MP", to: "OC", value: 9},
    # {
    # from: "MP", to: "LS", value: 6},
    # {
    # from: "MP", to: "OV", value: 3},
    # {
    # from: "MP", to: "MB", value: 8},
    # {
    # from: "MP", to: "GR", value: 6},
    # {
    # from: "OC", to: "LS", value: 15},
    # {
    # from: "OC", to: "OV", value: 8},
    # {
    # from: "OC", to: "MB", value: 20},
    # {
    # from: "OC", to: "GR", value: 10},
    # {
    # from: "OC", to: "FS", value: 6},
    # {
    # from: "LS", to: "OV", value: 8},
    # {
    # from: "LS", to: "MB", value: 20},
    # {
    # from: "LS", to: "GR", value: 6},
    # {
    # from: "LS", to: "FS", value: 7},
    # {
    # from: "OV", to: "MB", value: 11},
    # {
    # from: "OV", to: "GR", value: 8},
    # {
    # from: "OV", to: "FS", value: 1},
    # {
    # from: "MB", to: "GR", value: 13},
    # {
    # from: "MB", to: "FS", value: 4}, ];