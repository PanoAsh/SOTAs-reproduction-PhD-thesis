import os
import cv2
import numpy as np
from scipy import ndimage
import stts_utils as utils


# parameters (to generate overlays)
genOverlay = 0

Width = 3840
Height= 2048
Fps = 30

seq2frm = 0
frm2oly_2 = 0
frm2oly = 0
oly2vid = 0

# parameters (vr_scene)
vr_scene = 0
seq_height = 300
seq_width = 600

# auto overlay
auto_oly = 0
pixel_shift_lat = 18
pixel_shift_lon = 0
bool_rescale = False
bool_shift = False

# video check
bool_frm2vid = False

# auto overlay (final version)
frm_interval = 1
bool_shift_scale = True
pixel_shift = 20
pixel_reserve = 15
pixel_up = 10
pole_cut = 30

bool_numFrm = False
bool_SOD_crop = False

class PanoVSOD_stts():
    def __init__(self):
        self.path_seq = os.getcwd() + '/stimulis/'
        self.path_sor = os.getcwd() + '/source_videos/'
        self.path_frm = os.getcwd() + '/frames/' # one by one
        self.path_syn = os.getcwd() + '/synthetic_video.avi'
        self.path_oly = os.getcwd() + '/overlay_salmap/w/'
        self.path_fix = os.getcwd() + '/fixations/' # one by one
        self.path_FM = os.getcwd() + '/fixation_maps/'

    def num_frames_count(self):
        seq_list = os.listdir(self.path_sor)

        f = open(os.getcwd() + '/360vSOD_stts.txt', 'w')
        frames_num = []
        duration_num = []
        count = 0
        for seq in seq_list:
            if seq.endswith('.mp4'):
                seq_path = os.path.join(os.path.abspath(self.path_sor), seq)
                cap = cv2.VideoCapture(seq_path)
                frames_num.append(int(cap.get(7)))
                duration_num.append(int(cap.get(7) / (cap.get(5))))
                line = seq[:-4] + '    ' + str(frames_num[count]) + '    ' + str(duration_num[count]) + '    ' + '\n'
                f.write(line)
                count += 1
                print(" {} videos processed".format(count))

        total_frames = np.sum(frames_num)
        total_duration = np.sum(duration_num)
        f.write(str(total_frames) + '\n')
        f.write(str(total_duration))
        f.close()

        return total_frames

    def VideoToImg(self):
        seq_list = os.listdir(self.path_seq)

        count = 1
        for seq in seq_list:
            if seq.endswith('.mp4'):
                seq_path = os.path.join(os.path.abspath(self.path_seq), seq)
                frm_path = os.path.join(os.path.abspath(self.path_frm), seq)
                cap = cv2.VideoCapture(seq_path)
                frames_num = int(cap.get(7))
                countF = 0
                for i in range(frames_num):
                    ret, frame = cap.read()
                    #frame = cv2.resize(frame, (Width, Height))
                    cv2.imwrite(frm_path[:-4] + '_' +
                                format(str(countF), '0>6s') + '.png',
                                frame)
                    countF += 1
                    print(" {} frames processed".format(countF))
                print(" {} videos processed".format(count))
                count += 1

    def ImgToVideo(self): # to generate the fixation overlays as guidance for salient object annotation
        oly_list = os.listdir(self.path_oly)

        count_oly = 0
        for oly in oly_list:
            oly_vid = cv2.VideoWriter(os.getcwd() + '/' + oly + '.avi', 0, 1, (600, 300))

            oly_path = self.path_oly + oly
            oly_sub_list =  os.listdir(oly_path)
            oly_sub_list.sort(key=lambda x: x[:-4])

            for idx in range(len(oly_sub_list)):
                frm_path = oly_path + '/' + oly_sub_list[idx]
                oly_vid.write(cv2.imread(frm_path))
                print("{} writen".format(idx + 1))

            oly_vid.release()
            count_oly += 1
            print("{} videos processed.".format(count_oly))

    def fixation_overlay(self):
        frm_list = os.listdir(self.path_frm)
        frm_list.sort(key=lambda x: x[:-4])
        fix_list = os.listdir(self.path_fix)
        fix_list.sort(key=lambda x: x[:-4])

        for idx in range(len(frm_list)):
            fix_path = os.path.join(os.path.abspath(self.path_fix), fix_list[idx])
            frm_path = os.path.join(os.path.abspath(self.path_frm), frm_list[idx])
            oly_path = os.path.join(os.path.abspath(self.path_oly), frm_list[idx])

            fix = np.load(fix_path)

            fix = fix[:, :, np.newaxis]
            fixation = []
            for i in range(3):
                fixation.append(fix)
            fixation = np.concatenate(fixation, axis=2)
            fixation = cv2.resize(fixation, (Width, Height))
            fixation = cv2.normalize(fixation, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            fixation = cv2.applyColorMap(fixation, cv2.COLORMAP_JET)

            image = cv2.imread(frm_path)

            overlay = cv2.addWeighted(image, 1, fixation, 1, 0)
            cv2.imwrite(oly_path, overlay)
            print("{} frames processed".format(idx + 1))

    def fixation_overlay_2(self):
        frm_list = os.listdir(self.path_frm)
        frm_list.sort(key=lambda x: x[:-4])
        fix_list = np.load(os.getcwd() + '/003.npy')

        for idx in range(len(frm_list)):
            frm_path = os.path.join(os.path.abspath(self.path_frm), frm_list[idx])
            oly_path = os.path.join(os.path.abspath(self.path_oly), frm_list[idx])

            fix = fix_list[:, :, idx]
            fix = self.gaussian_smooth(fix, 2 * seq_width / 360)
            fix = fix[:, :, np.newaxis]
            fixation = []
            for i in range(3):
                fixation.append(fix)
            fixation = np.concatenate(fixation, axis=2)
            fixation = cv2.resize(fixation, (Width, Height))
            fixation = cv2.normalize(fixation, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            fixation = cv2.applyColorMap(fixation, cv2.COLORMAP_JET)

            image = cv2.imread(frm_path)

            overlay = cv2.addWeighted(image, 1, fixation, 1, 0)
            cv2.imwrite(oly_path, overlay)
            print("{} frames processed".format(idx + 1))

    def gaussian_smooth(self, fix, sigma_y):
        salmap = ndimage.filters.gaussian_filter1d(fix, sigma=sigma_y, mode='wrap', axis=0)

        # In x-direction, we scale the radius of the gaussian kernel the closer we get to the pole
        for row in range(salmap.shape[0]):
            angle = (row / float(salmap.shape[0]) - 0.5) * np.pi
            sigma_x = sigma_y / (np.cos(angle) + 1e-3)
            salmap[row, :] = ndimage.filters.gaussian_filter1d(salmap[row, :], sigma=sigma_x, mode='wrap')

        salmap /= float(np.sum(salmap))

        return salmap

    def vrs_txt_rename(self):
        txt_path = os.getcwd() + '/txt_fixations/'
        sbj_list = os.listdir(os.getcwd() + '/source_fixations/')
        sbj_list.sort(key=lambda x: x)

        count_sbj = 0
        for sbj in sbj_list:
            sbj_path = os.path.join(os.getcwd(), 'source_fixations', sbj)
            txt_list = os.listdir(sbj_path)
            txt_list.sort(key=lambda x: x)

            for item in txt_list:
                if item.endswith('.txt'):
                    item_list = item.split('_')
                    new_item = item_list[0] + '_' + sbj + '_' + item_list[1]
                    src = os.path.join(sbj_path, item)
                    dst = os.path.join(txt_path, new_item + '.txt')
                    try:
                        os.rename(src, dst)
                        print('converting %s to %s ...' % (src, dst))
                    except:
                        continue

            count_sbj += 1
            print("{} subjects processed.".format(count_sbj))

    def vrs_coor(self):
        seq_list = os.listdir(self.path_sor)
        seq_list.sort(key=lambda x: x[:-4])

        txt_list = os.listdir(os.getcwd() + '/txt_fixations/')
        txt_list.sort(key=lambda x: x)

        count_seq = 0
        for seq in seq_list:
            seq_file = cv2.VideoCapture(self.path_sor + '/' + seq)
            #seq_width = int(seq_file.get(3))
            #seq_height = int(seq_file.get(4))
            seq_frm = int(seq_file.get(7))
            seq_fix = np.zeros((seq_height, seq_width, seq_frm))
            seq_idx = seq[:3]

            count_txt = 0
            for txt in txt_list:
                if txt[:3] == seq_idx:
                    txt_fix = np.genfromtxt(os.path.join(os.getcwd() + '/txt_fixations/' + txt), dtype='str',  delimiter=',')
                    num_fix = txt_fix.shape[0]
                    if num_fix > seq_frm:
                        num_judge = seq_frm
                    else:
                        num_judge = num_fix
                    for idx in range(num_judge):
                        coor_x = int(float(txt_fix[idx, 6]) * seq_width)
                        coor_y = int(float(txt_fix[idx, 7]) * seq_height)
                        coordH = seq_height - coor_y
                        idx_frm = int(txt_fix[idx, 1]) - 1
                        if coordH >= seq_height:
                            continue
                        elif idx_frm >= seq_frm:
                            continue
                        elif coor_x >= seq_width:
                            continue
                        else:
                            seq_fix[coordH, coor_x, idx_frm] = 1

                    count_txt += 1
                    print("{} txt files processed.".format(count_txt))

            np.save(os.getcwd() + '/' + seq[:3] + '.npy', seq_fix)
            count_seq += 1
            print("{} videos processed.".format(count_seq))

    def auto_oly_vid(self):
        file_ids = os.listdir(os.getcwd() + '/test/')

        count_seq = 0
        for id in file_ids:
            fix_wo_path = os.path.join(os.getcwd() + '/fixations_wo_sound/', id)
            fix_wo_files = os.listdir(fix_wo_path)
            fix_wo_files.sort(key=lambda x: x[:-4])
            fix_w_path = os.path.join(os.getcwd() + '/fixations_w_sound/', id)
            fix_w_files = os.listdir(fix_w_path)
            fix_w_files.sort(key=lambda x: x[:-4])

            seq_item = cv2.VideoCapture(os.getcwd() + '/source_videos/' + id + '.mp4')
            seq_width = int(seq_item.get(3))
            seq_height = int(seq_item.get(4))
            seq_fps = int(seq_item.get(5)) + 1
            seq_numFrm = int(seq_item.get(7))

            oly_vid_path = os.getcwd() + '/' + id + '.avi'
            oly_vid = cv2.VideoWriter(oly_vid_path, 0, seq_fps, (seq_width, seq_height))

            count_frm = 0
            for idx in range(seq_numFrm):
                if (idx + 1) % frm_interval == 0:
                    seq_item.set(1, idx)
                    ret, frame = seq_item.read()
                    if ret == False:
                        continue

                    # find the corresponding w/wo sound fixation maps + shifting
                    npy_wo_path = fix_wo_path + '/' + fix_wo_files[idx]
                    fix_wo = np.load(npy_wo_path)
                    npy_w_path = fix_w_path + '/' + fix_w_files[idx]
                    fix_w = np.load(npy_w_path)

                    # overlay two fixation maps to the current key frame + rescale
                    pixel_pad = int((seq_height - seq_width / 2) / 2)

                    # without sound
                    fix_wo = fix_wo[:, :, np.newaxis]
                    fixation_wo = []
                    for i in range(3):
                        fixation_wo.append(fix_wo)
                    fixation_wo = np.concatenate(fixation_wo, axis=2)

                    if bool_shift == True:
                        fixation_wo = np.roll(fixation_wo, pixel_shift_lon, axis=0)  # shifting down
                        fixation_wo[:pixel_shift_lon, :, :] = 0
                        cmp = utils.e2c(fixation_wo, int(600 / 4))
                        cmp[0] = np.roll(cmp[0], -1 * pixel_shift_lat, axis=1)  # shifting left
                        cmp[0][:, (-1 * pixel_shift_lat):, :] = 0
                        cmp[1] = np.roll(cmp[1], pixel_shift_lat, axis=1)  # shifting right
                        cmp[1][:, :pixel_shift_lat, :] = 0
                        cmp[3] = np.roll(cmp[3], int((-1 * pixel_shift_lat) / 2), axis=1)  # shifting left
                        cmp[3][:, int((-1 * pixel_shift_lat) / 2):, :] = 0
                        cmp[4] = np.roll(cmp[4], int((-1 * pixel_shift_lat) / 2), axis=1)  # shifting left
                        cmp[4][:, int((-1 * pixel_shift_lat) / 2):, :] = 0
                        fixation_wo = utils.c2e(cmp, 300, 600)

                    if bool_rescale == True:
                        fix_wo = cv2.resize(fix_wo, (seq_width, int(seq_width / 2))) # resize to normal size (ratio of 1:2)
                        fix_wo = np.pad(fix_wo, ((pixel_pad, pixel_pad), (0, 0)), 'constant') # padding (top, bottom, left, right)
                    else:
                        fixation_wo = cv2.resize(fixation_wo, (seq_width, seq_height))

                    fixation_wo = cv2.normalize(fixation_wo, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    fixation_wo = cv2.applyColorMap(fixation_wo, cv2.COLORMAP_OCEAN)

                    # with sound
                    fix_w = fix_w[:, :, np.newaxis]
                    fixation_w = []
                    for i in range(3):
                        fixation_w.append(fix_w)
                    fixation_w = np.concatenate(fixation_w, axis=2)

                    if bool_shift == True:
                        fixation_w = np.roll(fixation_w, pixel_shift_lon, axis=0)  # shifting down
                        fixation_w[:pixel_shift_lon, :, :] = 0
                        cmp = utils.e2c(fixation_w, int(600 / 4))
                        cmp[0] = np.roll(cmp[0], -1 * pixel_shift_lat, axis=1)  # shifting left
                        cmp[0][:, (-1 * pixel_shift_lat):, :] = 0
                        cmp[1] = np.roll(cmp[1], pixel_shift_lat, axis=1)  # shifting right
                        cmp[1][:, :pixel_shift_lat, :] = 0
                        cmp[3] = np.roll(cmp[3], int((-1 * pixel_shift_lat) / 2), axis=1)  # shifting left
                        cmp[3][:, int((-1 * pixel_shift_lat) / 2):, :] = 0
                        cmp[4] = np.roll(cmp[4], int((-1 * pixel_shift_lat) / 2), axis=1)  # shifting left
                        cmp[4][:, int((-1 * pixel_shift_lat) / 2):, :] = 0
                        fixation_w = utils.c2e(cmp, 300, 600)

                    if bool_rescale == True:
                        fix_w = cv2.resize(fix_w, (seq_width, int(seq_width / 2)))
                        fix_w = np.pad(fix_w, ((pixel_pad, pixel_pad), (0, 0)), 'constant')
                    else:
                        fixation_w = cv2.resize(fixation_w, (seq_width, seq_height))

                    fixation_w = cv2.normalize(fixation_w, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                dtype=cv2.CV_8UC1)
                    fixation_w = cv2.applyColorMap(fixation_w, cv2.COLORMAP_HOT)

                    heatmap = cv2.addWeighted(fixation_wo, 1, fixation_w, 1, 0)
                    overlay = cv2.addWeighted(frame, 0.5, heatmap, 1, 0)

                    # write the current key frame
                    oly_vid.write(overlay)

                count_frm += 1
                print("{} frames processed.".format(count_frm))

            count_seq += 1
            print("{} videos processed.".format(count_seq))

            oly_vid.release()

    def auto_oly_2(self):
        file_ids = os.listdir(os.getcwd() + '/fixations_w_sound/')
        #file_ids = os.listdir(os.getcwd() + '/temp/')

        fix_wo_special_list = ['_-1An41lDIJ6Q', '_-SdGCX2H-_Uk', '_-3DMhSnlf3Oo', '_-gSueCRQO_5g', '_-MFVmxoXgeNQ',
                               '_-MzcdEI-tSUc_1', '_-nDu57CGqbLM', '_-dd39herpgXA', '_-eqmjLZGZ36k', '_-G8pABGosD38',
                               '_-G8pABGosD38_B', '_-gqLmxlTXU64', '_-ByBF08H-wDA_B', '_-I6EHQIQ_StU_B', '_-JbGlLbNYBy8_B',
                               '_-ZOqHvzhhb_8_B']
        fix_w_special_list = ['_-ZuXCMpVR24I', '_-gSueCRQO_5g', '_-MFVmxoXgeNQ', '_-MzcdEI-tSUc_1', '_-nDu57CGqbLM',
                              '_-dd39herpgXA', '_-eqmjLZGZ36k', '_-G8pABGosD38', '_-G8pABGosD38_B', '_-gqLmxlTXU64',
                              '_-ByBF08H-wDA_B', '_-I6EHQIQ_StU_B', '_-JbGlLbNYBy8_B', '_-ZOqHvzhhb_8_B']
        fix_reserve_left_list = ['_-0suxwissusc', '_-7P37xEKbLrQ', '_-69Aw5PC1h4Y', '_-72f3ayGhMEA_2',
                                 '_-72f3ayGhMEA_6', '_-Bvu9m__ZX60', '_-HNQMF7e6IL0', '_-IRG9Z7Y2uS4', '_-Ngj6C_RMK1g_2',
                                 '_-72f3ayGhMEA_6_B', '_-MzcdEI-tSUc_1', '_-0cfJOmUaNNI_1']
        fix_reserve_right_list = ['_-Ngj6C_RMK1g_1', '_-72f3ayGhMEA_4', '_-gqLmxlTXU64', '_-ZOqHvzhhb_8_B']

        count_seq = 0
        count_special = 0
        count_reserve = 0
        for id in file_ids:
            #debug
            #id = '_-Uy5LTocHmoA_2'
            #if count_seq == 1:
              #  break
            bool_special_shift_wo = False
            bool_special_shift_w = False
            bool_reserve_left = False
            bool_reserve_right = False
            if id in fix_wo_special_list:
                bool_special_shift_wo = True
                special_shift_wo = int(-1 * 13)
                count_special += 1
            if id in fix_w_special_list:
                bool_special_shift_w = True
                special_shift_w = int(-1 * 13)
                count_special += 1
            if id in fix_reserve_left_list:
                bool_reserve_left = True
                count_reserve += 1
            if id in fix_reserve_right_list:
                bool_reserve_right = True
                count_reserve += 1

            fix_wo_path = os.path.join(os.getcwd() + '/fixations_wo_sound/', id)
            fix_wo_files = os.listdir(fix_wo_path)
            fix_wo_files.sort(key=lambda x: x[:-4])
            fix_w_path = os.path.join(os.getcwd() + '/fixations_w_sound/', id)
            fix_w_files = os.listdir(fix_w_path)
            fix_w_files.sort(key=lambda x: x[:-4])

            seq_item = cv2.VideoCapture(os.getcwd() + '/source_videos/' + id + '.mp4')
            seq_width = int(seq_item.get(3))
            seq_height = int(seq_item.get(4))
            seq_fps = int(seq_item.get(5))
            seq_numFrm = int(seq_item.get(7))

            oly_vid_path = os.getcwd() + '/' + id + '.avi'
            #oly_vid = cv2.VideoWriter(oly_vid_path, 0, seq_fps, (560, 300))
            oly_vid = cv2.VideoWriter(oly_vid_path, 0, seq_fps, (seq_width, seq_height))

            count_frm = 0
            for idx in range(seq_numFrm):
                if (idx + 1) % frm_interval == 0:
                    seq_item.set(1, idx)
                    ret, frame = seq_item.read()
                    if ret == False:
                        continue

                    # find the corresponding w/wo sound fixation maps + shifting
                    png_wo_path = fix_wo_path + '/' + fix_wo_files[idx]
                    fix_wo = cv2.imread(png_wo_path, cv2.IMREAD_GRAYSCALE)

                    if bool_special_shift_wo == True:
                        fix_wo = np.roll(fix_wo, special_shift_wo, axis=0)

                    png_w_path = fix_w_path + '/' + fix_w_files[idx]
                    fix_w = cv2.imread(png_w_path, cv2.IMREAD_GRAYSCALE)

                    if bool_special_shift_w == True:
                        fix_w = np.roll(fix_w, special_shift_w, axis=0)

                    # process the wo sound fixation maps with shift and scale
                    fix_wo_shift = fix_wo.copy()
                    fix_wo_l = fix_wo[:, :pixel_shift]
                    fix_wo_r = fix_wo[:, pixel_shift:]

                    fix_wo_shift[:, :-pixel_shift] = fix_wo_r
                    fix_wo_shift[:, -pixel_shift:] = fix_wo_l

                    if seq_height == 1920:  # up shift
                        fix_wo_shift = np.roll(fix_wo_shift, -pixel_up, axis=0)

                    if seq_height == 2160: # down shift
                        fix_wo_shift = np.roll(fix_wo_shift, int(pixel_up / 2), axis=0)

                    fix_wo_shift = cv2.resize(fix_wo_shift, (600 - pixel_shift * 2, 300))

                    fix_wo_shift_scale = np.zeros((300, 600))
                    fix_wo_shift_scale[:, pixel_shift:600 - pixel_shift] = fix_wo_shift

                    if bool_reserve_left == True:
                        fix_wo_shift_scale[:, (pixel_shift-pixel_reserve):pixel_shift] = fix_wo_shift[:, -pixel_reserve:]
                        fix_wo_shift_scale[:, -(pixel_reserve + pixel_shift):-pixel_shift] = 0

                    if bool_reserve_right == True:
                        fix_wo_shift_scale[:, -pixel_shift:-(pixel_shift-pixel_reserve)] = fix_wo_shift[:, :pixel_reserve]
                        fix_wo_shift_scale[:, pixel_shift:(pixel_reserve + pixel_shift)] = 0

                    fix_wo_shift_scale[-pole_cut:, :] = 0 # cut the border
                    fix_wo_shift_scale[:pole_cut, :] = 0

                    # process the w sound fixation maps with shift and scale
                    fix_w_shift = fix_w.copy()
                    fix_w_l = fix_w[:, :pixel_shift]
                    fix_w_r = fix_w[:, pixel_shift:]

                    fix_w_shift[:, :-pixel_shift] = fix_w_r
                    fix_w_shift[:, -pixel_shift:] = fix_w_l

                    if seq_height == 1920:  # up shift
                        fix_w_shift = np.roll(fix_w_shift, -pixel_up, axis=0)

                    if seq_height == 2160:  # down shift
                        fix_w_shift = np.roll(fix_w_shift, int(pixel_up / 2), axis=0)

                    fix_w_shift = cv2.resize(fix_w_shift, (600 - pixel_shift * 2, 300))

                    fix_w_shift_scale = np.zeros((300, 600))
                    fix_w_shift_scale[:, pixel_shift:600 - pixel_shift] = fix_w_shift

                    if bool_reserve_left == True:
                        fix_w_shift_scale[:, (pixel_shift - pixel_reserve):pixel_shift] = fix_w_shift[:, -pixel_reserve:]
                        fix_w_shift_scale[:, -(pixel_reserve + pixel_shift):-pixel_shift] = 0

                    if bool_reserve_right == True:
                        fix_w_shift_scale[:, -pixel_shift:-(pixel_shift-pixel_reserve)] = fix_w_shift[:, :pixel_reserve]
                        fix_w_shift_scale[:, pixel_shift:(pixel_reserve + pixel_shift)] = 0

                    fix_w_shift_scale[-pole_cut:, :] = 0
                    fix_w_shift_scale[:pole_cut, :] = 0

                    # generate the wo sound heatmap of the current frame
                    fix_wo_shift_scale = fix_wo_shift_scale[:, :, np.newaxis]
                    fixation_wo = []
                    for i in range(3):
                        fixation_wo.append(fix_wo_shift_scale)
                    fixation_wo = np.concatenate(fixation_wo, axis=2)

                    fixation_wo = cv2.resize(fixation_wo, (seq_width, seq_height))
                    fixation_wo = cv2.normalize(fixation_wo, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_8UC1)
                    fixation_wo = cv2.applyColorMap(fixation_wo, cv2.COLORMAP_OCEAN)

                    # generate the w sound heatmap of the current frame
                    fix_w_shift_scale = fix_w_shift_scale[:, :, np.newaxis]
                    fixation_w = []
                    for i in range(3):
                        fixation_w.append(fix_w_shift_scale)
                    fixation_w = np.concatenate(fixation_w, axis=2)

                    fixation_w = cv2.resize(fixation_w, (seq_width, seq_height))
                    fixation_w = cv2.normalize(fixation_w, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                dtype=cv2.CV_8UC1)
                    fixation_w = cv2.applyColorMap(fixation_w, cv2.COLORMAP_HOT)

                    # overlay
                    heatmap = cv2.addWeighted(fixation_wo, 1, fixation_w, 1, 0)
                    overlay = cv2.addWeighted(frame, 0.5, heatmap, 1, 0)

                    # write the current key frame
                   # oly_write = cv2.resize(overlay, (1080, 540))
                    oly_vid.write(overlay)

                count_frm += 1
                print("{} frames processed.".format(count_frm))

            count_seq += 1
            print("{} videos processed.".format(count_seq))

            oly_vid.release()

        print("totally {} special videos.".format(count_special))
        print("totally {} videos with special edge processing.".format(count_reserve))

    def SOD_crop(self):
        xml_names = os.listdir(os.getcwd() + '/xml_files/')

        for name_xml in xml_names:
            utils.bbox2sod(name_xml)


if __name__ == '__main__':
    pvsod = PanoVSOD_stts()

    if bool_numFrm == True:
        nFrames = pvsod.num_frames_count()
        print("There are totally {} frames.".format(nFrames))

    if genOverlay == 1:
        if seq2frm == 1:
            pvsod.VideoToImg()

        if frm2oly == 1:
            pvsod.fixation_overlay()

        if frm2oly_2 == 1:
            pvsod.fixation_overlay_2()

        if oly2vid == 1:
            pvsod.ImgToVideo()

    if vr_scene == 1:
        #pvsod.vrs_txt_rename()
        pvsod.vrs_coor()

    if auto_oly == 1:
        pvsod.auto_oly_vid()

    if bool_frm2vid == True:
        pvsod.ImgToVideo()

    if bool_shift_scale == True:
        pvsod.auto_oly_2()

    if bool_SOD_crop == True:
        pvsod.SOD_crop()