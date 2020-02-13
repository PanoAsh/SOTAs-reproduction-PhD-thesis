import os
import cv2
import numpy as np

# parameters
Width = 3840
Height= 2160
Fps = 25

class PanoVSOD_stts():
    def __init__(self):
        self.path_seq = os.getcwd() + '/stimulis/'
        self.path_sor = os.getcwd() + '/source_videos/'
        self.path_frm = os.getcwd() + '/frames/' # one by one
        self.path_syn = os.getcwd() + '/synthetic_video.avi'
        self.path_oly = os.getcwd() + '/overlays/'
        self.path_fix = os.getcwd() + '/fixations/' # one by one

    def num_frames_count(self):
        seq_list = os.listdir(self.path_sor)

        frames_num = []
        count = 0
        for seq in seq_list:
            if seq.endswith('.mp4'):
                seq_path = os.path.join(os.path.abspath(self.path_sor), seq)
                cap = cv2.VideoCapture(seq_path)
                frames_num.append(int(cap.get(7)))
                count += 1
                print(" {} videos processed".format(count))

        total_frames = np.sum(frames_num)

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
        frm = os.listdir(self.path_oly)
        frm.sort(key=lambda x: x[:-4])
        video = cv2.VideoWriter(self.path_syn, 0, Fps, (Width, Height)) # modify the resolution, fps accordingly

        count = 1
        for item in frm:
            frame_path = os.path.join(os.path.abspath(self.path_oly), item)
            video.write(cv2.imread(frame_path))
            print("{} writen".format(count))
            count += 1

        cv2.destroyAllWindows()
        video.release()

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


if __name__ == '__main__':
    pvsod = PanoVSOD_stts()

   # nFrames = pvsod.num_frames_count()
    #print("There are totally {} frames.".format(nFrames))

    # generate the frames
    #pvsod.VideoToImg()

    pvsod.fixation_overlay()
    pvsod.ImgToVideo()