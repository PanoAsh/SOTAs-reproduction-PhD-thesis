import os
import torch
#from spherical_distortion.util import load_torch_img, torch2numpy
#from spherical_distortion.functional import create_tangent_images, tangent_images_to_equirectangular
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import numpy as np
import math


def listTrain():
    f1 = open(os.getcwd() + '/train_img.lst', 'w')
    f2 = open(os.getcwd() + '/train_msk.lst', 'w')
    f3 = open(os.getcwd() + '/test_img.lst', 'w')
    f4 = open(os.getcwd() + '/test_msk.lst', 'w')
    f5 = open(os.getcwd() + '/test_ins_lst', 'w')
    frm_count = 0
    for cls in os.listdir(os.getcwd() + '/data/train_img/'):
        for vid in os.listdir(os.path.join(os.getcwd() + '/data/train_img/', cls)):
            vidList = os.listdir(os.path.join(os.path.join(os.getcwd() + '/data/train_img/', cls), vid))
            vidList.sort(key=lambda x: x[:-4])
            for frm in vidList:
                lineImg = os.path.join(os.path.join(os.path.join(os.getcwd() + '/data/train_img/', cls), vid), frm) + \
                          '\n'
                lineMsk = os.path.join(os.path.join(os.path.join(os.getcwd() + '/data/train_msk/', cls), vid),
                                       'frame_' + frm[-10:]) + '\n'
                f1.write(lineImg)
                f2.write(lineMsk)
                frm_count += 1
    print(" {} frames wrote. ".format(frm_count))
    for cls in os.listdir(os.getcwd() + '/data/test_img/'):
        for vid in os.listdir(os.path.join(os.getcwd() + '/data/test_img/', cls)):
            vidList = os.listdir(os.path.join(os.path.join(os.getcwd() + '/data/test_img/', cls), vid))
            vidList.sort(key=lambda x: x[:-4])
            for frm in vidList:
                lineImg = os.path.join(os.path.join(os.path.join(os.getcwd() + '/data/test_img/', cls), vid), frm) + \
                          '\n'
                lineMsk = os.path.join(os.path.join(os.path.join(os.getcwd() + '/data/test_msk/', cls), vid),
                                       'frame_' + frm[-10:]) + '\n'
                lineIns = os.path.join(os.path.join(os.path.join(os.getcwd() + '/data/test_ins/', cls), vid),
                                       'frame_' + frm[-10:]) + '\n'
                f3.write(lineImg)
                f4.write(lineMsk)
                f5.write(lineIns)
                frm_count += 1
    print(" {} frames wrote. ".format(frm_count))
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()

#def ER2TI(ER, base_order, sample_order):
    #ER = ER.cuda() # not suggested for the dataload get_item process
 #   TIs = create_tangent_images(ER, base_order, sample_order)
 #   TIs = TIs.permute(1, 0, 2, 3)

 #   return TIs

#def TI2ER(TIs, base_level, sample_level):
 #   ER = tangent_images_to_equirectangular(TIs, [int(2048 / 2 ** (10 - sample_level)),
    #                                                 int(4096 / 2 ** (10 - sample_level))],
   #                                        base_level, sample_level)

   # return ER

def demo():
    img_pth = os.getcwd() + '/results_analysis/Img/'
    gt_pth = os.getcwd() + '/results_analysis/GT/'
    ins_pth = os.getcwd() + '/results_analysis/InsGT/'
    salEr_pth = os.getcwd() + '/results_analysis/Sal_ER_FcnResNet101/'
    salTi_pth = os.getcwd() + '/results_analysis/Sal_TI_S7B0_FcnResNet101/'
    salBasnet = os.getcwd() + '/results_analysis/Sal_basnet/'
    salCpdr = os.getcwd() + '/results_analysis/Sal_cpd-r/'
    salEgnet = os.getcwd() + '/results_analysis/Sal_egnet/'
    salF3net = os.getcwd() + '/results_analysis/Sal_f3net/'
    salPoolnet = os.getcwd() + '/results_analysis/Sal_poolnet/'
    salScribbleSOD = os.getcwd() + '/results_analysis/Sal_scribbleSOD/'
    salScrn = os.getcwd() + '/results_analysis/Sal_scrn/'
    salRcrnet = os.getcwd() + '/results_analysis/Sal_rcrnet/'


    demo = cv2.VideoWriter(os.getcwd() + '/' + 'demo.avi', 0, 100, (1536, 1280))
    img_list = os.listdir(img_pth)
    img_list.sort(key=lambda x: x[:-4])

    count = 1
    for item in img_list:
        img = cv2.imread(img_pth + item)
        gt = cv2.imread(gt_pth + item)
        ins = cv2.imread(ins_pth + item)
        salEr = cv2.imread(salEr_pth + item)
        salTi = cv2.imread(salTi_pth + item)
        salBas = cv2.imread(salBasnet + item)
        salCpd = cv2.imread(salCpdr + item)
        salEgn = cv2.imread(salEgnet + item)
        salF3n = cv2.imread(salF3net + item)
        salPoo = cv2.imread(salPoolnet + item[:-4] + '_sal_fuse.png')
        salSSOD = cv2.imread(salScribbleSOD + item)
        salScr = cv2.imread(salScrn + item)
        salRcr = cv2.imread(salRcrnet + item)


        frm = np.zeros((1280, 1536, 3))
        frm[:256, :512, :] = img
        frm[:256, 512:1024, :] = gt
        frm[:256, 1024:, :] = ins

        frm[256:512, :512, :] = salEr
        frm[256:512, 512:1024, :] = salTi
        frm[256:512, 1024:, :] = salBas

        frm[512:768, :512, :] = salCpd
        frm[512:768, 512:1024, :] = salEgn
        frm[512:768, 1024:, :] = salF3n

        frm[768:1024, :512, :] = salPoo
        frm[768:1024, 512:1024, :] = salSSOD
        frm[768:1024, 1024:, :] = salScr

        frm[1024:, :512, :] = salRcr

        demo.write(np.uint8(frm))
        print("{} writen".format(count))
        count += 1

    demo.release()

def demo_facile():
    img_pth = os.getcwd() + '/result_analysis/GTs/Img/'
    sal_pth = os.getcwd() + '/result_analysis/Sal_test_raft_kitti/'

    demo = cv2.VideoWriter(os.getcwd() + '/' + 'demo.avi', 0, 100, (1024, 256))
    img_list = os.listdir(img_pth)
    img_list.sort(key=lambda x: x[:-4])

    count = 1
    for item in img_list:
        img = cv2.imread(img_pth + item)
        sal = cv2.imread(sal_pth + item)

        frm = np.zeros((256, 1024, 3))
        frm[:256, :512, :] = img
        frm[:256, 512:1024, :] = sal

        demo.write(np.uint8(frm))
        print("{} writen".format(count))
        count += 1

    demo.release()

def listTest():
    img_pth = os.getcwd() + '/results_analysis/Img/'
    img_list = os.listdir(img_pth)
    img_list.sort(key=lambda x: x[:-4])
    f = open(os.getcwd() + '/test.lst', 'w')

    for item in img_list:
        f.write(item + '\n')
    f.close()

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)

    return dn

class Cube2Equirec(object):
    def __init__(self, batch_size, cube_size, output_h, output_w, cube_fov=90, CUDA=True):
        self.batch_size = batch_size # NOTE: not in use at all
        self.cube_h = cube_size
        self.cube_w = cube_size
        self.output_h = output_h
        self.output_w = output_w
        self.fov = cube_fov
        self.fov_rad = self.fov * np.pi / 180
        self.CUDA = CUDA

        # Compute the parameters for projection
        assert self.cube_w == self.cube_h
        self.radius = int(0.5 * cube_size)

        # Map equirectangular pixel to longitude and latitude
        # NOTE: Make end a full length since arange have a right open bound [a, b)
        theta_start = math.pi - (math.pi / output_w)
        theta_end = -math.pi
        theta_step = 2 * math.pi / output_w
        theta_range = torch.arange(theta_start, theta_end, -theta_step)

        phi_start = 0.5 * math.pi - (0.5 * math.pi / output_h)
        phi_end = -0.5 * math.pi
        phi_step = math.pi / output_h
        phi_range = torch.arange(phi_start, phi_end, -phi_step)

        # Stack to get the longitude latitude map
        self.theta_map = theta_range.unsqueeze(0).repeat(output_h, 1)
        self.phi_map = phi_range.unsqueeze(-1).repeat(1, output_w)
        self.lonlat_map = torch.stack([self.theta_map, self.phi_map], dim=-1)

        # Get mapping relation (h, w, face)
        # [back, down, front, left, right, up] => [0, 1, 2, 3, 4, 5]
        # self.orientation_mask = self.get_orientation_mask()

        # Project each face to 3D cube and convert to pixel coordinates
        self.grid, self.orientation_mask = self.get_grid2()

        if self.CUDA:
            self.grid#.cuda()
            self.orientation_mask#.cuda()

    # Compute the orientation mask for the lonlat map
    def get_orientation_mask(self):
        mask_back_lon = (self.lonlat_map[:, :, 0] > np.pi - 0.5 * self.fov_rad) + \
                        (self.lonlat_map[:, :, 0] < - np.pi + 0.5 * self.fov_rad)
        mask_back_lat = (self.lonlat_map[:, :, 1] < 0.5 * self.fov_rad) * \
                        (self.lonlat_map[:, :, 1] > - 0.5 * self.fov_rad)
        mask_back = mask_back_lon * mask_back_lat

        mask_down_lat = (self.lonlat_map[:, :, 1] <= - 0.5 * self.fov_rad)
        mask_down = mask_down_lat

        mask_front_lon = (self.lonlat_map[:, :, 0] < 0.5 * self.fov_rad) * \
                         (self.lonlat_map[:, :, 0] > - 0.5 * self.fov_rad)
        mask_front_lat = (self.lonlat_map[:, :, 1] < 0.5 * self.fov_rad) * \
                         (self.lonlat_map[:, :, 1] > - 0.5 * self.fov_rad)
        mask_front = mask_front_lon * mask_front_lat

        mask_left_lon = (self.lonlat_map[:, :, 0] < np.pi - 0.5 * self.fov_rad) * \
                        (self.lonlat_map[:, :, 0] > 0.5 * self.fov_rad)
        mask_left_lat = (self.lonlat_map[:, :, 1] < 0.5 * self.fov_rad) * \
                        (self.lonlat_map[:, :, 1] > - 0.5 * self.fov_rad)
        mask_left = mask_left_lon * mask_left_lat

        mask_right_lon = (self.lonlat_map[:, :, 0] < - 0.5 * self.fov_rad) * \
                         (self.lonlat_map[:, :, 0] > - np.pi + 0.5 * self.fov_rad)
        mask_right_lat = (self.lonlat_map[:, :, 1] < 0.5 * self.fov_rad) * \
                         (self.lonlat_map[:, :, 1] > - 0.5 * self.fov_rad)
        mask_right = mask_right_lon * mask_right_lat

        # mask_up_lat = (self.lonlat_map[:, :, 1] >= 0.5 * self.fov_rad)
        mask_up = torch.ones([self.output_h, self.output_w])
        mask_up = mask_up - (self.lonlat_map[:, :, 1] < 0).float() - \
                  (mask_front.float() + mask_right.float() + mask_left.float() + mask_back.float())
        mask_up = (mask_up == 1)

        # Face map contains numbers correspond to that face
        orientation_mask = mask_back * 0 + mask_down * 1 + mask_front * 2 + mask_left * 3 + mask_right * 4 + mask_up * 5

        return orientation_mask

    def get_grid2(self):
        # Get the point of equirectangular on 3D ball
        x_3d = (self.radius * torch.cos(self.phi_map) * torch.sin(self.theta_map)).view(self.output_h, self.output_w, 1)
        y_3d = (self.radius * torch.sin(self.phi_map)).view(self.output_h, self.output_w, 1)
        z_3d = (self.radius * torch.cos(self.phi_map) * torch.cos(self.theta_map)).view(self.output_h, self.output_w, 1)

        self.grid_ball = torch.cat([x_3d, y_3d, z_3d], 2).view(self.output_h, self.output_w, 3)

        # Compute the down grid
        radius_ratio_down = torch.abs(y_3d / self.radius)
        grid_down_raw = self.grid_ball / radius_ratio_down.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_down_w = (-grid_down_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_down_h = (-grid_down_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_down = torch.cat([grid_down_w, grid_down_h], 2).unsqueeze(0)
        mask_down = (((grid_down_w <= 1) * (grid_down_w >= -1)) * ((grid_down_h <= 1) * (grid_down_h >= -1)) *
                    (grid_down_raw[:, :, 1] == -self.radius).unsqueeze(2)).float()

        # Compute the up grid
        radius_ratio_up = torch.abs(y_3d / self.radius)
        grid_up_raw = self.grid_ball / radius_ratio_up.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_up_w = (-grid_up_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_up_h = (grid_up_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_up = torch.cat([grid_up_w, grid_up_h], 2).unsqueeze(0)
        mask_up = (((grid_up_w <= 1) * (grid_up_w >= -1)) * ((grid_up_h <= 1) * (grid_up_h >= -1)) *
                  (grid_up_raw[:, :, 1] == self.radius).unsqueeze(2)).float()

        # Compute the front grid
        radius_ratio_front = torch.abs(z_3d / self.radius)
        grid_front_raw = self.grid_ball / radius_ratio_front.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_front_w = (-grid_front_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_front_h = (-grid_front_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_front = torch.cat([grid_front_w, grid_front_h], 2).unsqueeze(0)
        mask_front = (((grid_front_w <= 1) * (grid_front_w >= -1)) * ((grid_front_h <= 1) * (grid_front_h >= -1)) *
                  (torch.round(grid_front_raw[:, :, 2]) == self.radius).unsqueeze(2)).float()

        # Compute the back grid
        radius_ratio_back = torch.abs(z_3d / self.radius)
        grid_back_raw = self.grid_ball / radius_ratio_back.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_back_w = (grid_back_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_back_h = (-grid_back_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_back = torch.cat([grid_back_w, grid_back_h], 2).unsqueeze(0)
        mask_back = (((grid_back_w <= 1) * (grid_back_w >= -1)) * ((grid_back_h <= 1) * (grid_back_h >= -1)) *
                  (torch.round(grid_back_raw[:, :, 2]) == -self.radius).unsqueeze(2)).float()


        # Compute the right grid
        radius_ratio_right = torch.abs(x_3d / self.radius)
        grid_right_raw = self.grid_ball / radius_ratio_right.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_right_w = (-grid_right_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_right_h = (-grid_right_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_right = torch.cat([grid_right_w, grid_right_h], 2).unsqueeze(0)
        mask_right = (((grid_right_w <= 1) * (grid_right_w >= -1)) * ((grid_right_h <= 1) * (grid_right_h >= -1)) *
                  (torch.round(grid_right_raw[:, :, 0]) == -self.radius).unsqueeze(2)).float()

        # Compute the left grid
        radius_ratio_left = torch.abs(x_3d / self.radius)
        grid_left_raw = self.grid_ball / radius_ratio_left.view(self.output_h, self.output_w, 1).expand(-1, -1, 3)
        grid_left_w = (grid_left_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_left_h = (-grid_left_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_left = torch.cat([grid_left_w, grid_left_h], 2).unsqueeze(0)
        mask_left = (((grid_left_w <= 1) * (grid_left_w >= -1)) * ((grid_left_h <= 1) * (grid_left_h >= -1)) *
                  (torch.round(grid_left_raw[:, :, 0]) == self.radius).unsqueeze(2)).float()

        # Face map contains numbers correspond to that face
        orientation_mask = mask_back * 0 + mask_down * 1 + mask_front * 2 + mask_left * 3 + mask_right * 4 + mask_up * 5

        return torch.cat([grid_back, grid_down, grid_front, grid_left, grid_right, grid_up], 0), orientation_mask

    # Convert cubic images to equirectangular
    def _ToEquirec(self, batch, mode):
        batch_size, ch, H, W = batch.shape
        if batch_size != 6:
            raise ValueError("Batch size mismatch!!")

        if self.CUDA:
            output = Variable(torch.zeros(1, ch, self.output_h, self.output_w), requires_grad=False).cuda()
        else:
            output = Variable(torch.zeros(1, ch, self.output_h, self.output_w), requires_grad=False)

        for ori in range(6):
            grid = self.grid[ori, :, :, :].unsqueeze(0) # 1, self.output_h, self.output_w, 2
            mask = (self.orientation_mask == ori).unsqueeze(0) # 1, self.output_h, self.output_w, 1

            if self.CUDA:
                masked_grid = Variable(grid * mask.float().expand(-1, -1, -1, 2)).cuda() # 1, self.output_h, self.output_w, 2
            else:
                masked_grid = Variable(grid * mask.float().expand(-1, -1, -1, 2))

            source_image = batch[ori].unsqueeze(0) # 1, ch, H, W

            sampled_image = torch.nn.functional.grid_sample(
                                source_image,
                                masked_grid,
                                mode=mode
                                ) # 1, ch, self.output_h, self.output_w

            if self.CUDA:
                sampled_image_masked = sampled_image * \
                                    Variable(mask.float().view(1, 1, self.output_h, self.output_w).expand(1, ch, -1, -1)).cuda()
            else:
                sampled_image_masked = sampled_image * \
                                       Variable(mask.float().view(1, 1, self.output_h, self.output_w).expand(1, ch, -1, -1))
            output = output + sampled_image_masked # 1, ch, self.output_h, self.output_w

        return output

    # Convert input cubic tensor to output equirectangular image
    def ToEquirecTensor(self, batch, mode='bilinear'):
        # Check whether batch size is 6x
        assert mode in ['nearest', 'bilinear']
        batch_size = batch.size()[0]
        if batch_size % 6 != 0:
            raise ValueError("Batch size should be 6x")

        processed = []
        for idx in range(int(batch_size / 6)):
            target = batch[idx * 6 : (idx + 1) * 6, :, :, :]
            target_processed = self._ToEquirec(target, mode)
            processed.append(target_processed)

        output = torch.cat(processed, 0)
        return output

if __name__ == '__main__':
    print()
    #listTrain()
    #ER2TI()
    demo_facile()
    #listTest()
