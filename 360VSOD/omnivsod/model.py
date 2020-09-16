import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from retrain.RCRNet.libs.networks import VideoModel
from collections import OrderedDict
from util import Cube2Equirec
import cv2
import numpy as np
import torch.nn.functional as F


def convert_state_dict_omni(state_dict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name_1 = 'mainGUN.' + k
        name_2 = 'branchGUN.' + k
        state_dict_new[name_1] = v
        state_dict_new[name_2] = v

    return state_dict_new

class CETransform(nn.Module):
    def __init__(self, feat_h):
        super(CETransform, self).__init__()
        cube_h = [feat_h]
        self.c2e = dict()
        for h in cube_h:
            a = Cube2Equirec(1, h, h, h*2)
            self.c2e['(%d)' % (h)] = a

    def C2E(self, x):
        [bs, c, h, w] = x.shape
        key = '(%d)' % (h)
        assert key in self.c2e and h == w

        return self.c2e[key].ToEquirecTensor(x)

    def forward(self, cube):

        return self.C2E(cube)

class ECInteract(nn.Module):
    def __init__(self, feat_num, feat_h):
        super(ECInteract, self).__init__()
        self.feat_num = feat_num
        self.conv_fusion = nn.Sequential(nn.Conv2d(self.feat_num, self.feat_num, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))
        self.conv_mask = nn.Sequential(nn.Conv2d(self.feat_num*2, 1, kernel_size=1, padding=0), nn.Sigmoid())
        self.feat_h = feat_h
        self.ECInteract = CETransform(self.feat_h)  # B, D, F, L, R, U

    def forward(self, m, f, r, b, l, u, d):
        AuxFeat = []
        for idx in range(self.feat_num):
            front = f[:, idx, :, :]
            right = r[:, idx, :, :]
            back = b[:, idx, :, :]
            left = l[:, idx, :, :]
            up = u[:, idx, :, :]
            down = d[:, idx, :, :]
            CMMap = []
            CMMap.append(back)
            CMMap.append(down)
            CMMap.append(front)
            CMMap.append(left)
            CMMap.append(right)
            CMMap.append(up)
            CMMap = torch.stack(CMMap)
            ERMap = self.ECInteract(CMMap)
           # debug = np.squeeze(ERMap.cpu().data.numpy())
           # cv2.imwrite('debug.png', debug*255)
            AuxFeat.append(ERMap)
        AuxFeat = torch.stack(AuxFeat, dim=2)[0]

        # compute mask for stable fusion
        M_AuxFeat = self.conv_fusion(AuxFeat)
        M_MainFeat = self.conv_fusion(m)
        M_joint = torch.cat((M_MainFeat, M_AuxFeat), 1)
        Mask = self.conv_mask(M_joint)

        # fusion (nailing done)
        fuseFeat = m + Mask * M_AuxFeat

        return fuseFeat

class RefineSalMap(nn.Module):
    def __init__(self):
        super(RefineSalMap, self).__init__()
        self.refine_1 = nn.Sequential(
                        nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
                        )
        self.refine_2 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        )
        self.deconv_1 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1,
                                           bias=True, dilation=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(inplace=True),
                        )
        self.deconv_2 = nn.Sequential(
                        nn.ConvTranspose2d(192, 32, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1,
                                           bias=True, dilation=1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(inplace=True),
                        )
        self.refine_3 = nn.Sequential(
                        nn.Conv2d(96, 16, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(16),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
                        )
        self.bilinear_1 = nn.UpsamplingBilinear2d(size=(128, 256))
        self.bilinear_2 = nn.UpsamplingBilinear2d(size=(256, 512))

    def forward(self, inputs):
        x = inputs
        out_1 = self.refine_1(x)
        out_2 = self.refine_2(out_1)
        deconv_out1 = self.deconv_1(out_2)
        up_1 = self.bilinear_1(out_2)
        deconv_out2 = self.deconv_2(torch.cat((deconv_out1, up_1), dim=1))
        up_2 = self.bilinear_2(out_1)
        out_3 = self.refine_3(torch.cat((deconv_out2, up_2), dim = 1))

        return out_3

# OmniVNet
class OmniVNet(nn.Module):
    def __init__(self):
        super(OmniVNet, self).__init__()
        self.mainGUN = VideoModel()
        self.branchGUN = VideoModel()
        self.genPreds = ECInteract(1, 256)
        self.refineGUN = RefineSalMap()

    def forward(self, ER, CM_f, CM_r, CM_b, CM_l, CM_u, CM_d, Sound_map):
        clip = ER.unsqueeze(0)

        # Encoder: branches
        feats_f = self.branchGUN.backbone.feat_conv(CM_f)
        feats_r = self.branchGUN.backbone.feat_conv(CM_r)
        feats_b = self.branchGUN.backbone.feat_conv(CM_b)
        feats_l = self.branchGUN.backbone.feat_conv(CM_l)
        feats_u = self.branchGUN.backbone.feat_conv(CM_u)
        feats_d = self.branchGUN.backbone.feat_conv(CM_d)

        # Encoder: main stream: equirectangular
        L0 = self.mainGUN.backbone.resnet.conv1(ER)
        L0 = self.mainGUN.backbone.resnet.bn1(L0)
        L0 = self.mainGUN.backbone.resnet.relu(L0)
        L0 = self.mainGUN.backbone.resnet.maxpool(L0)
        L1 = self.mainGUN.backbone.resnet.layer1(L0)
        L2 = self.mainGUN.backbone.resnet.layer2(L1)
        L3 = self.mainGUN.backbone.resnet.layer3(L2)
        L4 = self.mainGUN.backbone.resnet.layer4(L3)
        L4 = self.mainGUN.backbone.aspp(L4)

        # main stream to NER
        feats_time = L4.unsqueeze(2)
        feats_time = self.mainGUN.non_local_block(feats_time)
        # Deep Bidirectional ConvGRU
        frame = clip[0]
        feat = feats_time[:, :, 0, :, :]
        feats_forward = []
        # forward
        for i in range(len(clip)):
            feat = self.mainGUN.convgru_forward(feats_time[:, :, i, :, :], feat)
            feats_forward.append(feat)
        # backward
        feat = feats_forward[-1]
        feats_backward = []
        for i in range(len(clip)):
            feat = self.mainGUN.convgru_backward(feats_forward[len(clip) - 1 - i], feat)
            feats_backward.append(feat)
        feats_backward = feats_backward[::-1]
        feats = []
        for i in range(len(clip)):
            feat = torch.tanh(
                self.mainGUN.bidirection_conv(torch.cat((feats_forward[i], feats_backward[i]), dim=1)))
            feats.append(feat)
        feats = torch.stack(feats, dim=2)
        feats = self.mainGUN.non_local_block2(feats)

        # branch to NER
        feats_f4 = self.NERbranch(feats_f[4], CM_f.unsqueeze(0), self.branchGUN)
        feats_r4 = self.NERbranch(feats_r[4], CM_r.unsqueeze(0), self.branchGUN)
        feats_b4 = self.NERbranch(feats_b[4], CM_b.unsqueeze(0), self.branchGUN)
        feats_l4 = self.NERbranch(feats_l[4], CM_l.unsqueeze(0), self.branchGUN)
        feats_u4 = self.NERbranch(feats_u[4], CM_u.unsqueeze(0), self.branchGUN)
        feats_d4 = self.NERbranch(feats_d[4], CM_d.unsqueeze(0), self.branchGUN)

        # Decoder: branches
        preds_F = self.branchGUN.backbone.seg_conv(feats_f[1], feats_f[2], feats_f[3], feats_f4, [384, 384])
        preds_R = self.branchGUN.backbone.seg_conv(feats_r[1], feats_r[2], feats_r[3], feats_r4, [384, 384])
        preds_B = self.branchGUN.backbone.seg_conv(feats_b[1], feats_b[2], feats_b[3], feats_b4, [384, 384])
        preds_L = self.branchGUN.backbone.seg_conv(feats_l[1], feats_l[2], feats_l[3], feats_l4, [384, 384])
        preds_U = self.branchGUN.backbone.seg_conv(feats_u[1], feats_u[2], feats_u[3], feats_u4, [384, 384])
        preds_D = self.branchGUN.backbone.seg_conv(feats_d[1], feats_d[2], feats_d[3], feats_d4, [384, 384])

        # Decoder: mainstream
        Lbu1 = self.mainGUN.backbone.refinement1(L3, feats[:, :, 0, :, :])
        Lbu1 = F.interpolate(Lbu1, size=L2.shape[2:], mode="bilinear", align_corners=False)
        Lbu2 = self.mainGUN.backbone.refinement2(L2, Lbu1)
        Lbu2 = F.interpolate(Lbu2, size=L1.shape[2:], mode="bilinear", align_corners=False)
        Lbu3 = self.mainGUN.backbone.refinement3(L1, Lbu2)
        Lbu3 = F.interpolate(Lbu3, size=[256, 512], mode="bilinear", align_corners=False)
        preds_ER = self.mainGUN.backbone.decoder(Lbu3)

        # B, D, F, L, R, U # CM_f, CM_r, CM_b, CM_l, CM_u, CM_d
        preds_branch = []
        preds_branch.append(preds_B[0][0][:, 64:-64, 64:-64])
        preds_branch.append(preds_D[0][0][:, 64:-64, 64:-64])
        preds_branch.append(preds_F[0][0][:, 64:-64, 64:-64])
        preds_branch.append(preds_L[0][0][:, 64:-64, 64:-64])
        preds_branch.append(preds_R[0][0][:, 64:-64, 64:-64])
        preds_branch.append(preds_U[0][0][:, 64:-64, 64:-64])
        preds_branch = torch.stack(preds_branch)
        preds_branch_ER = self.genPreds.ECInteract(preds_branch)

        # refine
        predsConcatenate = torch.cat((preds_ER, preds_branch_ER), dim=1)
        predsConcatenate = predsConcatenate * Sound_map
        preds_fin = self.refineGUN(predsConcatenate)

        #debug1 = np.squeeze(preds_branch_ER.cpu().data.numpy())
        #cv2.imwrite('debug1.png', debug1 * 255)
        #debug2 = np.squeeze(preds_ER.cpu().data.numpy())
        #cv2.imwrite('debug2.png', debug2 * 255)
        #debug3 = np.squeeze(preds_fin.cpu().data.numpy())
        #cv2.imwrite('debug3.png', debug3 * 255)
        #debug4 = np.squeeze(Sound_map.cpu().data.numpy())
        #cv2.imwrite('debug4.png', debug4 * 255)

        # return preds_fin, preds_F[0], preds_R[0], preds_B[0], preds_L[0], preds_U[0], preds_D[0]
        return preds_fin

    def NERbranch(self, L4, clip, BranchBone):
        feats_time = L4.unsqueeze(2)
        feats_time = BranchBone.non_local_block(feats_time)
        # Deep Bidirectional ConvGRU
        frame = clip[0]
        feat = feats_time[:, :, 0, :, :]
        feats_forward = []
        # forward
        for i in range(len(clip)):
            feat = BranchBone.convgru_forward(feats_time[:, :, i, :, :], feat)
            feats_forward.append(feat)
        # backward
        feat = feats_forward[-1]
        feats_backward = []
        for i in range(len(clip)):
            feat = BranchBone.convgru_backward(feats_forward[len(clip) - 1 - i], feat)
            feats_backward.append(feat)
        feats_backward = feats_backward[::-1]
        feats = []
        for i in range(len(clip)):
            feat = torch.tanh(
                BranchBone.bidirection_conv(torch.cat((feats_forward[i], feats_backward[i]), dim=1)))
            feats.append(feat)
        feats = torch.stack(feats, dim=2)
        feats = BranchBone.non_local_block2(feats)

        return feats[:, :, 0, :, :]


# GLOmni network
class GTNet(nn.Module):
    def __init__(self, base, model_type, base_level):
        super(GTNet, self).__init__()
        self.base = base
        self.layer0_conv1 = self.base.backbone.conv1
        self.layer0_bn1 = self.base.backbone.bn1
        self.layer0_relu = self.base.backbone.relu
        self.layer0_maxpool = self.base.backbone.maxpool
        self.layer1 = self.base.backbone.layer1
        self.layer2 = self.base.backbone.layer2
        self.layer3 = self.base.backbone.layer3
        self.layer4 = self.base.backbone.layer4
        self.classifier = self.base.classifier

        self.layer_g2l = glo2loc()
        self.layer_l2g = loc2glo()
        self.layer_fusion = nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.model_type = model_type
     #   self.base_level = base_level
       # self.num_TIs = 20 * 4 ** self.base_level
       # self.baseIter = nn.ModuleList([self.base for idx in range(self.num_TIs)])
        self.register_parameter('FMsInit', param=None)

    def forward(self, x):
        # batch size msut equal to 1
        if self.model_type == 'G':
            y = self.base(x)['out']

        elif self.model_type == 'L':
            y = self.sumFeaMap(x)
            y[0] = self.base(x[0])['out']
          #  for idx, currIter in enumerate(self.baseIter):
           #     y[:, idx, :, :, :] = currIter(x[:, idx, :, :, :])['out']

        else:
            y = x
            print('under built...')

        return y

    def sumFeaMap(self, input):
        C1, C2, C3, C4, C5 = input.size()[0], input.size()[1], 1, input.size()[3], input.size()[4]
        output = input.new(C1, C2, C3, C4, C5)
        self.FMsInit = nn.Parameter(output)

        return self.FMsInit @ output

    # Global guidance to local branch
class glo2loc(nn.Module):
    def __init__(self):
        super(glo2loc, self).__init__()

    def forward(self, x, x_shape):

        return x

# Local refinement to global branch
class loc2glo(nn.Module):
    def __init__(self):
        super(loc2glo, self).__init__()

    def forward(self, x, x_shape):

        return x

# build the whole network
def build_model(backbone_config, coco_model, mode, model_type, base_level):
    if backbone_config == 'fcn_resnet101':
        base = torchvision.models.segmentation.fcn_resnet101(pretrained=False, num_classes=1)

        if mode == 'train':
            base_dict = base.state_dict()
            state_dict = torch.load(coco_model)
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k in base_dict and v.shape == base_dict[k].shape
            }
            base.load_state_dict(state_dict, strict=False)

    if backbone_config == 'deeplabv3_resnet101':
        base = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)

    return GTNet(base, model_type, base_level)

def build_OmniVNet():

    return OmniVNet()


if __name__ == '__main__':
    net = build_model('fcn_resnet101', os.getcwd() + '/pretrained/fcn_resnet101_coco-7ecb50ca.pth', 'train').cuda()

    img = Image.open(os.getcwd() + '/debug/img1.png')
    preprocess = transforms.Compose([
        transforms.Resize([256, 512]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = Variable(img_tensor).cuda()

    plt.imshow(img_tensor.cpu().detach().numpy()[0, 0, :, :], cmap='gray')
    plt.show()

    out = net(img_tensor)[0].cpu().detach().numpy()
    plt.imshow(out[0, :, :], cmap='gray')
    plt.show()

    #-------------------------------------------------------------------------------------------
    # supplemental for the testing of OmniVNet
    clip = ER.unsqueeze(0)

    # Encoder: main stream: equirectangular
    L0 = self.mainGUN.backbone.resnet.conv1(ER)
    L0 = self.mainGUN.backbone.resnet.bn1(L0)
    L0 = self.mainGUN.backbone.resnet.relu(L0)
    L0 = self.mainGUN.backbone.resnet.maxpool(L0)
    L1 = self.mainGUN.backbone.resnet.layer1(L0)
    L2 = self.mainGUN.backbone.resnet.layer2(L1)
    L3 = self.mainGUN.backbone.resnet.layer3(L2)
    L4 = self.mainGUN.backbone.resnet.layer4(L3)
    L4 = self.mainGUN.backbone.aspp(L4)

    # main stream to NER
    feats_time = L4.unsqueeze(2)
    feats_time = self.mainGUN.non_local_block(feats_time)
    # Deep Bidirectional ConvGRU
    frame = clip[0]
    feat = feats_time[:, :, 0, :, :]
    feats_forward = []
    # forward
    for i in range(len(clip)):
        feat = self.mainGUN.convgru_forward(feats_time[:, :, i, :, :], feat)
        feats_forward.append(feat)
    # backward
    feat = feats_forward[-1]
    feats_backward = []
    for i in range(len(clip)):
        feat = self.mainGUN.convgru_backward(feats_forward[len(clip) - 1 - i], feat)
        feats_backward.append(feat)
    feats_backward = feats_backward[::-1]
    feats = []
    for i in range(len(clip)):
        feat = torch.tanh(
            self.mainGUN.bidirection_conv(torch.cat((feats_forward[i], feats_backward[i]), dim=1)))
        feats.append(feat)
    feats = torch.stack(feats, dim=2)
    feats = self.mainGUN.non_local_block2(feats)

    # Decoder: mainstream
    preds_ER = self.mainGUN.backbone.seg_conv(L1, L2, L3, feats[:, :, 0, :, :], [256, 512])

    # -------------------------------------- v2 ------------------------------------------------
    class OmniVNet(nn.Module):
        def __init__(self):
            super(OmniVNet, self).__init__()
            self.mainGUN = VideoModel()
            # self.auxBGUN = VideoModel().backbone
            # self.auxUGUN = VideoModel().backbone
            # self.auxDGUN = VideoModel().backbone
            self.branchGUN = VideoModel

            self.nailGUN_L0 = ECInteract(64, 64)  # the number and height of the feature map
            # self.nailGUN_L1 = ECInteract(256, 64)
            #  self.nailGUN_L2 = ECInteract(512, 32)
            #   self.nailGUN_L3 = ECInteract(1024, 16)
            # self.nailGUN_L4 = ECInteract(128, 32)
            # self.nailGUN_L5 = ECInteract(128, 64)
            self.nailGUN_L6 = ECInteract(128, 256)
            self.register_parameter('emptyFace', param=None)
            self.genPreds = ECInteract(1, 256)

        def GenEmptyFace(self, feat_h):
            feat_h = int(feat_h / 2)
            self.emptyFace = nn.Parameter(torch.zeros(1, feat_h, feat_h))

            return self.emptyFace

        def forward(self, ER, CM_f, CM_r, CM_b, CM_l, CM_u, CM_d):
            clip = ER.unsqueeze(0)

            # Encoder: auxiliary branches: including behind, up, down
            feats_b = self.auxBGUN.feat_conv(CM_b)
            feats_u = self.auxUGUN.feat_conv(CM_u)
            feats_d = self.auxDGUN.feat_conv(CM_d)

            # Encoder: main stream: equirectangular
            L0 = self.mainGUN.backbone.resnet.conv1(ER)
            L0 = self.mainGUN.backbone.resnet.bn1(L0)
            L0 = self.mainGUN.backbone.resnet.relu(L0)
            L0 = self.mainGUN.backbone.resnet.maxpool(L0)
            L0_nailed = self.nailGUN_L0(L0, feats_b[0], feats_u[0], feats_d[0])

            L1 = self.mainGUN.backbone.resnet.layer1(L0_nailed)
            L1_nailed = self.nailGUN_L1(L1, feats_b[1], feats_u[1], feats_d[1])

            L2 = self.mainGUN.backbone.resnet.layer2(L1_nailed)
            #   L2_nailed = self.nailGUN_L2(L2, feats_b[2], feats_u[2], feats_d[2])

            L3 = self.mainGUN.backbone.resnet.layer3(L2)
            #    L3_nailed = self.nailGUN_L3(L3, feats_b[3], feats_u[3], feats_d[3])

            L4 = self.mainGUN.backbone.resnet.layer4(L3)
            L4 = self.mainGUN.backbone.aspp(L4)

            # main stream to NER
            feats_time = L4.unsqueeze(2)
            feats_time = self.mainGUN.non_local_block(feats_time)
            # Deep Bidirectional ConvGRU
            frame = clip[0]
            feat = feats_time[:, :, 0, :, :]
            feats_forward = []
            # forward
            for i in range(len(clip)):
                feat = self.mainGUN.convgru_forward(feats_time[:, :, i, :, :], feat)
                feats_forward.append(feat)
            # backward
            feat = feats_forward[-1]
            feats_backward = []
            for i in range(len(clip)):
                feat = self.mainGUN.convgru_backward(feats_forward[len(clip) - 1 - i], feat)
                feats_backward.append(feat)
            feats_backward = feats_backward[::-1]
            feats = []
            for i in range(len(clip)):
                feat = torch.tanh(
                    self.mainGUN.bidirection_conv(torch.cat((feats_forward[i], feats_backward[i]), dim=1)))
                feats.append(feat)
            feats = torch.stack(feats, dim=2)
            feats = self.mainGUN.non_local_block2(feats)

            # Decoder: auxiliary branches: including behind, up, down
            preds_Back = self.auxBGUN.seg_conv(feats_b[1], feats_b[2], feats_b[3], feats_b[4], [128, 128])
            preds_Up = self.auxUGUN.seg_conv(feats_u[1], feats_u[2], feats_u[3], feats_u[4], [128, 128])
            preds_Down = self.auxDGUN.seg_conv(feats_d[1], feats_d[2], feats_d[3], feats_d[4], [128, 128])

            # Decoder: mainstream
            Lbu1 = self.mainGUN.backbone.refinement1(L3, feats[:, :, 0, :, :])
            Lbu1 = F.interpolate(Lbu1, size=L2.shape[2:], mode="bilinear", align_corners=False)
            #   Lbu1_nailed = self.nailGUN_L4(Lbu1, preds_Back[3], preds_Up[3], preds_Down[3])

            Lbu2 = self.mainGUN.backbone.refinement2(L2, Lbu1)
            Lbu2 = F.interpolate(Lbu2, size=L1_nailed.shape[2:], mode="bilinear", align_corners=False)
            #  Lbu2_nailed = self.nailGUN_L5(Lbu2, preds_Back[2], preds_Up[2], preds_Down[2])

            Lbu3 = self.mainGUN.backbone.refinement3(L1_nailed, Lbu2)
            Lbu3 = F.interpolate(Lbu3, size=[256, 512], mode="bilinear", align_corners=False)
            Lbu3_nailed = self.nailGUN_L6(Lbu3, preds_Back[1], preds_Up[1], preds_Down[1])
            preds_ER = self.mainGUN.backbone.decoder(Lbu3_nailed)

            empty = self.GenEmptyFace(256).cuda()
            preds_branch = []
            preds_branch.append(preds_Back[0][0])
            preds_branch.append(preds_Down[0][0])
            preds_branch.append(empty)
            preds_branch.append(empty)
            preds_branch.append(empty)
            preds_branch.append(preds_Up[0][0])
            preds_branch = torch.stack(preds_branch)
            preds_branch_ER = self.genPreds.ECInteract(preds_branch)
            preds_fin = preds_ER + preds_branch_ER

            return preds_fin

    # ------------------------------------------------ v 3/4 ------------------------------------------------
        # get final salmap
        # B, D, F, L, R, U # CM_f, CM_r, CM_b, CM_l, CM_u, CM_d
        preds_branch = []
        preds_branch.append(preds_B[0][0])
        preds_branch.append(preds_D[0][0])
        preds_branch.append(preds_F[0][0])
        preds_branch.append(preds_L[0][0])
        preds_branch.append(preds_R[0][0])
        preds_branch.append(preds_U[0][0])
        preds_branch = torch.stack(preds_branch)
        preds_branch_ER = self.genPreds.ECInteract(preds_branch)
        predsFin = torch.cat((preds_ER, preds_branch_ER), dim=1)
        predsFin = self.ERFuse(predsFin)

        self.nailGUN_LE = ECInteract(64, 64)  # the number and height of the feature map
        self.nailGUN_LD = ECInteract(128, 256)

    # ------------------------------------------------ v 5 ----------------------------------------------------
    clip = ER.unsqueeze(0)

    # Encoder: branches
    feats_f = self.branchGUN.backbone.feat_conv(CM_f)
    feats_r = self.branchGUN.backbone.feat_conv(CM_r)
    feats_b = self.branchGUN.backbone.feat_conv(CM_b)
    feats_l = self.branchGUN.backbone.feat_conv(CM_l)
    feats_u = self.branchGUN.backbone.feat_conv(CM_u)
    feats_d = self.branchGUN.backbone.feat_conv(CM_d)

    # Encoder: main stream: equirectangular
    L0 = self.mainGUN.backbone.resnet.conv1(ER)
    L0 = self.mainGUN.backbone.resnet.bn1(L0)
    L0 = self.mainGUN.backbone.resnet.relu(L0)
    L0 = self.mainGUN.backbone.resnet.maxpool(L0)
    L1 = self.mainGUN.backbone.resnet.layer1(L0)
    L2 = self.mainGUN.backbone.resnet.layer2(L1)
    L3 = self.mainGUN.backbone.resnet.layer3(L2)
    L4 = self.mainGUN.backbone.resnet.layer4(L3)
    L4 = self.mainGUN.backbone.aspp(L4)

    # main stream to NER
    feats_time = L4.unsqueeze(2)
    feats_time = self.mainGUN.non_local_block(feats_time)
    # Deep Bidirectional ConvGRU
    frame = clip[0]
    feat = feats_time[:, :, 0, :, :]
    feats_forward = []
    # forward
    for i in range(len(clip)):
        feat = self.mainGUN.convgru_forward(feats_time[:, :, i, :, :], feat)
        feats_forward.append(feat)
    # backward
    feat = feats_forward[-1]
    feats_backward = []
    for i in range(len(clip)):
        feat = self.mainGUN.convgru_backward(feats_forward[len(clip) - 1 - i], feat)
        feats_backward.append(feat)
    feats_backward = feats_backward[::-1]
    feats = []
    for i in range(len(clip)):
        feat = torch.tanh(
            self.mainGUN.bidirection_conv(torch.cat((feats_forward[i], feats_backward[i]), dim=1)))
        feats.append(feat)
    feats = torch.stack(feats, dim=2)
    feats = self.mainGUN.non_local_block2(feats)

    # branch to NER
    feats_f4 = self.NERbranch(feats_f[4], CM_f.unsqueeze(0), self.branchGUN)
    feats_r4 = self.NERbranch(feats_r[4], CM_r.unsqueeze(0), self.branchGUN)
    feats_b4 = self.NERbranch(feats_b[4], CM_b.unsqueeze(0), self.branchGUN)
    feats_l4 = self.NERbranch(feats_l[4], CM_l.unsqueeze(0), self.branchGUN)
    feats_u4 = self.NERbranch(feats_u[4], CM_u.unsqueeze(0), self.branchGUN)
    feats_d4 = self.NERbranch(feats_d[4], CM_d.unsqueeze(0), self.branchGUN)

    # Decoder: branches
    preds_F = self.branchGUN.backbone.seg_conv(feats_f[1], feats_f[2], feats_f[3], feats_f4, [384, 384])
    preds_R = self.branchGUN.backbone.seg_conv(feats_r[1], feats_r[2], feats_r[3], feats_r4, [384, 384])
    preds_B = self.branchGUN.backbone.seg_conv(feats_b[1], feats_b[2], feats_b[3], feats_b4, [384, 384])
    preds_L = self.branchGUN.backbone.seg_conv(feats_l[1], feats_l[2], feats_l[3], feats_l4, [384, 384])
    preds_U = self.branchGUN.backbone.seg_conv(feats_u[1], feats_u[2], feats_u[3], feats_u4, [384, 384])
    preds_D = self.branchGUN.backbone.seg_conv(feats_d[1], feats_d[2], feats_d[3], feats_d4, [384, 384])

    # Decoder: mainstream
    Lbu1 = self.mainGUN.backbone.refinement1(L3, feats[:, :, 0, :, :])
    Lbu1 = F.interpolate(Lbu1, size=L2.shape[2:], mode="bilinear", align_corners=False)
    Lbu2 = self.mainGUN.backbone.refinement2(L2, Lbu1)
    Lbu2 = F.interpolate(Lbu2, size=L1.shape[2:], mode="bilinear", align_corners=False)
    Lbu3 = self.mainGUN.backbone.refinement3(L1, Lbu2)
    Lbu3 = F.interpolate(Lbu3, size=[256, 512], mode="bilinear", align_corners=False)
    preds_ER = self.mainGUN.backbone.decoder(Lbu3)

    # B, D, F, L, R, U # CM_f, CM_r, CM_b, CM_l, CM_u, CM_d
    preds_branch = []
    preds_branch.append(preds_B[0][0][:, 64:-64, 64:-64])
    preds_branch.append(preds_D[0][0][:, 64:-64, 64:-64])
    preds_branch.append(preds_F[0][0][:, 64:-64, 64:-64])
    preds_branch.append(preds_L[0][0][:, 64:-64, 64:-64])
    preds_branch.append(preds_R[0][0][:, 64:-64, 64:-64])
    preds_branch.append(preds_U[0][0][:, 64:-64, 64:-64])
    preds_branch = torch.stack(preds_branch)
    preds_branch_ER = self.genPreds.ECInteract(preds_branch)

    # refine
    predsConcatenate = torch.cat((preds_ER, preds_branch_ER), dim=1)
    preds_fin = self.refineGUN(predsConcatenate)

    # debug1 = np.squeeze(preds_branch_ER.cpu().data.numpy())
    # cv2.imwrite('debug1.png', debug1 * 255)
    # debug2 = np.squeeze(preds_ER.cpu().data.numpy())
    # cv2.imwrite('debug2.png', debug2 * 255)
    # debug3 = np.squeeze(preds_fin.cpu().data.numpy())
    # cv2.imwrite('debug3.png', debug3 * 255)

    # return preds_fin, preds_F[0], preds_R[0], preds_B[0], preds_L[0], preds_U[0], preds_D[0]
    #return preds_fin

    # Encoder: branches
    feats_f = self.branchGUN.backbone.feat_conv(CM_f)
    feats_r = self.branchGUN.backbone.feat_conv(CM_r)
    feats_b = self.branchGUN.backbone.feat_conv(CM_b)
    feats_l = self.branchGUN.backbone.feat_conv(CM_l)
    feats_u = self.branchGUN.backbone.feat_conv(CM_u)
    feats_d = self.branchGUN.backbone.feat_conv(CM_d)

    # branch to NER
    feats_f4 = self.NERbranch(feats_f[4], CM_f.unsqueeze(0), self.branchGUN)
    feats_r4 = self.NERbranch(feats_r[4], CM_r.unsqueeze(0), self.branchGUN)
    feats_b4 = self.NERbranch(feats_b[4], CM_b.unsqueeze(0), self.branchGUN)
    feats_l4 = self.NERbranch(feats_l[4], CM_l.unsqueeze(0), self.branchGUN)
    feats_u4 = self.NERbranch(feats_u[4], CM_u.unsqueeze(0), self.branchGUN)
    feats_d4 = self.NERbranch(feats_d[4], CM_d.unsqueeze(0), self.branchGUN)

    # Decoder: branches
    preds_F = self.branchGUN.backbone.seg_conv(feats_f[1], feats_f[2], feats_f[3], feats_f4, [384, 384])
    preds_R = self.branchGUN.backbone.seg_conv(feats_r[1], feats_r[2], feats_r[3], feats_r4, [384, 384])
    preds_B = self.branchGUN.backbone.seg_conv(feats_b[1], feats_b[2], feats_b[3], feats_b4, [384, 384])
    preds_L = self.branchGUN.backbone.seg_conv(feats_l[1], feats_l[2], feats_l[3], feats_l4, [384, 384])
    preds_U = self.branchGUN.backbone.seg_conv(feats_u[1], feats_u[2], feats_u[3], feats_u4, [384, 384])
    preds_D = self.branchGUN.backbone.seg_conv(feats_d[1], feats_d[2], feats_d[3], feats_d4, [384, 384])

    # B, D, F, L, R, U # CM_f, CM_r, CM_b, CM_l, CM_u, CM_d
    preds_branch = []
    preds_branch.append(preds_B[0][0][:, 64:-64, 64:-64])
    preds_branch.append(preds_D[0][0][:, 64:-64, 64:-64])
    preds_branch.append(preds_F[0][0][:, 64:-64, 64:-64])
    preds_branch.append(preds_L[0][0][:, 64:-64, 64:-64])
    preds_branch.append(preds_R[0][0][:, 64:-64, 64:-64])
    preds_branch.append(preds_U[0][0][:, 64:-64, 64:-64])
    preds_branch = torch.stack(preds_branch)
    preds_branch_ER = self.genPreds.ECInteract(preds_branch)

    # debug1 = np.squeeze(preds_branch_ER.cpu().data.numpy())
    # cv2.imwrite('debug1.png', debug1 * 255)

   # return preds_branch_ER

