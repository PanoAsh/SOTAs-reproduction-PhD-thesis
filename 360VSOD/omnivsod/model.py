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

def convert_state_dict_omni(state_dict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name_1 = 'mainGUN.' + k
        state_dict_new[name_1] = v
        if 'backbone' in k:
            name_2 = 'auxBGUN.' + k[9:]
            name_3 = 'auxUGUN.' + k[9:]
            name_4 = 'auxDGUN.' + k[9:]
            state_dict_new[name_2] = v
            state_dict_new[name_3] = v
            state_dict_new[name_4] = v

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
        self.register_parameter('emptyFace', param=None)

    def forward(self, m, b, u, d):
        AuxFeat = []
        for idx in range(self.feat_num):
            back = b[:, idx, :, :]
            up = u[:, idx, :, :]
            down = d[:, idx, :, :]
            empty = self.GenEmptyFace(self.feat_h).cuda()
            CMMap = []
            CMMap.append(back)
            CMMap.append(down)
            CMMap.append(empty)
            CMMap.append(empty)
            CMMap.append(empty)
            CMMap.append(up)
            CMMap = torch.stack(CMMap)
            ERMap = self.ECInteract(CMMap)
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

    def GenEmptyFace(self, feat_h):
        self.emptyFace = nn.Parameter(torch.zeros(1, feat_h, feat_h))

        return self.emptyFace

# OmniVNet
class OmniVNet(nn.Module):
    def __init__(self):
        super(OmniVNet, self).__init__()
        self.mainGUN = VideoModel()
        self.auxBGUN = VideoModel().backbone
        self.auxUGUN = VideoModel().backbone
        self.auxDGUN = VideoModel().backbone

        self.nailGUN_L0 = ECInteract(64, 64)  # the number and height of the feature map
        self.nailGUN_L1 = ECInteract(256, 64)
        self.nailGUN_L2 = ECInteract(512, 32)

    def forward(self, ER, CM_b, CM_u, CM_d):
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
        L2_nailed = self.nailGUN_L2(L2, feats_b[2], feats_u[2], feats_d[2])

        L3 = self.mainGUN.backbone.resnet.layer3(L2_nailed)

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
            feat = torch.tanh(self.mainGUN.bidirection_conv(torch.cat((feats_forward[i], feats_backward[i]), dim=1)))
            feats.append(feat)
        feats = torch.stack(feats, dim=2)
        feats = self.mainGUN.non_local_block2(feats)

        # Decoder: mainstream
        preds_ER = self.mainGUN.backbone.seg_conv(L1_nailed, L2_nailed, L3, feats[:, :, 0, :, :], [256, 512])

        # Decoder: auxiliary branches: including behind, up, down
        preds_Back = self.auxBGUN.seg_conv(feats_b[1], feats_b[2], feats_b[3], feats_b[4], [256, 256])
        preds_Up =  self.auxUGUN.seg_conv(feats_u[1], feats_u[2], feats_u[3], feats_u[4], [256, 256])
        preds_Down = self.auxDGUN.seg_conv(feats_d[1], feats_d[2], feats_d[3], feats_d[4], [256, 256])

        return preds_ER, preds_Back, preds_Up, preds_Down

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
