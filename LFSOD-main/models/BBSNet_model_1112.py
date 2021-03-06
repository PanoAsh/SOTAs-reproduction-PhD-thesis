import torch
import torch.nn as nn
import torchvision.models as models
from models.ResNet import ResNet50
from models.conv_lstm import ConvLSTM
from torch.nn import functional as F
import cv2
import numpy as np


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
        self.bilinear_1 = nn.UpsamplingBilinear2d(size=(22, 22))
        self.bilinear_2 = nn.UpsamplingBilinear2d(size=(44, 44))

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

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

#Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

#aggregation of the high-level(teacher) features
class aggregation_init(nn.Module):

    def __init__(self, channel):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

#aggregation of the low-level(student) features
class aggregation_final(nn.Module):

    def __init__(self, channel):
        super(aggregation_final, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x1)) \
               * self.conv_upsample3(x2) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2

#Refinement flow
class Refine(nn.Module):
    def __init__(self):
        super(Refine,self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, attention, x1, x2, x3):
        #Note that there is an error in the manuscript. In the paper, the refinement strategy is depicted as ""f'=f*S1"", it should be ""f'=f+f*S1"".
        x1 = x1+torch.mul(x1, self.upsample2(attention))
        x2 = x2+torch.mul(x2,self.upsample2(attention))
        x3 = x3+torch.mul(x3,attention)

        return x1, x2, x3
    
#BBSNet
class BBSNet(nn.Module):
    def __init__(self, channel=32):
        super(BBSNet, self).__init__()
        
        #Backbone model
        self.resnet = ResNet50('rgb')
        self.resnet_depth = ResNet50('rgbd')
        self.resnet_fs = ResNet50('fs')
        self.convLSTM_0 = ConvLSTM(input_size=88, output_channels=64, input_channels=64, hidden_channels=[64, 32, 64],
                                 kernel_size=5, step=4, effective_step=[2, 4, 8])
        self.pre_convLSTM_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=True)
        self.convLSTM_1 = ConvLSTM(input_size=88, output_channels=256, input_channels=64, hidden_channels=[64, 32, 64],
                                 kernel_size=5, step=4, effective_step=[2, 4, 8])
        self.pre_convLSTM_2 = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=True)
        self.convLSTM_2 = ConvLSTM(input_size=44, output_channels=512, input_channels=64, hidden_channels=[64, 32, 64],
                                 kernel_size=5, step=4, effective_step=[2, 4, 8])
        self.pre_convLSTM_3 = nn.Conv2d(1024, 64, kernel_size=1, stride=1, bias=True)
        self.convLSTM_3 = ConvLSTM(input_size=22, output_channels=1024, input_channels=64, hidden_channels=[64, 32, 64],
                                   kernel_size=5, step=4, effective_step=[2, 4, 8])
        self.pre_convLSTM_4 = nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=True)
        self.convLSTM_4 = ConvLSTM(input_size=11, output_channels=2048, input_channels=64, hidden_channels=[64, 32, 64],
                                   kernel_size=5, step=4, effective_step=[2, 4, 8])
        self.pre_convLSTM_fin = nn.Conv2d(1, 64, kernel_size=1, stride=1, bias=True)
        self.convLSTM_fin = ConvLSTM(input_size=44, output_channels=1, input_channels=64, hidden_channels=[64, 32, 64],
                                   kernel_size=5, step=4, effective_step=[2, 4, 8])

        #Decoder 1
        self.rfb2_1 = GCM(512, channel)
        self.rfb3_1 = GCM(1024, channel)
        self.rfb4_1 = GCM(2048, channel)
        self.agg1 = aggregation_init(channel)

        self.rfb2_1_fs = GCM(512, channel)
        self.rbf3_1_fs = GCM(1024, channel)
        self.rbf4_1_fs = GCM(2048, channel)
        self.agg1_fs = aggregation_init(channel)

        #Decoder 2
        self.rfb0_2 = GCM(64, channel)
        self.rfb1_2 = GCM(256, channel)
        self.rfb5_2 = GCM(512, channel)
        self.agg2 = aggregation_final(channel)

        #upsample function
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #Refinement flow
        self.HA = Refine()

        #Components of DEM module
        self.atten_x_channel_0 = ChannelAttention(64)
        self.atten_x_channel_1 = ChannelAttention(256)
        self.atten_x_channel_2 = ChannelAttention(512)
        self.atten_x_channel_3_1 = ChannelAttention(1024)
        self.atten_x_channel_4_1 = ChannelAttention(2048)

        self.atten_depth_channel_0 = ChannelAttention(64)
        self.atten_depth_channel_1 = ChannelAttention(256)
        self.atten_depth_channel_2 = ChannelAttention(512)
        self.atten_depth_channel_3_1 = ChannelAttention(1024)
        self.atten_depth_channel_4_1 = ChannelAttention(2048)

        self.atten_fs_channel_0 = ChannelAttention(64)
        self.atten_fs_channel_1 = ChannelAttention(256)
        self.atten_fs_channel_2 = ChannelAttention(512)
        self.atten_fs_channel_3_1 = ChannelAttention(1024)
        self.atten_fs_channel_4_1 = ChannelAttention(2048)

        self.atten_x_spatial_0 = SpatialAttention()
        self.atten_x_spatial_1 = SpatialAttention()
        self.atten_x_spatial_2 = SpatialAttention()
        self.atten_x_spatial_3_1 = SpatialAttention()
        self.atten_x_spatial_4_1 = SpatialAttention()

        self.atten_depth_spatial_0 = SpatialAttention()
        self.atten_depth_spatial_1 = SpatialAttention()
        self.atten_depth_spatial_2 = SpatialAttention()
        self.atten_depth_spatial_3_1 = SpatialAttention()
        self.atten_depth_spatial_4_1 = SpatialAttention()

        self.atten_fs_spatial_0 = SpatialAttention()
        self.atten_fs_spatial_1 = SpatialAttention()
        self.atten_fs_spatial_2 = SpatialAttention()
        self.atten_fs_spatial_3_1 = SpatialAttention()
        self.atten_fs_spatial_4_1 = SpatialAttention()

        #Components of PTM module
        self.inplanes = 32*2
        self.deconv1 = self._make_transpose(TransBasicBlock, 32*2, 3, stride=2)
        self.inplanes =32
        self.deconv2 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.agant1 = self._make_agant_layer(32*3, 32*2)
        self.agant2 = self._make_agant_layer(32*2, 32)
        self.out0_conv = nn.Conv2d(32*3, 1, kernel_size=1, stride=1, bias=True)
        self.out1_conv = nn.Conv2d(32*2, 1, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(32*1, 1, kernel_size=1, stride=1, bias=True)

        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth, x_fss):
        x_fss = self.resnet_fs.conv1(x_fss)
        x_fss = self.resnet_fs.bn1(x_fss)
        x_fss = self.resnet_fs.relu(x_fss)
        x_fss = self.resnet_fs.maxpool(x_fss)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)

        # layer0 merge
        temp = x_depth.mul(self.atten_depth_channel_0(x_depth))
        temp = temp.mul(self.atten_depth_spatial_0(temp))
        temp_fss = self.convLSTM_0(x_fss)
        temp_fss = temp_fss.mul(self.atten_fs_channel_0(temp_fss))
        temp_fss = temp_fss.mul(self.atten_fs_spatial_0(temp_fss))
        temp_x = x.mul(self.atten_x_channel_0(x))
        temp_x = temp_x.mul(self.atten_x_spatial_0(temp_x))
        x = x + temp + temp_fss
        x_fss = x_fss + temp + temp_x + temp_fss
        # layer0 merge end

        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x1_depth = self.resnet_depth.layer1(x_depth)
        x1_fss = self.resnet_fs.layer1(x_fss)

        # layer1 merge
        temp = x1_depth.mul(self.atten_depth_channel_1(x1_depth))
        temp = temp.mul(self.atten_depth_spatial_1(temp))
        temp_fss = self.pre_convLSTM_1(x1_fss)
        temp_fss = self.convLSTM_1(temp_fss)
        temp_fss = temp_fss.mul(self.atten_fs_channel_1(temp_fss))
        temp_fss = temp_fss.mul(self.atten_fs_spatial_1(temp_fss))
        temp_x = x1.mul(self.atten_x_channel_1(x1))
        temp_x = temp_x.mul(self.atten_x_spatial_1(temp_x))
        x1 = x1 + temp + temp_fss
        x1_fss = x1_fss + temp + temp_x + temp_fss
        # layer1 merge end

        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x2_depth = self.resnet_depth.layer2(x1_depth)
        x2_fss = self.resnet_fs.layer2(x1_fss)

        # layer2 merge
        temp = x2_depth.mul(self.atten_depth_channel_2(x2_depth))
        temp = temp.mul(self.atten_depth_spatial_2(temp))
        temp_fss = self.pre_convLSTM_2(x2_fss)
        temp_fss = self.convLSTM_2(temp_fss)
        temp_fss = temp_fss.mul(self.atten_fs_channel_2(temp_fss))
        temp_fss = temp_fss.mul(self.atten_fs_spatial_2(temp_fss))
        temp_x = x2.mul(self.atten_x_channel_2(x2))
        temp_x = temp_x.mul(self.atten_x_spatial_2(temp_x))
        x2 = x2 + temp + temp_fss
        x2_fss = x2_fss + temp + temp_x + temp_fss
        # layer2 merge end

        x2_1 = x2

        x3_1 = self.resnet.layer3_1(x2_1)  # 1024 x 16 x 16
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)
        x3_1_fss = self.resnet_fs.layer3_1(x2_fss)

        # layer3_1 merge
        temp = x3_1_depth.mul(self.atten_depth_channel_3_1(x3_1_depth))
        temp = temp.mul(self.atten_depth_spatial_3_1(temp))
        temp_fss = self.pre_convLSTM_3(x3_1_fss)
        temp_fss = self.convLSTM_3(temp_fss)
        temp_fss = temp_fss.mul(self.atten_fs_channel_3_1(temp_fss))
        temp_fss = temp_fss.mul(self.atten_fs_spatial_3_1(temp_fss))
        temp_x = x3_1.mul(self.atten_x_channel_3_1(x3_1))
        temp_x = temp_x.mul(self.atten_x_spatial_3_1(temp_x))
        x3_1 = x3_1 + temp + temp_fss
        x3_1_fss = x3_1_fss + temp + temp_x + temp_fss
        # layer3_1 merge end

        x4_1 = self.resnet.layer4_1(x3_1)  # 2048 x 8 x 8
        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)
        x4_1_fss = self.resnet_fs.layer4_1(x3_1_fss)

        # layer4_1 merge
        temp = x4_1_depth.mul(self.atten_depth_channel_4_1(x4_1_depth))
        temp = temp.mul(self.atten_depth_spatial_4_1(temp))
        temp_fss = self.pre_convLSTM_4(x4_1_fss)
        temp_fss = self.convLSTM_4(temp_fss)
        temp_fss = temp_fss.mul(self.atten_fs_channel_4_1(temp_fss))
        temp_fss = temp_fss.mul(self.atten_fs_spatial_4_1(temp_fss))
        temp_x = x4_1.mul(self.atten_x_channel_4_1(x4_1))
        temp_x = temp_x.mul(self.atten_x_spatial_4_1(temp_x))
        x4_1 = x4_1 + temp + temp_fss
        x4_1_fss = x4_1_fss + temp + temp_x + temp_fss
        # layer4_1 merge end

        # produce initial saliency map by decoder1
        x2_1 = self.rfb2_1(x2_1)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        attention_map = self.agg1(x4_1, x3_1, x2_1)

        x2_fss = self.rfb2_1_fs(x2_fss)
        x3_1_fss = self.rbf3_1_fs(x3_1_fss)
        x4_1_fss = self.rbf4_1_fs(x4_1_fss)
        attention_map_fss = self.agg1_fs(x4_1_fss, x3_1_fss, x2_fss)
        attention_map_fss = self.pre_convLSTM_fin (attention_map_fss)
        attention_map_fss = self.convLSTM_fin(attention_map_fss)

        # Refine low-layer features by initial map
        x, x1, x5 = self.HA(attention_map.sigmoid(), x, x1, x2)

        # produce final saliency map by decoder2
        x0_2 = self.rfb0_2(x)
        x1_2 = self.rfb1_2(x1)
        x5_2 = self.rfb5_2(x5)
        y = self.agg2(x5_2, x1_2, x0_2)  # *4

        # PTM module
        y = self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)

        return self.upsample(attention_map), y, self.upsample(attention_map_fss)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
    
    #initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k=='conv1.weight':
                all_params[k]=torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)

