#------------------------ load the necessary packages ------------------------
import torch
import torch.nn as nn
import numpy as np
from utils import debug_vision

#------------------------ define the base model ------------------------
class FCN8s(nn.Module):

    def __init__(self, num_classes, pre_net):
        super(FCN8s, self).__init__()

        self.upsampling2x = bilinear_upsampling(num_classes, num_classes, 4, 2, 1)
        self.upsampling2x.initialize_weights()
        self.upsampling8x = bilinear_upsampling(num_classes, num_classes, 16, 8, 4)
        self.upsampling8x.initialize_weights()
        for p in self.parameters(): # freeze the upsampling layers
            p.requires_grad = False

        self.stage01 = nn.Sequential(*list(pre_net.children())[0][0:17])# 1/8
        self.stage02 = nn.Sequential(*list(pre_net.children())[0][17:24])# 1/16
        self.stage03 = nn.Sequential(*list(pre_net.children())[0][24:31])# 1/32

        self.scores01 = nn.Conv2d(512, num_classes, 1)
        self.scores02 = nn.Conv2d(256, num_classes, 1)

        self.classifier = nn.Sigmoid()

    def forward(self, x):
        x = self.stage01(x)
        s1 = x# 1/8

        x = self.stage02(x)
        s2 = x# 1/16

        x = self.stage03(x)
        s3 = x# 1/32

        s3 = self.scores01(s3)
        s3 = self.upsampling2x(s3)# 1/16
        s2 = self.scores01(s2)
        s2 = s2 + s3# 1/16

        s2 = self.upsampling2x(s2)# 1/8
        s1 = self.scores02(s1)
        s1 = s1 + s2# 1/8

        s = self.upsampling8x(s1)
        s = self.classifier(s)

        return s

#------------------------ initialize the bilinear upsampling ------------------------
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    Make a 2D bilinear kernel suitable for unsampling
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) \
                      * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter

    return torch.from_numpy(weight).float()

class bilinear_upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                        stride, padding, bias=False):
        super(bilinear_upsampling, self).__init__()

        self.CI = in_channels
        self.CO = out_channels
        self.ks = kernel_size
        self.convTrans = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size,stride,
                                            padding, bias)

    def forward(self, x):
        x = self.convTrans(x)

        return x

    def initialize_weights(self):
        initial_weight = get_upsampling_weight(self.CI, self.CO, self.ks)

        self.convTrans.weight.data.copy_(initial_weight)

# ------------------------ define the U net model ------------------------
class unet(nn.Module):

    def __init__(self):
        super(unet, self).__init__()

        self.layer1_1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.layer1_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.layer1_3 = nn.MaxPool2d(2, 2) # 1/2

        self.layer2_1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.layer2_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer2_3 = nn.MaxPool2d(2, 2) # 1/4

        self.layer3_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.layer3_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.layer3_3 = nn.MaxPool2d(2, 2) # 1/8

        self.layer4_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.layer4_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer4_3 = nn.MaxPool2d(2, 2) # 1/16

        self.layer5_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.layer5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer5_3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)

        self.layer6_1 = nn.Conv2d(512, 256, 3, 1, 1)
        self.layer6_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer6_3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)

        self.layer7_1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.layer7_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.layer7_3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)

        self.layer8_1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.layer8_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer8_3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)

        self.layer9_1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.layer9_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.layer9_3 = nn.Conv2d(32, 1, 1)

        self.classifier = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        l1 = x
        x = self.layer1_3(x) # 1/2

        x = self.layer2_1(x)
        x = self.layer2_2(x)
        l2 = x
        x = self.layer2_3(x) # 1/4

        x = self.layer3_1(x)
        x = self.layer3_2(x)
        l3 = x
        x = self.layer3_3(x) # 1/8

        x = self.layer4_1(x)
        x = self.layer4_2(x)
        l4 = x
        x = self.layer4_3(x) # 1/16

        x = self.layer5_1(x)
        x = self.layer5_2(x)
        x = self.layer5_3(x) # 1/8

        x = torch.cat([x, l4], dim=1)
        x = self.layer6_1(x)
        x = self.layer6_2(x)
        x = self.layer6_3(x) # 1/4

        x = torch.cat([x, l3], dim=1)
        x = self.layer7_1(x)
        x = self.layer7_2(x)
        x = self.layer7_3(x) # 1/2

        x = torch.cat([x, l2], dim=1)
        x = self.layer8_1(x)
        x = self.layer8_2(x)
        x = self.layer8_3(x)

        x = torch.cat([x, l1], dim=1)
        x = self.layer9_1(x)
        x = self.layer9_2(x)
        x = self.layer9_3(x)

        x = self.classifier(x)

        return x