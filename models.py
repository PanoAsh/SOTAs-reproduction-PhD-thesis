#------------------------ load the necessary packages ------------------------
import torch
import torch.nn as nn
import numpy as np
from utils import debug_vision

#------------------------ define the base model ------------------------
class FCN8s(nn.Module):

    def __init__(self, num_classes, pre_net):
        super(FCN8s, self).__init__()

        self.n_class = num_classes

        self.stage01 = nn.Sequential(*list(pre_net.children())[0][0:17])# 1/8
        self.stage02 = nn.Sequential(*list(pre_net.children())[0][17:24])# 1/16
        self.stage03 = nn.Sequential(*list(pre_net.children())[0][24:31])# 1/32

        self.scores01 = nn.Conv2d(512, num_classes, 1)
        self.scores02 = nn.Conv2d(256, num_classes, 1)

        self.upsampling2x = bilinear_upsampling(num_classes, num_classes, 4, 2, 1)
        self.upsampling8x = bilinear_upsampling(num_classes, num_classes, 16, 8, 4)

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

def bilinear_upsampling(in_channels, out_channels, kernel_size,
                        stride, padding, bias=False):
    initial_weight = get_upsampling_weight(in_channels, out_channels,
                                           kernel_size)
    layer = nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding, bias=bias)
    layer.weight.data.copy_(initial_weight)
    layer.weight.requires_grad = False # weight is frozen

    return layer