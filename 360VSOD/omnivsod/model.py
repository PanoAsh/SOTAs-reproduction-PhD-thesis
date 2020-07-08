import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
import torchvision

# GLOmni network
class GLOmni_bone(nn.Module):
    def __init__(self, base_2d):
        super(GLOmni_bone, self).__init__()
        self.base_2d = base_2d

    def forward(self, x):
        x = self.base_2d(x)

        return x

# build the whole network
def build_model():
    base_2d = torchvision.models.segmentation.fcn_resnet101(pretrained=False, num_classes=1)
   # base_2d = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)

    return GLOmni_bone(base_2d)


if __name__ == '__main__':
    from torch.autograd import Variable
    net = build_model().cuda()
    img = Variable(torch.randn((1, 3, 512, 512))).cuda()
    out = net(img)
    print(len(out))
    print(len(out[0]))
    print(out[0].shape)
    print(len(out[1]))
    print(net)
    input('Press Any to Continue...')
