import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# GLOmni network
class GTNet(nn.Module):
    def __init__(self, base):
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

    def forward(self, x):
        x = self.base(x)['out']

        return x

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
def build_model(backbone_config, coco_model, mode):
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

    return GTNet(base)


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

    plt.imshow(img_tensor.cpu().detach().numpy()[0,0,:,:], cmap='gray')
    plt.show()

    out = net(img_tensor)[0].cpu().detach().numpy()
    plt.imshow(out[0,:,:], cmap='gray')
    plt.show()
