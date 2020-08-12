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
