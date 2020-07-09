import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# GLOmni network
class GLOmniNet(nn.Module):
    def __init__(self, base_2d):
        super(GLOmniNet, self).__init__()
        self.base_2d = base_2d

    def forward(self, x):
        x = self.base_2d(x)['out']

        return x

# build the whole network
def build_model(backbone_config, coco_model, mode):
    if backbone_config == 'fcn_resnet101':
        base_2d = torchvision.models.segmentation.fcn_resnet101(pretrained=False, num_classes=1)

        if mode == 'train':
            base_2d_dict = base_2d.state_dict()
            state_dict = torch.load(coco_model)
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k in base_2d_dict and v.shape == base_2d_dict[k].shape
            }
            base_2d.load_state_dict(state_dict, strict=False)

    if backbone_config == 'deeplabv3_resnet101':
        base_2d = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)

    return GLOmniNet(base_2d)


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
