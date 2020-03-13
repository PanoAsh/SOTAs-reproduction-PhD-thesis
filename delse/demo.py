import os
import sys
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from torch.nn.functional import upsample

import networks.deeplab_resnet as resnet
from mypath import Path
from dataloaders import helpers as helpers

import networks.backend_cnn as backend_cnn
import argparse
parser = argparse.ArgumentParser()
import torch.nn.functional as F
from layers.lse import levelset_evolution

modelName = 'cityscapes_ckpt'
#modelName = 'MS_DeepLab_resnet_trained_VOC'
pad = 50
thres = 0.8
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

#  Create the network and load the weights
parser.add_argument('--classifier', type=str, default='psp', choices=['atrous', 'psp'])
parser.add_argument('--concat_dim', type=int, default=128, choices=[64, 128], help='concat dim of skip features')
parser.add_argument('--input_channels', type=int, default=4, help='Number of input channels')
parser.add_argument('--backend_cnn', type=str, default='resnet101-skip-pretrain',
                    choices=['resnet101-pretrain', 'resnet101-skip-pretrain'])
parser.add_argument('--resolution', type=int, default=512, help='resolution of input images')
parser.add_argument('--T', type=int, default=5)
parser.add_argument('--dt_max', type=float, default=30)
args = parser.parse_args()
net = backend_cnn.backend_cnn_model(args)

print("Initializing weights from: {}".format(os.path.join(Path.models_dir(), modelName + '.pth')))
state_dict_checkpoint = torch.load(os.path.join(Path.models_dir(), modelName + '.pth'),
                                   map_location=lambda storage, loc: storage)
# Remove the prefix .module from the model when it is trained using DataParallel
if 'module.' in list(state_dict_checkpoint.keys())[0]:
    new_state_dict = OrderedDict()
    for k, v in state_dict_checkpoint.items():
        name = k[7:]  # remove `module.` from multi-gpu training
        new_state_dict[name] = v
else:
    new_state_dict = state_dict_checkpoint
net.load_state_dict(new_state_dict)
net.eval()
net.to(device)

#  Read image and click the points
#image = np.array(Image.open('ims/dog-cat.jpg'))
image = np.array(Image.open('ims/_-Uy5LTocHmoA_1_0206_person_3.png'))
#image = np.array(Image.open('ims/206.jpg'))
plt.ion()
plt.axis('off')
plt.imshow(image)
plt.title('Click the four extreme points of the objects\nHit enter when done (do not close the window)')

results = []

with torch.no_grad():
    while 1:
        extreme_points_ori = np.array(plt.ginput(4, timeout=0)).astype(np.int)
        if extreme_points_ori.shape[0] < 4:
            if len(results) > 0:
                helpers.save_mask(results, 'demo.png')
                print('Saving mask annotation in demo.png and exiting...')
            else:
                print('Exiting...')
            sys.exit()

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad,
                                                                                                                      pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
        inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

        # Run a forward pass
        inputs = inputs.to(device)
        phi_0, energy, g = net.forward(inputs)
        phi_0 = F.upsample(phi_0, size=args.resolution, mode='bilinear', align_corners=True)
        energy = F.upsample(energy, size=args.resolution, mode='bilinear', align_corners=True)
        g = F.sigmoid(F.upsample(g, size=args.resolution, mode='bilinear', align_corners=True))
        phi_T = levelset_evolution(phi_0, energy, g, T=args.T, dt_max=args.dt_max, _test=True)
        phi_T = phi_T.to(torch.device('cpu'))
        phi_T = np.transpose(phi_T.data.numpy()[0, ...], (1, 2, 0))
        phi_T = 1 / (1 + np.exp(-phi_T))
        phi_T = np.squeeze(phi_T)
        result = helpers.crop2fullmask(phi_T, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres

        results.append(result)

        # Plot the results
        plt.imshow(helpers.overlay_masks(image / 255, results))
        plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')
