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

modelName = 'dextr_pascal-sbd'
pad = 0
thres = 0.8
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

#  Create the network and load the weights
net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
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
image = np.array(Image.open('ims/_-Uy5LTocHmoA_1_0206_person_2.png'))
#image = np.array(Image.open('ims/206.jpg'))

img_shape = image.shape
img_H = img_shape[0]
img_W = img_shape[1]

with torch.no_grad():
    resize_image = helpers.fixed_resize(image, (512, 512)).astype(np.float32)
    extreme_heatmap = np.zeros((512, 512), np.float32)

    #  Concatenate inputs and convert to tensor
    input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
    inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

    # Run a forward pass
    inputs = inputs.to(device)
    outputs = net.forward(inputs)
    outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
    outputs = outputs.to(torch.device('cpu'))

    pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
    pred = 1 / (1 + np.exp(-pred))
    pred = np.squeeze(pred)
    bbox = (0, 0, img_W, img_H)
    result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres
    results = []
    results.append(result)
    helpers.save_mask(results, 'demo.png')
    print('Saving mask annotation in demo.png and exiting...')


