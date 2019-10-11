#------------------------ load the necessary packages ------------------------
import os
import torch
pj_path = os.getcwd()

#########################################################################
# PATHS          														#
#########################################################################

# paths of omnidirectional images
imgs_train_path = pj_path + '/Data/TrainSet/Imgs'
imgs_val_path = pj_path + '/Data/ValSet/Imgs'
imgs_test_path = pj_path + '/Data/TestSet/Imgs'

# paths of corresponding object-level maps
objms_train_path = pj_path + '/Data/TrainSet/Object GT'
objms_val_path = pj_path + '/Data/ValSet/Object GT'
objms_test_path = pj_path + '/Data/TestSet/Object GT'

# paths of corresponding instance-level maps
insms_train_path = pj_path + '/Data/TrainSet/Instance GT/GT'
insms_val_path = pj_path + '/Data/ValSet/Instance GT/GT'
insms_test_path = pj_path + '/Data/TestSet/Instance GT/GT'

# paths of corresponding instance indexs
insids_train_path = pj_path + '/Data/TrainSet/Instance GT/Instance Name'
insids_val_path = pj_path + '/Data/ValSet/Instance GT/Instance Name'
insids_test_path = pj_path + '/Data/TestSet/Instance GT/Instance Name'

# paths of attributes
insfts_train_path = pj_path + '/Data/TrainSet/Instance GT/Attributes'
insfts_val_path = pj_path + '/Data/ValSet/Instance GT/Attributes'
insfts_test_path = pj_path + '/Data/TestSet/Instance GT/Attributes'

# paths of TXT files
ids_imgs_train_path = pj_path + '/Data/imgs_train.txt'
ids_imgs_val_path = pj_path + '/Data/imgs_val.txt'
ids_imgs_test_path = pj_path + '/Data/imgs_test.txt'
ids_objms_train_path = pj_path + '/Data/objms_train.txt'
ids_objms_val_path = pj_path + '/Data/objms_val.txt'
ids_objms_test_path = pj_path + '/Data/objms_test.txt'

# path of log for model training
log_dir = pj_path + '/Log'

# path for model saving
mod_dir = pj_path + '/Log/net_params.pkl'

#########################################################################
# PARAMETERS      														#
#########################################################################
update_dts = 0
train_on = 1
test_on = 0
size_train = 784
scale_crop = 1
num_train = 51
batch_size = 4
Epochs = 100
debug_on = 1
lr_init = 0.001
epsilon_DC = 0.001
step_num = 35
pretrain_on = 1
num_classes = 1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
model_select = 'unet'