#------------------------ load the necessary packages ------------------------
import os
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

#########################################################################
# PARAMETERS      														#
#########################################################################
update_dts = 0
train_on = 1
test_on =1
size_train = 896
scale_crop = 1
num_train = 51
batch_size = 2
Epochs = 10
debug_on = 1