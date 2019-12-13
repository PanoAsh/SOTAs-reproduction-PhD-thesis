import os

# The project root directory
PROJECT_ROOT = os.getcwd()

# commonly-used path
IMG_PATH = os.path.join(PROJECT_ROOT, 'data', 'img_files')
DATASET_PATH_VR = os.path.join(PROJECT_ROOT, 'data', 'vr')
DATASET_PATH_SITTING = os.path.join(PROJECT_ROOT, 'data', 'seated_vr')
DATASET_PATH_BROWSER = os.path.join(PROJECT_ROOT, 'data', 'browser')
MAP_PATH = os.path.join(PROJECT_ROOT, 'data', 'show_map_')
IOC_PATH = os.path.join(PROJECT_ROOT, 'data', 'IOC_')
IOC_PATH_TT = os.path.join(PROJECT_ROOT, 'data', 'IOC.txt')
SALIENCY_PATH = os.path.join(PROJECT_ROOT, 'saliency')
ENTROPY_PATH = os.path.join(PROJECT_ROOT, 'Entropy.txt')

L_PATH_RAW = PROJECT_ROOT + '/fixpos/scanpath_nante/L/'
R_PATH_RAW = PROJECT_ROOT + '/fixpos/scanpath_nante/R/'
L_PATH_TGT = PROJECT_ROOT + '/fixpos/fixations/L/'
R_PATH_TGT = PROJECT_ROOT + '/fixpos/fixations/R/'

CMP_PATH = PROJECT_ROOT + '/cubmaps/'
ERP_PATH = PROJECT_ROOT + '/stimulis/'

TRAIN_TXT_PATH = PROJECT_ROOT + '/train.txt'
TEST_TXT_PATH = PROJECT_ROOT + '/test.txt'

EASY_TXT_PATH = PROJECT_ROOT + '/easy.txt'
MEDIUM_TXT_PATH = PROJECT_ROOT + '/medium.txt'
HARD_TXT_PATH = PROJECT_ROOT + '/hard.txt'

PANOISOD_IMG_PATH = PROJECT_ROOT + '/360ISOD_img/'
PANOISOD_MSK_PATH = PROJECT_ROOT + '/360ISOD_msk/'
PANOISOD_IMG_TRAIN_PATH = PROJECT_ROOT + '/360ISOD_img_train/'
PANOISOD_MSK_TRAIN_PATH = PROJECT_ROOT + '/360ISOD_msk_train/'
PANOISOD_IMG_TEST_PATH = PROJECT_ROOT + '/360ISOD_img_test/'
PANOISOD_MSK_TEST_PATH = PROJECT_ROOT + '/360ISOD_msk_test/'

IOC_2_PATH_TT = PROJECT_ROOT + '/IOC_Nantes_total.txt'
IOC_2_PATH = PROJECT_ROOT + '/IOC_'

# Default of the 360ISOD dataset
width_360ISOD = 2048
height_360ISOD = 1024