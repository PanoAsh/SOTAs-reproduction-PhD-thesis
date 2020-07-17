import argparse
import os
from dataset import get_loader
from solver import Solver
from torch.multiprocessing import set_start_method


def main(config):
    set_start_method('spawn')

    if config.mode == 'train':
        train_loader, dataset = get_loader(config.batch_size, num_thread=config.num_thread,
                                           data_type=config.model_type, base_level=config.base_level,
                                           sample_level=config.sample_level)
        run = "omnivsod"
        if not os.path.exists("%s/run-%s" % (config.save_fold, run)): 
            os.mkdir("%s/run-%s" % (config.save_fold, run))
            os.mkdir("%s/run-%s/logs" % (config.save_fold, run))
            os.mkdir("%s/run-%s/models" % (config.save_fold, run))
        config.save_fold = "%s/run-%s" % (config.save_fold, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        test_loader, dataset = get_loader(config.test_batch_size, mode='test', num_thread=config.num_thread,
                                          data_type=config.model_type, base_level=config.base_level,
                                          sample_level=config.sample_level)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':

    fcn_resnet101_path = os.getcwd() + '/pretrained/fcn_resnet101_coco-7ecb50ca.pth'
    deeplabv3_resnet101_path = os.getcwd() + '/pretrained/deeplabv3_resnet101_coco-586e9e4e.pth'

    pretrained_path = os.getcwd() + '/pretrained/epoch_4_bone.pth'

    test_model_path = os.getcwd() + '/results/models/epoch_9_bone.pth'
    test_save_path = os.getcwd() + '/results/sal_predicted/'

    parser = argparse.ArgumentParser()

    # Training settings
    # GPU
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--cuda', type=bool, default=True)

    #Backbones
    parser.add_argument('--backbone', type=str, default='fcn_resnet101') # or deeplabv3_resnet101
    parser.add_argument('--fcn', type=str, default=fcn_resnet101_path)
    parser.add_argument('--deeplab', type=str, default=deeplabv3_resnet101_path)
    #parser.add_argument('--min_inputSize', type=int, default=224) # the minimum input size for fcn_resnet101 backbone

    # Hyper_parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1) # must be 1
    parser.add_argument('--lr', type=int, default=0.00001)
    parser.add_argument('--wd', type=int, default=0.0005)
    parser.add_argument('--nAveGrad', type=int, default=10)
    parser.add_argument('--lr_decay_epoch', type=int, default=10)


    # Recording & Visualization
    parser.add_argument('--pre_trained', type=str, default=pretrained_path)
    parser.add_argument('--save_fold', type=str, default='./results')
    parser.add_argument('--showEvery', type=int, default=100)
    parser.add_argument('--epoch_save', type=int, default=1)
    parser.add_argument('--epoch_show', type=int, default=1)

    # Testing settings
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--model', type=str, default=test_model_path)
    parser.add_argument('--test_fold', type=str, default=test_save_path)

    # Mode
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--base_level', type=int, default=0)  # for tangent image branch
    parser.add_argument('--sample_level', type=int, default=7) # for tangent image branch / comparison with 2D SOTAs
    parser.add_argument('--model_type', type=str, default='G') # L for TI-based trainig/testing
    
    config = parser.parse_args()

    main(config)
