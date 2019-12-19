import argparse
import os
from dataset import get_loader
from solver import Solver


def main(config):
    if config.mode == 'train':
        train_loader, dataset = get_loader(config.batch_size, num_thread=config.num_thread)
        run = "nnet"
        if not os.path.exists("%s/run-%s" % (config.save_fold, run)): 
            os.mkdir("%s/run-%s" % (config.save_fold, run))
            os.mkdir("%s/run-%s/logs" % (config.save_fold, run))
            os.mkdir("%s/run-%s/models" % (config.save_fold, run))
        config.save_fold = "%s/run-%s" % (config.save_fold, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        test_loader, dataset = get_loader(config.test_batch_size, mode='test',num_thread=config.num_thread, test_mode=config.test_mode, sal_mode=config.sal_mode)

        test = Solver(None, test_loader, config, dataset.save_folder())
        test.test(test_mode=config.test_mode)
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':

    vgg_path = os.getcwd() + '/pretrained/vgg16_20M.pth'
    resnet_path = os.getcwd() + '/pretrained/resnet50_caffe.pth'
    resnet_pretrained_path = os.getcwd() + '/pretrained/epoch_resnet.pth'

    test_model_path = os.getcwd() + '/results/models/final.pth'
    test_save_path = os.getcwd() + '/results/predicted/'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)

    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--vgg', type=str, default=vgg_path)
    parser.add_argument('--resnet', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=50) # 12, now x3
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load_bone', type=str, default='')
    # parser.add_argument('--load_branch', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./results')
    # parser.add_argument('--epoch_val', type=int, default=20)
    parser.add_argument('--epoch_save', type=int, default=9) # 2, now x3
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=resnet_pretrained_path)

    # Testing settings
    parser.add_argument('--model', type=str, default=test_model_path)
    parser.add_argument('--test_fold', type=str, default=test_save_path)
    parser.add_argument('--test_mode', type=int, default=3)
    parser.add_argument('--sal_mode', type=str, default='vr') # useless for 360ISOD, set to []

    # Misc
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)
    
    config = parser.parse_args()

    main(config)
