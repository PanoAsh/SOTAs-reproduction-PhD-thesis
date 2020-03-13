import os.path
import glob
import copy
import multiprocessing
import json

from torch.utils.data import DataLoader
from evaluation.eval import *
from dataloaders import pascal, davis, cityscapes_processed


def param_generateor():
    # change the following parameters for parallel evaluation
    exp_root_dir = './exp'
    method_names = ['run_0001']
    epoch = 60
    threshold_list = [-6, -4, -2, -1, 0, 1, 2, 4]
    F_threshold = (1, 2)

    # Iterate through all the different methods
    for method in method_names:
        # Dataloader
        dataset = pascal.VOCSegmentation(split='val', transform=None, retname=True)
        dataset = davis.DAVIS2016(train=False, transform=None, retname=True)
        dataset = cityscapes_processed.CityScapesProcessed(train=False, split='val', transform=None, retname=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        if epoch is None:
            for i in range(199, 0, -1):
                if os.path.exists(os.path.join(exp_root_dir, method, 'results_ep' + str(i))):
                    epoch = i
                    break
        if epoch is None:
            print(method + ' has no results inside! ')
            continue
        results_folder = os.path.join(exp_root_dir, method, 'results_ep' + str(epoch))

        for threshold in threshold_list:
            filename = os.path.join(exp_root_dir, method,
                                    '{}_ep{}_TH{}.json'.format(method, epoch, threshold))
            yield (dataloader, results_folder, threshold, filename, F_threshold)


def generate_append_param_report(logfile, param):
    f = open(logfile, 'w+')
    for key, val in param.items():
        param[key] = str(val) if len(str(val)) < 1000 else None
    json.dump(param, f, indent=4)
    f.close()


def test_one_setting(params):
    dataloader, results_folder, threshold, filename, F_threshold = params
    results = eval_one_result(dataloader, results_folder, mask_thres=threshold, bd_threshold=F_threshold)
    p = results
    p['mean_jaccards'] = results["all_jaccards"].mean()
    p['folder'] = results_folder
    p['mask_thres'] = threshold
    print("Result for {}: {} (MASK_TH={})".format(results_folder, str.format("{0:.4f}", 100 * p['mean_jaccards']), threshold))
    generate_append_param_report(filename, p)


if __name__ == '__main__':
    #cores = multiprocessing.cpu_count()
    #print('CPU_CORES=' + str(cores))
    pool = multiprocessing.Pool(processes=8)
    pool.map(test_one_setting, param_generateor())


