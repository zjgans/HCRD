from __future__ import print_function

import argparse
import socket
import time
import os
import mkl


import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import MetaImageNet
from dataset.tiered_imagenet import MetaTieredImageNet
from dataset.cifar import MetaCIFAR100
from dataset.transform_cfg import transforms_test_options, transforms_list
from util import set_gpu

from eval.meta_eval import meta_test
from eval.cls_eval import validate, embedding
from dataloader import get_eval_dataloader,get_dataloaders

mkl.set_num_threads(2)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_path', type=str, default="./save/S:resnet12_T:resnet12_CIFAR-FS_kd_r:1_a:0.5_dist:1_rkd:1_gT_:0.07_Lt:0.07_mixT:0.07_tag_ ssl_test_dist_kl/ckpt_epoch_90.pth",
                        help='absolute path to .pth model')

    # dataset   ckpt_epoch_90.pth
    parser.add_argument('--dataset', type=str, default='CIFAR-FS', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100', 'cub','cars','places'])
    parser.add_argument('--cross_dataset', type=str, default='places', choices=['cub', 'cars', 'places', 'plantae'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

    # specify data_root
    parser.add_argument('--data_root_path', type=str, default='/data/lxj/odata/dataset', help='path to data root')
    parser.add_argument('--simclr', type=bool, default=False, help='use simple contrastive learning representation')
    parser.add_argument('--trans', type=int, default=12, help='number of transformations')
    parser.add_argument('--proj_dim', type=float, default=128)
    parser.add_argument('--cross_domain', type=bool, default=False)

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('-gpu', default='1', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    opt = parser.parse_args()
    
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'
    elif opt.dataset in ['cub', 'cars', 'places', 'plantae']:
        opt.transform = 'C'
        
    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    # set the path according to the environment
    if not opt.data_root_path:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root_path, opt.dataset)
    if opt.cross_domain:
        opt.out_data_root = '{}/{}'.format(opt.data_root_path, opt.cross_dataset)
    opt.data_aug = True

    opt.num_gpu = set_gpu(opt)
    opt.device_ids = None if opt.gpu == '-1' else list(range(opt.num_gpu))
    opt.n_gpu = torch.cuda.device_count()

    return opt

def main():

    opt = parse_option()

    meta_testloader, meta_valloader, n_cls = get_eval_dataloader(opt)

    # load model
    model = create_model(opt.model, n_cls, opt.dataset,opt.proj_dim,opt.trans)
    ckpt = torch.load(opt.model_path)["model"]

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace("module.","")
        new_state_dict[name]=v

    model.load_state_dict(new_state_dict)

    # model.load_state_dict(ckpt["model"])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    start = time.time()
    test_acc_feat, test_std_feat = meta_test(model, meta_testloader, is_feat=True)
    test_time = time.time() - start
    print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat, test_std_feat, test_time))

    start = time.time()
    test_acc, test_std = meta_test(model, meta_testloader)
    test_time = time.time() - start
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))

if __name__ == '__main__':
    main()
