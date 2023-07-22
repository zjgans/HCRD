from __future__ import print_function

import os
import argparse
import socket
import time
import wandb
import sys
import mkl
import numpy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import collections
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable
from dataset.transform_cfg import transforms_options, transforms_list
from models import model_pool
from models.util import create_model
from util import adjust_learning_rate, accuracy, AverageMeter,Meter, rotrate_concat, Logger, generate_final_report, \
    restart_from_checkpoint,data_write_csv,get_model_complexity,set_gpu,to_one_hot,distance,mix_data_lab,record_data
from eval.meta_eval import meta_test
from local_branch.base_hybird import Aggregation
from dataloader import get_dataloaders
from dataset.utils import Trans_to_Num,rot_color_transformation

mkl.set_num_threads(2)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--eval_freq', type=int, default=5, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=5, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=90, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100','cub'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', type=bool, help='use trainval set')
    parser.add_argument('--use_resume', action='store_true', help='use the result of training before')
    parser.add_argument('--resume_file', type=str, default='ckpt_epoch_10.pth')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    # specify folder
    parser.add_argument('--model_path', type=str, default='./save', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='./tb', help='path to tensorboard')
    parser.add_argument('--record_path',type=str,default='./record',help='record the data of results')
    parser.add_argument('--data_root', type=str, default='/~/dataset', help='path to data root')
    parser.add_argument('--data_root_path', type=str, default='/~/dataset', help='path to data root')
    parser.add_argument('--cross_domain', type=bool, default=False)

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N', help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')

    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size', help='Size of test batch)')
    parser.add_argument('-t', '--trial', type=str, default='simple_ce', help='the experiment id')

    # hyper parameters

    parser.add_argument('--num_classes',type=int,default=64)
    parser.add_argument('--image_size',type=int,default=84)

    parser.add_argument('-gpu', default='2', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')

    opt = parser.parse_args()
    # opt.trans = Trans_to_Num[opt.trans_type]
    opt.trans = 1
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'
        opt.crop_size = 32
        opt.image_size = 32

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_pretrained'
    if not opt.tb_path:
        opt.tb_path = './tensorboard'
    if not opt.data_root_path:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root_path, opt.dataset)
    if opt.cross_domain:
        opt.out_data_root = '{}/{}'.format(opt.data_root_path, opt.cross_dataset)
    if opt.record_path:
        opt.record_path = './record/{}'.format(opt.dataset)
    else:
        opt.record_path = '{}/{}'.format(opt.record_path,opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(opt.model, opt.dataset, opt.learning_rate, opt.weight_decay,
                                                            opt.transform)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)

    opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if not os.path.isdir(opt.record_path):
        os.makedirs(opt.record_path)
    opt.record_file = os.path.join(opt.record_path,opt.trial+'.csv')

    opt.num_gpu = set_gpu(opt)
    opt.device_ids = None if opt.gpu == '-1' else list(range(opt.num_gpu))
    opt.n_gpu = torch.cuda.device_count()

    # extras
    opt.fresh_start = True
    return opt

def main():
    opt = parse_option()
    wandb.init(project=opt.model_path.split("/")[-1], tags='training')
    wandb.config.update(opt)
    wandb.save('*.py')
    wandb.run.save()
    print(opt.save_folder)

    train_loader,meta_testloader, meta_valloader, n_cls = get_dataloaders(opt)
    opt.num_classes = n_cls

    # model
    model = create_model(opt.model, n_cls, opt.dataset)

    wandb.watch(model)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        if opt.n_gpu > 1:
           model = nn.DataParallel(model,device_ids=opt.device_ids)
        cudnn.benchmark = True

    # optimizer
    if opt.adam:
        print("Adam")
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        print("SGD")
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    to_restore = {'epoch': 0}
    if opt.use_resume:
        print('------load the parameters from  checkpoint--------')
        restart_from_checkpoint(
            os.path.join(opt.save_folder, opt.resume_file),
            run_variables=to_restore,
            model=model,
            optimizer=optimizer,
        )
    start_epoch = to_restore['epoch']
    for epoch in range(start_epoch, opt.epochs+1):
        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        time1 = time.time()
        loss,top1,top5= train(epoch, train_loader,model,optimizer,criterion,opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        start = time.time()
        feat_meta_test_acc, feat_meta_test_std = meta_test(model, meta_testloader, is_feat=True)
        test_time = time.time() - start
        print('Feat Meta Test Acc: {:.4f}, Feat Meta Test std: {:.4f}, Time: {:.1f}'.format(feat_meta_test_acc,
                                                                                            feat_meta_test_std,
                                                                                            test_time))
        feat_meta_val_acc, feat_meta_val_std = meta_test(model, meta_valloader, is_feat=True)

        # if epoch % opt.eval_freq ==0 or epoch==opt.epochs:
        #     start = time.time()
        #     feat_meta_test_acc, feat_meta_test_std = meta_test(model_s, meta_testloader, is_feat=True)
        #     test_time = time.time() - start
        #     print('Feat Meta Test Acc: {:.4f}, Feat Meta Test std: {:.4f}, Time: {:.1f}'.format(feat_meta_test_acc,
        #                                                                                         feat_meta_test_std,
        #                                                                                         test_time))
        # regular saving

        if epoch % opt.save_freq == 0 or epoch == opt.epochs:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
        record_data(epoch, top1.cpu().numpy(), top5.cpu().numpy(), feat_meta_test_acc, feat_meta_test_std, feat_meta_val_acc, feat_meta_val_std,
                    save_path=opt.record_file)

    # final report
    print("GENERATING FINAL REPORT")
    generate_final_report(model, opt,wandb)
    print(opt.save_folder)

def train(epoch, train_loader,model,optimizer,criterion,opt):
    """One epoch training"""

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()

    tqdm_gen = tqdm.tqdm(train_loader)
    end = time.time()
    for iter, (images,targets,_,indices) in enumerate(tqdm_gen):
        data_time.update(time.time() - end)

        targets = targets.cuda()
        images = images.cuda()
        # ===================forward computing for current batch ==========================
        _,logits,_,_ = model(images)

        loss = criterion(logits,targets)
        acc1, acc5 = accuracy(logits, targets,topk=(1, 5))
        losses.update(loss.item())
        top1.update(acc1[0] )
        top5.update(acc5[0])

        # ===================backward=====================
        # for optimizer in optimizers:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()
        tqdm_gen.set_description(
            f'[train] epo:{epoch:>3} | avg.loss:{losses.avg():.4f} | acc@1:{top1.avg():.3f} (acc@5:{top5.avg():.3f})')

    return loss,top1.avg(),top5.avg()

if __name__ == '__main__':
    main()