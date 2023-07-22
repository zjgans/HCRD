from __future__ import print_function

import os
import argparse
import socket
import time
import sys
import copy
import mkl
import numpy as np
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import tqdm
from models import model_pool
from models.util import create_model
from distill.criterion import DistillKL, RKDLoss, RKD_LOSS, kl_loss
from distill.dist_loss import DIST, DIST_LOSS, Feat_Loss
from dataset.transform_cfg import transforms_options, transforms_list

from util import adjust_learning_rate, accuracy, AverageMeter,  \
    Meter, generate_final_report, restart_from_checkpoint, set_gpu,  record_data
from eval.meta_eval import meta_test

from modules.loss import kl_loss
from dataloader import get_dataloaders


mkl.set_num_threads(2)

class Wrapper(nn.Module):

    def __init__(self, model, args):
        super(Wrapper, self).__init__()

        self.model = model
        self.feat = torch.nn.Sequential(*list(self.model.children())[:-2])

        self.last = torch.nn.Linear(list(self.model.children())[-2].in_features, 64)

    def forward(self, images):
        feat = self.feat(images)
        feat = feat.view(images.size(0), -1)
        out = self.last(feat)

        return feat, out


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=5, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
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

    # dataset and model
    parser.add_argument('--model_s', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_t', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100', 'cub', 'cross'])
    parser.add_argument('--cross_dataset', type=str, default='cub', choices=['cub', 'cars', 'places', 'plantae'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', type=bool, help='use trainval set')
    parser.add_argument('--use_resume', action='store_true', help='use the result of training before')
    parser.add_argument('--resume_file', type=str, default='ckpt_epoch_30.pth')

    # path to teacher model
    parser.add_argument('--path_t', type=str,
                        default="./save/~/ckpt_epoch_65.pth",
                        help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'contrast', 'hint', 'attention'])
    parser.add_argument('--trial', type=str, default='test', help='trial id')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='save/', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='tb/', help='path to tensorboard')
    parser.add_argument('--record_path', type=str, default='./record', help='record the data of results')
    parser.add_argument('--data_root_path', type=str, default='/~/dataset', help='path to data root')
    parser.add_argument('--cross_domain', type=bool, default=False)

    # setting for meta-learning
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    # memory hyper parameters
    parser.add_argument('--proj_dim', type=float, default=128)
    parser.add_argument('--pretrained_path', type=str, default="", help='student pretrained path')
    parser.add_argument('-gpu', default='1', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('--trans', type=int, default=1)
    parser.add_argument('--tea_trans', type=int, default=1)

    opt = parser.parse_args()
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'
        opt.crop_size = 32

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_distilled'
    if not opt.tb_path:
        opt.tb_path = './tensorboard'
    else:
        opt.data_root = '{}/{}'.format(opt.data_root_path, opt.dataset)
    if opt.cross_domain:
        opt.out_data_root = '{}/{}'.format(opt.data_root_path, opt.cross_dataset)
    if opt.record_path:
        opt.record_path = './record/{}'.format(opt.dataset)
    else:
        opt.record_path = '{}/{}'.format(opt.record_path, opt.dataset)
    opt.data_aug = True

    tags = opt.tags.split(',')
    opt.tags = list([])
    for it in tags:
        opt.tags.append(it)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}'.format(opt.model_s,opt.model_t,opt.dataset,opt.distill,opt.learning_rate)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.model_name = '{}_{}'.format(opt.model_name, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if not os.path.isdir(opt.record_path):
        os.makedirs(opt.record_path)
    opt.record_file = os.path.join(opt.record_path, opt.trial + '.csv')

    opt.num_gpu = set_gpu(opt)
    opt.device_ids = None if opt.gpu == '-1' else list(range(opt.num_gpu))
    opt.n_gpu = torch.cuda.device_count()

    # extras
    opt.fresh_start = True

    return opt


def load_teacher(model_path, model_name, n_cls, dataset='miniImageNet', embd_size=128, trans=12):
    """load the teacher model"""
    print('==> loading teacher model')
    print(model_name)
    model = create_model(model_name, n_cls, dataset, embd_size=embd_size, num_trans=trans)
    # if torch.cuda.device_count() > 1:
    #     print("gpu count:", torch.cuda.device_count())
    #     model = nn.DataParallel(model)
    state_dict = torch.load(model_path)['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print('==> done')
    return model


def main():

    opt = parse_option()
    wandb.init(project=opt.model_path.split("/")[-1], tags='training')
    wandb.config.update(opt)
    wandb.save('*.py')
    wandb.run.save()
    print(opt.save_folder)

    # dataloader
    train_loader,meta_testloader, meta_valloader, n_cls = get_dataloaders(opt)
    opt.num_classes = n_cls

    teacher = load_teacher(opt.path_t, opt.model_t, n_cls, opt.dataset, embd_size=opt.proj_dim, trans=opt.tea_trans)
    student = create_model(opt.model_s, n_cls, opt.dataset, embd_size=opt.proj_dim, num_trans=opt.trans)
    if torch.cuda.device_count() > 1:
        print("second gpu count:", torch.cuda.device_count())
        student = nn.DataParallel(student)
    if opt.pretrained_path != "":
        student.load_state_dict(torch.load(opt.pretrained_path)['model'])

    for p in teacher.parameters():
        p.requires_grad = False
    wandb.watch(student)

    optimizer = optim.SGD(student.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    if torch.cuda.is_available():
        teacher = teacher.cuda()
        student = student.cuda()
        cudnn.benchmark = True

    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    to_restore = {'epoch': 0}
    if opt.use_resume:
        print('------load the parameters from  checkpoint--------')
        restart_from_checkpoint(
            os.path.join(opt.save_folder, opt.resume_file),
            run_variables=to_restore,
            model=student,
            optimizer=optimizer,
        )
    start_epoch = to_restore['epoch']

    # routine: supervised model distillation
    trlog = {}
    trlog['max_acc'] = 0.
    trlog['max_acc_epoch'] = 0
    for epoch in range(start_epoch, opt.epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        loss, top1, top5 = train(epoch, train_loader, student, teacher, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # evaluate
        start = time.time()
        feat_meta_test_acc, feat_meta_test_std = meta_test(student, meta_testloader, is_feat=True)
        test_time = time.time() - start
        print('Feat Meta Test Acc: {:.4f}, Feat Meta Test std: {:.4f}, Time: {:.1f}'.format(feat_meta_test_acc,
                                                                                            feat_meta_test_std,
                                                                                            test_time))
        feat_meta_val_acc, feat_meta_val_std = meta_test(student, meta_valloader, is_feat=True)
        # regular saving
        if feat_meta_test_acc > trlog['max_acc']:
            trlog['max_acc'] = feat_meta_test_acc
            trlog['max_acc_epoch'] = epoch
            outfile = os.path.join(opt.save_folder, 'best_model.pth')
            torch.save({'epoch': epoch, 'model': student.state_dict()}, outfile)
        if epoch % opt.save_freq == 0 or epoch == opt.epochs:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': student.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # wandb saving
            torch.save(state, os.path.join(wandb.run.dir, "model.pth"))
        record_data(epoch, top1.cpu().numpy(), top5.cpu().numpy(), feat_meta_test_acc, feat_meta_test_std,
                    feat_meta_val_acc, feat_meta_val_std,
                    save_path=opt.record_file)

    # final report
    print("GENERATING FINAL REPORT")
    generate_final_report(student, opt, wandb)

    # remove output.txt log file
    output_log_file = os.path.join(wandb.run.dir, "output.log")
    if os.path.isfile(output_log_file):
        os.remove(output_log_file)
    else:  ## Show an error ##
        print("Error: %s file not found" % output_log_file)
    print(opt.save_folder)

def train(epoch, train_loader, model_s, model_t, optimizer, opt):
    """One epoch training"""
    model_s.train()
    model_t.eval()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    tqdm_gen = tqdm.tqdm(train_loader)
    end = time.time()

    for iter, (images, targets, _,_) in enumerate(tqdm_gen):
        data_time.update(time.time() - end)

        targets = targets.cuda()
        images = images.cuda()

        _,logits,_,_ = model_s(images)

        with torch.no_grad():
            _, logits_t, _, _ = model_t(images)

        ce_loss = F.cross_entropy(logits, targets)
        distill_logits_div = kl_loss(logits, logits_t)
        loss = ce_loss + distill_logits_div

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item())
        top1.update(acc1[0])
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

    return loss, top1.avg(), top5.avg()


if __name__ == '__main__':
    main()
