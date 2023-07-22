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
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,ConcatDataset
import tqdm
from PIL import Image
from dataset.transform_cfg import transforms_options, transforms_list
from dataset.utils import rot_color_perm12
from models import model_pool
from models.util import create_model
from util import adjust_learning_rate, accuracy, AverageMeter,Meter, generate_final_report, \
    restart_from_checkpoint,init_distributed_mode,has_batchnorms,is_main_process,save_on_master
from eval.meta_eval import meta_test
from dataset.utils import Trans_to_Num,rot_color_transformation
from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100

from modules.sla_hcrd import  HCRD
from dataloader import Preprocessor,JigCluTransform,get_eval_dataloader

os.environ["CUDA_VISIBLE_DEVICES"] ='1,2'
os.environ["CUDA_LAUNCH_BLOCKING"]='1'
mkl.set_num_threads(2)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--eval_freq', type=int, default=5, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=5, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--rate',type=float,default=1)
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,50', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--tags', type=str, default="gen0", help='add tags for the experiment')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='CIFAR-FS',
                        choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100', 'cub', 'cross'])
    parser.add_argument('--cross_dataset', type=str, default='cub', choices=['cub', 'cars', 'places', 'plantae'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', type=bool, help='use trainval set')
    parser.add_argument('--use_resume', action='store_true', help='use the result of training before')
    parser.add_argument('--resume_file', type=str, default='ckpt_epoch_10.pth')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    # specify folder
    parser.add_argument('--model_path', type=str, default='./save', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='./tb', help='path to tensorboard')
    parser.add_argument('--record_path', type=str, default='./record', help='record the data of results')
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
    parser.add_argument('-t', '--trial', type=str, default='new_test_distill_2gpu', help='the experiment id')

    # hyper parameters
    parser.add_argument('--alpha', type=float, default=0.5, help='loss coefficient for knowledge distillation loss')

    parser.add_argument('--proj_dim',type=float,default=128)
    parser.add_argument('--cross-ratio', default=0.2, type=float,help='four patches crop cross ratio')
    parser.add_argument('--crop_size', type=int, default=84)
    parser.add_argument('--mix_t', type=float, default=0.07)
    parser.add_argument('--sup_t', type=float,default=0.07, help='temperature for local supervision loss')
    parser.add_argument('--distill_t', type=float, default=4.)
    parser.add_argument('--contrast_t',type=float,default=0.07)
    parser.add_argument('--mvavg_rate', type=float, default=0.999, help='temperature for contrastive ssl loss')
    parser.add_argument('--trans_type', type=str, default='rot_color_perm12',help='rotation,rotation2,color_perm6,color_perm3,rot_color_perm6,'
                             'rot_color_perm12,rot_color_perm24')
    parser.add_argument('--m_patch', type=int,default=2,help='the rule of split for one image,m_patch x m_patch')
    parser.add_argument('--use_global_feat', type=bool, default=True, help='Whether to add global feature instance for mixup contrastive loss')

    parser.add_argument('-gpu', default='1,2', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for ei ther single node or '
                             'multi node data parallel training')

    opt = parser.parse_args()
    opt.trans = Trans_to_Num[opt.trans_type]
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'
        opt.crop_size = 32

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
        opt.record_path = '{}/{}'.format(opt.record_path, opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    tags = opt.tags.split(',')
    opt.tags = list([])
    for it in tags:
        opt.tags.append(it)

    opt.model_name = '{}_{}_lr_{}_decay_{}_a_{}_supT_{}_mixT_{}_augtype_{}'.format(opt.model, opt.dataset,opt.learning_rate,opt.weight_decay,
                                                            opt.alpha, opt.sup_t,opt.mix_t, opt.trans_type)
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

    # opt.num_gpu = set_gpu(opt)
    # opt.device_ids = None if opt.gpu == '-1' else list(range(opt.num_gpu))
    # opt.n_gpu = torch.cuda.device_count()

    # extras
    opt.fresh_start = True
    return opt

def get_train_dataloader(opt):
    train_partition = 'trainval' if opt.use_trainval else 'train'
    train_trans, test_trans = transforms_options[opt.transform]
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        dataset = CIFAR100(args=opt, partition=train_partition, transform=train_trans)
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize = transforms.Normalize(mean=mean, std=std)
        trans_crop = transforms.Compose([

            # transforms.Resize((32,32)),
            # transforms.RandomCrop(32,padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_trans = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_trans_aug = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))

        dataset = Preprocessor(dataset.img_label, [train_trans, train_trans_aug], JigCluTransform(opt, trans_crop))

    elif opt.dataset == 'tieredImageNet':
        dataset = TieredImageNet(args=opt, partition=train_partition, transform=train_trans)
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        trans = transforms.Compose([
            transforms.RandomCrop(42, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])
        train_trans = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_trans_aug = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351

        dataset = Preprocessor(dataset.img_label,[train_trans,train_trans_aug], JigCluTransform(opt, trans))

    elif opt.dataset == 'miniImageNet' and not opt.cross_domain:
        dataset = ImageNet(args=opt, partition=train_partition, transform=train_trans)
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        trans = transforms.Compose([
            transforms.RandomCrop(42, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])
        train_trans = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_trans_aug = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
        dataset = Preprocessor(dataset.img_label,[train_trans,train_trans_aug], JigCluTransform(opt, trans))

    elif opt.cross_domain and opt.dataset =='miniImageNet':
        assert opt.transform == "A"
        train_dataset = ImageNet(args=opt, partition='train', transform=train_trans)
        val_dataset = ImageNet(args=opt, partition='val', transform=train_trans)
        test_dataset = ImageNet(args=opt, partition='test', transform=train_trans)

        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)

        trans = transforms.Compose([
            transforms.RandomCrop(42, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])
        train_trans = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_trans_aug = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = Preprocessor(train_dataset.img_label, [train_trans,train_trans_aug], JigCluTransform(opt, trans))
        val_dataset = Preprocessor(val_dataset.img_label, [train_trans,train_trans_aug], JigCluTransform(opt, trans))
        test_dataset = Preprocessor(test_dataset.img_label, [train_trans,train_trans_aug], JigCluTransform(opt, trans))
        dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

        n_cls = 64 + 16 + 20

    else:
        raise NotImplementedError(opt.dataset)

    return dataset,n_cls

def main():
    opt = parse_option()
    # wandb.init(project=opt.model_path.split("/")[-1], tags=opt.tags)
    # wandb.config.update(opt)
    # wandb.save('*.py')
    # wandb.run.save()
    # print(opt.save_folder)

    init_distributed_mode(opt)
    ngpus_per_node = torch.cuda.device_count()
    opt.batch_size = int(opt.batch_size / ngpus_per_node)
    # fix_random_seeds(opt)

    dataset,n_cls = get_train_dataloader(opt)
    meta_testloader, meta_valoader, _ = get_eval_dataloader(opt)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size = opt.batch_size,
        num_workers= opt.num_workers,
        pin_memory=True,
        drop_last =True
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # model
    model_s = create_model(opt.model, n_cls, opt.dataset,embd_size=opt.proj_dim,num_trans=opt.trans)
    model_t = create_model(opt.model, n_cls, opt.dataset,embd_size=opt.proj_dim,num_trans=opt.trans)
    # wandb.watch(model_s)

    model_s,model_t = model_s.cuda(),model_t.cuda()
    if has_batchnorms(model_s):
        model_s = nn.SyncBatchNorm.convert_sync_batchnorm(model_s)
        model_t = nn.SyncBatchNorm.convert_sync_batchnorm(model_t)

        model_t = nn.parallel.DistributedDataParallel(model_t,device_ids=[opt.gpu])
        model_t_without_ddp = model_t.module
    else:
        model_t_without_ddp = model_t

    model_s = nn.parallel.DistributedDataParallel(model_s,device_ids=[opt.gpu])
    model_t_without_ddp.load_state_dict(model_s.module.state_dict())

    for p in model_t.parameters():
        p.requires_grad = False

    # optimizer
    if opt.adam:
        print("Adam")
        optimizer = torch.optim.Adam(model_s.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        print("SGD")
        optimizer = optim.SGD(model_s.parameters(),
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
            model_s=model_s,
            model_t = model_t,
            optimizer=optimizer,
        )
    start_epoch = to_restore['epoch']
    multibranch_model = HCRD(opt,n_cls,model_s,model_t).cuda()

    for epoch in range(start_epoch, opt.epochs+1):
        train_loader.sampler.set_epoch(epoch)
        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        time1 = time.time()
        loss= train(epoch, train_loader, multibranch_model,optimizer,opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        if dist.get_rank() ==0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': model_s.state_dict(),
                'model_t': model_t.state_dict(),
            }
            if epoch % opt.eval_freq ==0 or epoch==opt.epochs:

                start = time.time()
                feat_meta_test_acc, feat_meta_test_std = meta_test(model_s, meta_testloader, is_feat=True)
                test_time = time.time() - start
                print('Feat Meta Test Acc: {:.4f}, Feat Meta Test std: {:.4f}, Time: {:.1f}'.format(feat_meta_test_acc,
                                                                                                    feat_meta_test_std,
                                                                                                    test_time))
            # regular saving
            if epoch % opt.save_freq == 0 or epoch == opt.epochs:
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_on_master(state,save_file)
            each_save_file = os.path.join(opt.save_folder, 'last.pth')
            torch.save(state, each_save_file)

    # final report
    if dist.get_rank()==0:
        print("GENERATING FINAL REPORT")
        generate_final_report(model_s, opt)
        print(opt.save_folder)

def train(epoch, train_loader, model,optimizer, opt):
    """One epoch training"""

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()

    tqdm_gen = tqdm.tqdm(train_loader)
    end = time.time()
    for iter, (jig_images, geo_images, targets, indices) in enumerate(tqdm_gen):
        data_time.update(time.time() - end)

        targets = targets.cuda()
        geo_images = [data.cuda() for data in geo_images]
        aug_data,_ = rot_color_transformation(geo_images[0],opt.trans_type)
        loss, logits = model(aug_data, geo_images, jig_images, targets)

        # ===================forward computing for current batch ==========================
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
        loss = reduce_value(loss,True)
        batch_time.update(time.time() - end)
        end = time.time()

        tqdm_gen.set_description(
            f'[train] epo:{epoch:>3} | avg.loss:{losses.avg():.4f} | acc@1:{top1.avg():.3f} (acc@5:{top5.avg():.3f})')

    return loss


def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value

if __name__ == '__main__':
    main()