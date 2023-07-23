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
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,ConcatDataset
import tqdm
from PIL import Image
from models import model_pool
from models.util import create_model
from distill.criterion import DistillKL, RKDLoss,RKD_LOSS
from dataset.transform_cfg import transforms_options, transforms_list
from distill.dist_loss import DIST,DIST_LOSS

from util import adjust_learning_rate, accuracy, AverageMeter, rotrate_concat, \
    Meter, generate_final_report, restart_from_checkpoint, set_gpu, mix_data_lab,has_batchnorms,init_distributed_mode,save_on_master,is_main_process,get_params_groups
from eval.meta_eval import meta_test
from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from losses import mixup_supcontrastive_loss
from modules.loss import kl_loss, SupCluLoss

from dataloader import get_eval_dataloader,JigCluTransform,Preprocessor
from dataset.utils import Trans_to_Num,rot_color_transformation

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
os.environ["CUDA_LAUNCH_BLOCKING"]='0'
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
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,50', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset and model
    parser.add_argument('--model_s', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_t', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100', 'cub', 'cross'])
    parser.add_argument('--cross_dataset', type=str, default='cub', choices=['cub', 'cars', 'places', 'plantae'])

    parser.add_argument('--tags', type=str, default="gen1, ssl", help='add tags for the experiment')
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', type=bool, help='use trainval set')
    parser.add_argument('--use_resume', action='store_true', help='use the result of training before')
    parser.add_argument('--resume_file', type=str, default='last.pth')

    # path to teacher model
    parser.add_argument('--path_t', type=str,
                        default="./save/~/ckpt_epoch_60.pth",
                        help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'contrast', 'hint', 'attention'])
    parser.add_argument('--trial', type=str, default='test', help='trial id')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

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

    parser.add_argument('--gamma', type=float, default=1, help='loss coefficient for local loss')
    parser.add_argument('--dist_w', type=float, default=1.5, help='loss coefficient for dist loss')
    parser.add_argument('-a', '--alpha', type=float, default=1, help='weight balance for KD')

    # memory hyper parameters
    parser.add_argument('--local_t', type=float, default=0.2, help='temperature for local supervision loss')
    parser.add_argument('--global_t', type=float, default=0.2, help='temperature for global supervision loss')
    parser.add_argument('--distill_t', type=float, default=4.)
    parser.add_argument('--mix_t', type=float, default=0.2)
    parser.add_argument('--proj_dim', type=float, default=128)
    parser.add_argument('--trans_type', type=str, default='rot_color_perm12', help='rotation,rotation2,color_perm6,color_perm3,rot_color_perm6,'
                             'rot_color_perm12,rot_color_perm24')
    parser.add_argument('--cross-ratio', default=0.2, type=float, help='four patches crop cross ratio')
    parser.add_argument('--crop_size', type=int, default=84)
    parser.add_argument('--m_patch', type=int, default=2, help='the rule of split for one image,m_patch x m_patch')
    parser.add_argument('--pretrained_path', type=str, default="", help='student pretrained path')
    parser.add_argument('-gpu', default='0,1,2', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
                distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--multiprocessing-distributed', action='store_true',help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the ''fastest way to use PyTorch for ei ther single node or '
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
        opt.model_path = './models_distilled'
    if not opt.tb_path:
        opt.tb_path = './tensorboard'
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    tags = opt.tags.split(',')
    opt.tags = list([])
    for it in tags:
        opt.tags.append(it)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_dist:{}_rkd:{}_gT_:{}_Lt:{}_mixT:{}_tag_{}'.format(opt.model_s,opt.model_t,opt.dataset,
                                                                                                   opt.distill,opt.gamma, opt.alpha,opt.dist_w,
                                                                                                   opt.rkd_w,opt.global_t, opt.local_t,opt.mix_t,
                                                                                                   opt.tags[-1])
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.model_name = '{}_{}'.format(opt.model_name, opt.trial)

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

def load_teacher(model_path, model_name, n_cls, dataset='miniImageNet', embd_size=128):
    """load the teacher model"""
    print('==> loading teacher model')
    print(model_name)
    model = create_model(model_name, n_cls, dataset, embd_size=embd_size)
    state_dict = torch.load(model_path)['model_s']
    state_dict = {k.replace("module.",""): v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)
    print('==> done')
    return model

def main():
    best_acc = 0

    opt = parse_option()
    # wandb.init(project=opt.model_path.split("/")[-1], tags=opt.tags)
    # wandb.config.update(opt)
    # wandb.save('*.py')
    # wandb.run.save()
    # print(opt.save_folder)
    init_distributed_mode(opt)
    ngpus_per_node = torch.cuda.device_count()
    opt.batch_size = int(opt.batch_size / ngpus_per_node)

    # dataloader
    dataset, n_cls = get_train_dataloader(opt)
    meta_testloader, meta_valoader, _ = get_eval_dataloader(opt)
    opt.num_classes = n_cls

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    teacher = load_teacher(opt.path_t, opt.model_t, n_cls, opt.dataset, embd_size=opt.proj_dim)
    student = create_model(opt.model_s, n_cls, opt.dataset, embd_size=opt.proj_dim)
    student, teacher = student.cuda(), teacher.cuda()

    if has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[opt.gpu])

    student = nn.parallel.DistributedDataParallel(student, device_ids=[opt.gpu], output_device=opt.gpu)
    for p in teacher.parameters():
        p.requires_grad = False

    criterion_kd = DistillKL(opt.kd_T).cuda()
    criterion_supclu_l = SupCluLoss(opt.local_t)
    criterion_supclu_g = SupCluLoss(opt.global_t)
    criterion_rkd = RKDLoss(opt.w_d, opt.w_a).cuda()
    criterion_dist = DIST().cuda()

    # params_list = nn.ModuleList([])
    # params_list.append(student)
    params_groups = get_params_groups(student)
    optimizer = optim.SGD(params_groups,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
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
    for epoch in range(start_epoch, opt.epochs + 1):
        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        loss = train(epoch, train_loader, student, teacher,
                     criterion_supclu_l,criterion_supclu_g, criterion_kd, criterion_rkd,criterion_dist, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        #  evaluation and saving
        if dist.get_rank()==0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': student.state_dict(),
            }
            if epoch % opt.save_freq == 0 or epoch == opt.epochs:
                start = time.time()
                feat_meta_test_acc, feat_meta_test_std = meta_test(student, meta_testloader, is_feat=True)
                test_time = time.time() - start
                print('Feat Meta Test Acc: {:.4f}, Feat Meta Test std: {:.4f}, Time: {:.1f}'.format(feat_meta_test_acc,
                                                                                                    feat_meta_test_std,
                                                                                                    test_time))
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(state, save_file)

            each_save_file = os.path.join(opt.save_folder,'last.pth')
            torch.save(state,each_save_file)

                # wandb saving
                # torch.save(state, os.path.join(wandb.run.dir, "model.pth"))

    # final report
    if dist.get_rank()==0:
        print("GENERATING FINAL REPORT")
        generate_final_report(student, opt)
        print(opt.save_folder)
    # remove output.txt log file
    # output_log_file = os.path.join(wandb.run.dir, "output.log")
    # if os.path.isfile(output_log_file):
    #     os.remove(output_log_file)
    # else:  ## Show an error ##
    #     print("Error: %s file not found" % output_log_file)

def train(epoch, train_loader, model_s, model_t, criterion_supclu_l,criterion_supclu_g,criterion_kd, criterion_rkd, criterion_dist,optimizer, opt):
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

    for iter, (jig_images, geo_images, targets, indices) in enumerate(tqdm_gen):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            targets = targets.cuda()
            geo_images = [data.cuda() for data in geo_images]
        rot_color_data = rot_color_transformation(geo_images[0],opt.trans_type)
        batch_size = geo_images[0].size(0)

        # data processing
        mix_x1, y_a1, y_b1, index1, lam1 = mix_data_lab(geo_images[0], targets)
        mix_x2, y_a2, y_b2, index2, lam2 = mix_data_lab(geo_images[1], targets)
        mix_data = torch.cat((mix_x1, mix_x2), dim=0)

        n = rot_color_data.size(0) // batch_size
        multi_labels = torch.stack([targets * n + i for i in range(n)], 1).view(-1)

        mix_images, permute, bs_all, un_shuffle_permute = montage_opera(jig_images)
        # forward compute

        _,logits, m_logits, out_s = model_s(rot_color_data)
        _,_, _, mix_feat = model_s(mix_data)

        mix_clu_feat = model_s(mix_images, use_clu=True)
        mix_clu_gather = multi_decouple_feature(mix_clu_feat, opt.m_patch)
        order_targets = targets.repeat(opt.m_patch * opt.m_patch)
        mix_targets = order_targets[permute]
        local_feat = model_s.module.local_head(mix_clu_gather)
        local_feat_norm = F.normalize(local_feat, dim=-1)
        local_agg_loss = criterion_supclu_l(local_feat_norm, mix_targets)

        # ===================forward=====================
        with torch.no_grad():
            _,logits_t, m_logits_t, out_t = model_t(rot_color_data)
            _,_, _, mix_feat_t = model_t(mix_data)
            mix_clu_feat_t = model_t(mix_images, use_clu=True)
            mix_clu_gather_t = multi_decouple_feature(mix_clu_feat_t, opt.m_patch)
            local_feat_t = model_t.module.local_head(mix_clu_gather_t)
            local_feat_t_norm = F.normalize(local_feat_t, dim=-1)

        # compute loss
        logits = logits[::n]
        logits_t = logits_t[::n]
        joint_loss = F.cross_entropy(m_logits, multi_labels)
        single_loss = F.cross_entropy(logits, targets)
        agg_preds = 0
        for i in range(n):
            agg_preds = agg_preds + m_logits[i::n, i::n] / n
        #
        distillation_loss = F.kl_div(F.log_softmax(logits / opt.distill_t, 1),
                                     F.softmax(agg_preds.detach() / opt.distill_t, 1),
                                     reduction='batchmean')

        if opt.use_global_feat:
            if n == 12 or n == 24:
                s = int(n // 4)
                out_s_sup = out_s.view(batch_size, n, out_s.size(1)).permute(1, 0, 2)  # n batch dim
                out_t_sup = out_t.view(batch_size,n,out_t.size(1)).permute(1,0,2)

                out_s_subfeat = out_s_sup[0::s].reshape(-1, out_s.size(1))
                out_t_subfeat = out_t_sup[0::s].reshape(-1,out_t.size(1))
                global_num = 4
            else:
                out_s_sup = out_s.view(batch_size, n, out_s.size(1)).permute(1, 0, 2)
                out_t_sup = out_t.view(batch_size, n, out_t.size(1)).permute(1, 0, 2)
                out_s_subfeat = out_s_sup.reshape(-1, out_s.size(1))
                out_t_subfeat = out_t_sup.reshape(-1,out_t.size(1))
                global_num = n
            mix_loss= mixup_supcontrastive_loss(mix_feat, targets,opt.mix_t, batch_size, index1, index2, lam1, lam2,out_s_subfeat,global_num)
        else:
            mix_loss= mixup_supcontrastive_loss(mix_feat, targets,opt.mix_t, batch_size, index1, index2, lam1, lam2)

        global_labels = torch.stack([targets for i in range(n)], 1).view(-1)
        global_agg_loss = criterion_supclu_g(out_s, global_labels)

        sup_loss = single_loss + joint_loss + distillation_loss.mul(opt.distill_t ** 2)

        # compute distillation loss
        loss_rd = DIST_LOSS(mix_feat, mix_feat_t) + DIST_LOSS(local_feat_norm, local_feat_t_norm) + DIST_LOSS(out_s,out_t)
        loss_nld = kl_loss(logits, logits_t) + kl_loss(m_logits, m_logits_t)
        distill_loss = (loss_nld + loss_rd * opt.dist_w) * opt.alpha

        loss = sup_loss + global_agg_loss +local_agg_loss + mix_loss + distill_loss

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

    return loss


def montage_opera(images):
    n, c, h, w = images[0].shape
    permute = torch.randperm(n * 4).cuda()

    un_shuffle_permute = torch.argsort(permute)
    images_gather = torch.cat(images, dim=0)
    images_gather = images_gather[permute, :, :, :]

    col1 = torch.cat([images_gather[0:n], images_gather[n:2 * n]], dim=3)
    col2 = torch.cat([images_gather[2 * n:3 * n], images_gather[3 * n:]], dim=3)
    images_gather = torch.cat([col1, col2], dim=2).cuda()
    return images_gather, permute, n, un_shuffle_permute


def multi_decouple_feature(feature, m):
    num_patch = int(m * m)
    n, c, h, w = feature.shape
    c1, c2 = feature.split([1, 1], dim=2)
    f1, f2 = c1.split([1, 1], dim=3)
    f3, f4 = c2.split([1, 1], dim=3)
    f_gather = torch.cat([f1, f2, f3, f4], dim=0)
    mix_gather = f_gather.view(n * num_patch, -1)
    return mix_gather


if __name__ == '__main__':
    main()
