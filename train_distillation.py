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
from distill.criterion import DistillKL,RKDLoss,RKD_LOSS,kl_loss
from distill.dist_loss import DIST,DIST_LOSS,Feat_Loss
from dataset.transform_cfg import transforms_options, transforms_list

from util import adjust_learning_rate, accuracy, AverageMeter, rotrate_concat,\
    Meter, generate_final_report,restart_from_checkpoint,set_gpu,mix_data_lab,to_one_hot,record_data
from eval.meta_eval import meta_test
from eval.cls_eval import validate
from losses import simple_contrstive_loss,mixup_supcontrastive_loss,Align_Loss
from modules.loss import kl_loss,mse_loss,SupCluLoss

from dataloader import get_train_dataloader,get_eval_dataloader
from dataset.utils import Trans_to_Num,rot_color_transformation

# os.environ["CUDA_VISIBLE_DEVICES"]='2'
# os.environ["CUDA_LAUNCH_BLOCKING"]='0'
mkl.set_num_threads(2)

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
    parser.add_argument('--tags', type=str, default="gen1, hcrd", help='add tags for the experiment')
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', type=bool, help='use trainval set')
    parser.add_argument('--use_resume', action='store_true', help='use the result of training before')
    parser.add_argument('--resume_file', type=str, default='ckpt_epoch_30.pth')

    # path to teacher model
    parser.add_argument('--path_t', type=str, default="./save/~/ckpt_epoch_90.pth",
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

    parser.add_argument('--dist_w', type=float, default=1.5,help='loss coefficient for dist loss')
    parser.add_argument('-a', '--alpha', type=float, default=1, help='weight balance for KD')

    #memory hyper parameters
    parser.add_argument('--use_global_feat', type=bool, default=True,
                        help='Whether to add global feature instance for mixup contrastive loss')
    parser.add_argument('--local_t', type=float, default=0.07, help='temperature for local supervision loss')
    parser.add_argument('--global_t',type=float, default=0.07, help='temperature for global supervision loss')
    parser.add_argument('--distill_t', type=float, default=4.)

    parser.add_argument('--mix_t',type=float,default=0.07)
    parser.add_argument('--proj_dim',type=float,default=128)
    parser.add_argument('--trans_type', type=str, default='rot_color_perm12',help='rotation,rotation2,color_perm6,color_perm3,rot_color_perm6,'
                             'rot_color_perm12,rot_color_perm24')
    parser.add_argument('--cross-ratio', default=0.2, type=float, help='four patches crop cross ratio')
    parser.add_argument('--crop_size', type=int, default=84)
    parser.add_argument('--m_patch', type=int, default=2, help='the rule of split for one image,m_patch x m_patch')
    parser.add_argument('--pretrained_path', type=str, default="", help='student pretrained path')
    parser.add_argument('-gpu', default='0' ,help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')

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

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_dist:{}_gT_:{}_Lt:{}_mixT:{}_tag_{}'.format(opt.model_s, opt.model_t, opt.dataset,
                                        opt.distill, opt.gamma, opt.alpha, opt.dist_w,opt.global_t,opt.local_t,opt.mix_t,opt.tags[-1])
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
    opt.record_file = os.path.join(opt.record_path, opt.trial+'.csv')

    opt.num_gpu = set_gpu(opt)
    opt.device_ids = None if opt.gpu == '-1' else list(range(opt.num_gpu))
    opt.n_gpu = torch.cuda.device_count()
    
    #extras
    opt.fresh_start = True

    return opt

def load_teacher(model_path, model_name, n_cls, dataset='miniImageNet',  embd_size=128,trans=12):
    """load the teacher model"""
    print('==> loading teacher model')
    print(model_name)
    model = create_model(model_name, n_cls, dataset,embd_size=embd_size,num_trans=trans)
    # if torch.cuda.device_count() > 1:
    #     print("gpu count:", torch.cuda.device_count())
    #     model = nn.DataParallel(model)
    state_dict = torch.load(model_path)['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print('==> done')
    return model

def main():
    best_acc = 0

    opt = parse_option()
    wandb.init(project=opt.model_path.split("/")[-1], tags=opt.tags)
    wandb.config.update(opt)
    wandb.save('*.py')
    wandb.run.save()
    print(opt.save_folder)

    # dataloader
    train_loader,val_loader,n_cls = get_train_dataloader(opt)
    meta_testloader, meta_valloader,_ = get_eval_dataloader(opt)
    opt.num_classes = n_cls

    teacher=load_teacher(opt.path_t, opt.model_t, n_cls, opt.dataset,  embd_size=opt.proj_dim,trans=opt.trans)
    student = create_model(opt.model_s, n_cls, opt.dataset, embd_size=opt.proj_dim,num_trans=opt.trans)
    if torch.cuda.device_count() > 1:
        print("second gpu count:", torch.cuda.device_count())
        student = nn.DataParallel(student)
    if opt.pretrained_path != "":
        student.load_state_dict(torch.load(opt.pretrained_path)['model'])

    for p in teacher.parameters():
        p.requires_grad = False
    wandb.watch(student)

    criterion_kd = DistillKL(opt.kd_T)
    criterion_supclu_l = SupCluLoss(opt.local_t)
    criterion_supclu_g = SupCluLoss(opt.global_t)
    criterion_rkd = RKDLoss(opt.w_d,opt.w_a)
    criterion_dist = DIST(beta=1, gamma=1)

    optimizer = optim.SGD(student.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    if torch.cuda.is_available():
        teacher = teacher.cuda()
        student = student.cuda()
        criterion_supclu_l = criterion_supclu_l.cuda()
        criterion_supclu_g = criterion_supclu_g.cuda()
        criterion_rkd = criterion_rkd.cuda()
        criterion_dist = criterion_dist.cuda()
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
    trlog['max_acc']=0.
    trlog['max_acc_epoch'] = 0
    for epoch in range(start_epoch, opt.epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        loss,top1,top5 = train(epoch, train_loader,student, teacher,
                                criterion_supclu_l, criterion_supclu_g,criterion_rkd, criterion_dist,optimizer, opt)
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
            outfile = os.path.join(opt.save_folder,'best_model.pth')
            torch.save({'epoch': epoch,'model': student.state_dict()},outfile)
        if epoch % opt.save_freq == 0 or epoch==opt.epochs:

            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': student.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            #wandb saving
            torch.save(state, os.path.join(wandb.run.dir, "model.pth"))
        record_data(epoch, top1.cpu().numpy(), top5.cpu().numpy(), feat_meta_test_acc, feat_meta_test_std,
                    feat_meta_val_acc, feat_meta_val_std,
                    save_path=opt.record_file)

    #final report
    print("GENERATING FINAL REPORT")
    generate_final_report(student, opt,wandb)

    #remove output.txt log file
    output_log_file = os.path.join(wandb.run.dir, "output.log")
    if os.path.isfile(output_log_file):
        os.remove(output_log_file)
    else:    ## Show an error ##
        print("Error: %s file not found" % output_log_file)
    print(opt.save_folder)


def train(epoch, train_loader, model_s, model_t, criterion_supclu_l,criterion_supclu_g,criterion_rkd,criterion_dist,optimizer, opt):
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
        aug_data, _ = rot_color_transformation(geo_images[0], opt.trans_type)
        batch_size = geo_images[0].size(0)

        # data processing
        mix_x1, y_a1, y_b1, index1, lam1 = mix_data_lab(geo_images[0], targets)
        mix_x2, y_a2, y_b2, index2, lam2 = mix_data_lab(geo_images[1], targets)
        mix_data = torch.cat((mix_x1,mix_x2),dim=0)

        n = aug_data.size(0) // batch_size
        multi_labels = torch.stack([targets * n + i for i in range(n)], 1).view(-1)

        mix_images, permute, bs_all, un_shuffle_permute = montage_opera(jig_images)
        # forward compute
        _,logits, m_logits,out_s = model_s(aug_data)
        _,_,_,mix_feat = model_s(mix_data)
        # local forward
        local_clu_feat = model_s(mix_images, use_clu=True)
        local_clu_gather = multi_decouple_feature(local_clu_feat, opt.m_patch)
        order_targets = targets.repeat(opt.m_patch * opt.m_patch)
        mix_targets = order_targets[permute]
        local_feat = model_s.local_head(local_clu_gather)
        local_feat_norm = F.normalize(local_feat, dim=-1)
        local_agg_loss = criterion_supclu_l(local_feat_norm, mix_targets)

        # ===================forward=====================
        with torch.no_grad():
            feat_t, logits_t, m_logits_t, out_t = model_t(aug_data)
            m_feat_t, _, _, mix_feat_t = model_t(mix_data)
            local_clu_feat_t = model_t(mix_images, use_clu=True)
            local_clu_gather_t = multi_decouple_feature(local_clu_feat_t, opt.m_patch)
            local_feat_t = model_t.local_head(local_clu_gather_t)
            local_feat_t_norm = F.normalize(local_feat_t, dim=-1)

        # compute loss
        logits = logits[::n]
        logits_t = logits_t[::n]
        joint_loss = F.cross_entropy(m_logits, multi_labels)
        single_loss = F.cross_entropy(logits, targets)
        agg_preds = 0

        for i in range(1,n):
            agg_preds = agg_preds + m_logits[i::n, i::n] / n

        distillation_loss = F.kl_div(F.log_softmax(logits / opt.distill_t, 1),
                                     F.softmax(agg_preds.detach() / opt.distill_t, 1),
                                     reduction='batchmean')
        sup_loss = single_loss + joint_loss + distillation_loss.mul(opt.distill_t ** 2)

        global_labels = torch.stack([targets for i in range(n)], 1).view(-1)
        global_agg_loss = criterion_supclu_g(out_s, global_labels)

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

        # compute distillation loss
        loss_rd = DIST_LOSS(mix_feat, mix_feat_t) + DIST_LOSS(local_feat_norm, local_feat_t_norm) + DIST_LOSS(out_s,out_t)
        loss_nld = kl_loss(logits, logits_t) + kl_loss(m_logits, m_logits_t)
        distill_loss = (loss_nld + loss_rd * opt.dist_w) * opt.alpha

        loss = sup_loss + global_agg_loss + local_agg_loss + mix_loss + distill_loss

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

    return loss,top1.avg(),top5.avg()

def montage_opera(images):
    n, c, h, w = images[0].shape
    permute = torch.randperm(n * 4).cuda()

    un_shuffle_permute = torch.argsort(permute)
    images_gather = torch.cat(images, dim=0)
    images_gather = images_gather[permute, :, :, :]

    col1 = torch.cat([images_gather[0:n], images_gather[n:2 * n]], dim=3)
    col2 = torch.cat([images_gather[2 * n:3 * n], images_gather[3 * n:]], dim=3)
    images_gather = torch.cat([col1, col2], dim=2).cuda()
    return images_gather, permute, n,un_shuffle_permute

def multi_decouple_feature(feature,m):
        num_patch = int(m*m)
        n, c, h, w = feature.shape
        c1, c2 = feature.split([1, 1], dim=2)
        f1, f2 = c1.split([1, 1], dim=3)
        f3, f4 = c2.split([1, 1], dim=3)
        f_gather = torch.cat([f1, f2, f3, f4], dim=0)
        mix_gather = f_gather.view(n * num_patch, -1)
        return mix_gather

    
if __name__ == '__main__':
    main()
