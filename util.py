from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import sys
import csv
import codecs
import random
import torch.nn.functional as F
import torch.distributed as dist
from dataloader import get_eval_dataloader
from ptflops import get_model_complexity_info
from thop import profile
import pandas as pd

def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, num_classes=64):
        super(BCEWithLogitsLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, 
                                              size_average=size_average, 
                                              reduce=reduce, 
                                              reduction=reduction,
                                              pos_weight=pos_weight)
    def forward(self, input, target):
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target_onehot)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Meter:

    def __init__(self):
        self.list = []

    def update(self, item):
        self.list.append(item)

    def avg(self):
        return torch.tensor(self.list).mean() if len(self.list) else None

    def confidence_interval(self):
        if len(self.list) == 0:
            return None
        std = torch.tensor(self.list).std()
        ci = std * 1.96 / math.sqrt(len(self.list))
        return ci

    def avg_and_confidence_interval(self):
        return self.avg(), self.confidence_interval()


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_model_complexity(model,input):
    flops,params = get_model_complexity_info(model,input,as_strings=True, print_per_layer_stat=True, verbose=True)
    # flops, params = profile(model,(input,))
    print('Flops:  ', flops)
    print('Params: ', params)

def to_one_hot(inp, num_classes, device='cuda'):
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)

    return y_onehot

def distance(z, dist_type='l2'):
    '''Return distance matrix between vectors'''
    with torch.no_grad():
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        if dist_type[:2] == 'l2':
            A_dist = (diff**2).sum(-1)
            if dist_type == 'l2':
                A_dist = torch.sqrt(A_dist)
            elif dist_type == 'l22':
                pass
        elif dist_type == 'l1':
            A_dist = diff.abs().sum(-1)
        elif dist_type == 'linf':
            A_dist = diff.abs().max(-1)[0]
        else:
            return None
    return A_dist

def mix_data_lab(x, y, alpha=1, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    lam = max(lam, 1 - lam)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, index, lam


def rotrate_concat(inputs):
    out = None
    for x in inputs:
        x_90 = x.transpose(2,3).flip(2)
        x_180 = x.flip(2).flip(3)
        x_270 = x.flip(2).transpose(2,3)
        if out is None:
            out = torch.cat((x, x_90, x_180, x_270),0)
        else:
            out = torch.cat((out, x, x_90, x_180, x_270),0)
    return out

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    # launched with torch.distributed.launch

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '172.25.56.150'
        os.environ['MASTER_PORT'] = '40140'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)
        

    def close(self):
        if self.file is not None:
            self.file.close()
def restart_from_checkpoint(ckpt_path,run_variables=None,**kwargs):
    if not os.path.isfile(ckpt_path):
        return
    print('Found checkpoint at {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    for key,value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key],strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            print('=>loaded {} from checkpoint {}'.format(key,ckpt_path))
        else:
            print(
                "=> failed to load {} from checkpoint '{}'".format(key, ckpt_path)
            )
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
                print('{}:{})'.format(var_name,run_variables[var_name]))

def data_write_csv(file_name,datas):
    file_csv = codecs.open(file_name, 'w+','utf-8')
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ',quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print('save file successfully,finally')

def record_data(epoch,top1,top5,test_acc,test_ic,val_acc,val_ic,save_path):
    if epoch==0:
        df = pd.DataFrame(columns=['epoch','top1','top5','test_acc','test_ic','val_acc','val_ic'])
        df.to_csv(save_path,index=False)
    list = [epoch,top1,top5,test_acc,test_ic,val_acc,val_ic]
    data = pd.DataFrame([list])
    data.to_csv(save_path,mode='a',header=False,index=False)

# report the final result in few-shot setting
def generate_final_report(model, opt,wandb):
    from eval.meta_eval import meta_test
    
    opt.n_shots = 1
    meta_testloader, meta_valloader, n_cls= get_eval_dataloader(opt)
    
    #validate
    meta_val_acc, meta_val_std = meta_test(model, meta_valloader)
    
    meta_val_acc_feat, meta_val_std_feat = meta_test(model, meta_valloader, is_feat=True)

    #evaluate
    meta_test_acc, meta_test_std = meta_test(model, meta_testloader)
    
    meta_test_acc_feat, meta_test_std_feat = meta_test(model, meta_testloader, is_feat=True)
        
    print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}'.format(meta_val_acc, meta_val_std))
    print('Meta Val Acc (feat): {:.4f}, Meta Val std (feat): {:.4f}'.format(meta_val_acc_feat, meta_val_std_feat))
    print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}'.format(meta_test_acc, meta_test_std))
    print('Meta Test Acc (feat): {:.4f}, Meta Test std (feat): {:.4f}'.format(meta_test_acc_feat, meta_test_std_feat))
    
    
    wandb.log({'Final Meta Test Acc @1': meta_test_acc,
               'Final Meta Test std @1': meta_test_std,
               'Final Meta Test Acc  (feat) @1': meta_test_acc_feat,
               'Final Meta Test std  (feat) @1': meta_test_std_feat,
               'Final Meta Val Acc @1': meta_val_acc,
               'Final Meta Val std @1': meta_val_std,
               'Final Meta Val Acc   (feat) @1': meta_val_acc_feat,
               'Final Meta Val std   (feat) @1': meta_val_std_feat
              })

    
    opt.n_shots = 5
    meta_testloader, meta_valloader, n_cls= get_eval_dataloader(opt)
    
    #validate
    meta_val_acc, meta_val_std = meta_test(model, meta_valloader)
    
    meta_val_acc_feat, meta_val_std_feat = meta_test(model, meta_valloader, is_feat=True)

    #evaluate
    meta_test_acc, meta_test_std = meta_test(model, meta_testloader)
    
    meta_test_acc_feat, meta_test_std_feat = meta_test(model, meta_testloader, is_feat=True)
        
    print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}'.format(meta_val_acc, meta_val_std))
    print('Meta Val Acc (feat): {:.4f}, Meta Val std (feat): {:.4f}'.format(meta_val_acc_feat, meta_val_std_feat))
    print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}'.format(meta_test_acc, meta_test_std))
    print('Meta Test Acc (feat): {:.4f}, Meta Test std (feat): {:.4f}'.format(meta_test_acc_feat, meta_test_std_feat))

    wandb.log({'Final Meta Test Acc @5': meta_test_acc,
               'Final Meta Test std @5': meta_test_std,
               'Final Meta Test Acc  (feat) @5': meta_test_acc_feat,
               'Final Meta Test std  (feat) @5': meta_test_std_feat,
               'Final Meta Val Acc @5': meta_val_acc,
               'Final Meta Val std @5': meta_val_std,
               'Final Meta Val Acc   (feat) @5': meta_val_acc_feat,
               'Final Meta Val std   (feat) @5': meta_val_std_feat
              })
