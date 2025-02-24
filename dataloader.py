from __future__ import print_function

import numpy as np
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,ConcatDataset
from PIL import Image
from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.cub import CUB,MetaCUB
from dataset.transform_cfg import transforms_options, transforms_test_options
from dataset.utils import RandAugmentMC, GaussianBlur
# dataloader for all dataset
def get_dataloaders(opt):
    # dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'

    if opt.dataset == 'miniImageNet':

        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_dataset = ImageNet(args=opt, partition='val', transform=test_trans)
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64

    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]

        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_dataset = TieredImageNet(args=opt, partition='val', transform=test_trans)
        val_loader = DataLoader(TieredImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]

        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351

    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options[opt.transform]

        train_loader = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_dataset = CIFAR100(args=opt, partition='val', transform=test_trans)

        val_loader = DataLoader(CIFAR100(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        train_trans, test_trans = transforms_test_options[opt.transform]


        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))

    elif opt.cross_domain and opt.dataset =='miniImageNet':
        assert opt.transform == "A"
        train_trans, test_trans = transforms_options[opt.transform]
        train_dataset = ImageNet(args=opt, partition='train', transform=train_trans)
        val_dataset = ImageNet(args=opt, partition='val', transform=train_trans)
        test_dataset = ImageNet(args=opt, partition='test', transform=train_trans)
        all_datasets = ConcatDataset([train_dataset, val_dataset, test_dataset])

        opt.transform = 'C'
        train_trans, test_trans = transforms_options[opt.transform]
        meta_testloader = DataLoader(MetaCUB(args=opt, partition='novel',
                                             train_transform=train_trans, test_transform=test_trans, fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCUB(args=opt, partition='val', train_transform=train_trans,
                                            test_transform=test_trans, fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        # n_cls = train_classes[opt.dataset]

        train_loader = DataLoader(all_datasets,
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers,
                                  )

        n_cls = 64 + 16 + 20  # train + val + test

    else:
        raise NotImplementedError(opt.dataset)

    return train_loader,meta_testloader, meta_valloader, n_cls



class JigCluTransform:
    def __init__(self, opt,transform):
        self.transform = transform
        self.c = opt.cross_ratio
        self.opt = opt
        self.m = opt.m_patch
        self.grid_ratio_default = 2.0

    def get_grid_location(self, size, ratio, num_grid):
        '''
        :param size: size of the height/width
        :param ratio: generate grid size/ even divided grid size
        :param num_grid: number of grid
        :return: a list containing the coordinate of the grid
        '''
        raw_grid_size = int(size / num_grid)
        enlarged_grid_size = int(size / num_grid * ratio)
        # enlarged_grid_size = int(size*0.2)
        center_location = raw_grid_size // 2

        location_list = []
        for i in range(num_grid):
            location_list.append((max(0, center_location - enlarged_grid_size // 2),
                                  min(size, center_location + enlarged_grid_size // 2)))
            center_location = center_location + raw_grid_size
        return location_list

    def get_pyramid(self, img, num_grid):

        grid_ratio = 1 + 3 * random.random()

        w, h = img.size
        grid_locations_w = self.get_grid_location(w, grid_ratio, num_grid)
        grid_locations_h = self.get_grid_location(h, grid_ratio, num_grid)

        patches_list=[]
        for i in range(num_grid):
            for j in range(num_grid):
                patch_location_w=grid_locations_w[j]
                patch_location_h=grid_locations_h[i]
                left_up_corner_w=patch_location_w[0]
                left_up_corner_h=patch_location_h[0]
                right_down_cornet_w=patch_location_w[1]
                right_down_cornet_h = patch_location_h[1]
                patch=img.crop((left_up_corner_w,left_up_corner_h,right_down_cornet_w,right_down_cornet_h))
                patch=self.transform_cand(patch)
                patches_list.append(patch)
        return patches_list

    def __call__(self, x,m):

        if self.opt.dataset == 'cub':

            x = Image.open(x).convert('RGB')
        else:
            x = Image.fromarray(x)
        h, w = x.size

        ch = self.c * h
        cw = self.c * w

        # x = transforms.RandomCrop(32,padding=4)(x)
        if self.opt.dataset =='FC100':
            x = transforms.RandomCrop(32,padding=4)(x)

            # return [self.transform(x.crop((0, 0, h // 2 + ch, w // 2 + cw))),
            #         self.transform(x.crop((0, w // 2 - cw, h // 2 + ch, w))),
            #         self.transform(x.crop((h // 2 - ch, 0, h, w // 2 + cw))),
            #         self.transform(x.crop((h // 2 - ch, w // 2 - cw, h, w)))]

            return [
                    self.transform(transforms.functional.resized_crop(x,0,0,h//2+ch,w//2+cw,(32,32))),
                    self.transform(transforms.functional.resized_crop(x,0, w//2-cw,  h//2+ch,w,(32,32))),
                    self.transform(transforms.functional.resized_crop(x,h//2-ch,0, h, w//2+cw,(32,32))),
                    self.transform(transforms.functional.resized_crop(x,h//2-ch, w//2-cw,  h, w,(32,32)))]

        if self.opt.dataset =='CIFAR-FS':

            x = transforms.RandomCrop(32, padding=4)(x)
            if m == 2:
                return [self.transform(x.crop((0, 0, h // 2 + ch, w // 2 + cw))),
                                 self.transform(x.crop((0, w // 2 - cw, h // 2 + ch, w))),
                                 self.transform(x.crop((h // 2 - ch, 0, h, w // 2 + cw))),
                                 self.transform(x.crop((h // 2 - ch, w // 2 - cw, h, w)))]
            else:
                return self.get_pyramid(x,m)


        if self.opt.dataset == 'miniImageNet' or self.opt.dataset == 'tieredImageNet':
            x = transforms.RandomCrop(84, padding=8)(x)

            return [self.transform(x.crop((0, 0, h // 2 + ch, w // 2 + cw))),
                            self.transform(x.crop((0, w // 2 - cw, h // 2 + ch, w))),
                            self.transform(x.crop((h // 2 - ch, 0, h, w // 2 + cw))),
                            self.transform(x.crop((h // 2 - ch, w // 2 - cw, h, w)))]


def get_train_dataloader(opt):

    train_partition = 'trainval' if opt.use_trainval else 'train'
    train_trans, test_trans = transforms_options[opt.transform]
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_dataset = CIFAR100(args=opt, partition=train_partition, transform=train_trans)
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


        train_trans_A = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_trans_B = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_loader = DataLoader(Preprocessor(train_dataset.img_label,[train_trans_A,train_trans_B], JigCluTransform(opt,trans_crop)),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers, pin_memory=True)
        val_loader = DataLoader(CIFAR100(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            else:
                 n_cls = 60

    elif opt.dataset == 'tieredImageNet':
        train_dataset = TieredImageNet(args=opt, partition=train_partition, transform=train_trans)
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        trans_crop = transforms.Compose([
            transforms.RandomCrop(42, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])


        train_trans_A = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_trans_B = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_loader = DataLoader(Preprocessor(train_dataset.img_label,[train_trans_A,train_trans_B], JigCluTransform(opt,trans_crop)),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers,
                                  )
        val_loader = DataLoader(TieredImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351

    elif opt.dataset == 'miniImageNet' and not opt.cross_domain:
        train_dataset = ImageNet(args=opt, partition=train_partition, transform=train_trans)

        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        trans_crop = transforms.Compose([
            transforms.RandomCrop(42, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_trans_A = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_trans_B = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_loader = DataLoader(Preprocessor(train_dataset.img_label,[train_trans_A,train_trans_B], JigCluTransform(opt,trans_crop)),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers,
                                  )
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)

        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64

    elif opt.cross_domain and opt.dataset =='miniImageNet':
        assert opt.transform == "A"

        train_dataset = ImageNet(args=opt, partition='train', transform=train_trans)
        val_dataset = ImageNet(args=opt, partition='val', transform=train_trans)
        test_dataset = ImageNet(args=opt, partition='test', transform=train_trans)
        # all_datasets = ConcatDataset([train_dataset, val_dataset, test_dataset])

        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        trans_crop = transforms.Compose([
            transforms.RandomCrop(42, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_trans_A = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_trans_B = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = Preprocessor(train_dataset.img_label,[train_trans_A, train_trans_B], JigCluTransform(opt, trans_crop))
        val_dataset = Preprocessor(val_dataset.img_label,[train_trans_A, train_trans_B], JigCluTransform(opt, trans_crop))
        test_dataset = Preprocessor(test_dataset.img_label,[train_trans_A, train_trans_B], JigCluTransform(opt, trans_crop))
        all_datasets = ConcatDataset([train_dataset, val_dataset, test_dataset])

        train_loader = DataLoader(all_datasets,
            batch_size=opt.batch_size, shuffle=True, drop_last=True,
            num_workers=opt.num_workers,
            )

        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        n_cls = 64 + 16 + 20  # train + val + test

    else:
        raise NotImplementedError(opt.dataset)

    return train_loader,val_loader,n_cls

class Preprocessor(Dataset):
    def __init__(self, dataset,transforms, jig_transforms=None):
        super(Preprocessor, self).__init__()
        self.jig_transforms = jig_transforms
        self.transforms = transforms
        self.label_transform = torch.LongTensor()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label,_ = self.dataset[item]

        img_jig = self.jig_transforms(img,2)
        img1 = self.transforms[0](img)
        img2 = self.transforms[1](img)
        return img_jig,[img1,img2], label,item

def get_eval_dataloader(opt):
    """
    Create the evaluation dataloaders
    """
    train_trans, test_trans = transforms_test_options[opt.transform]

    # ImagetNet derivatives - miniImageNet
    if opt.dataset == 'miniImageNet' and not opt.cross_domain:
        assert opt.transform == "A"
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test', train_transform=train_trans,
                                    test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val', train_transform=train_trans,
                                    test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64

    # ImagetNet derivatives - tieredImageNet
    elif opt.dataset == 'tieredImageNet':
        assert opt.transform == "A"
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                    train_transform=train_trans, test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val', train_transform=train_trans,
                                    test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351

    # CIFAR-100 derivatives - both CIFAR-FS & FC100
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        assert opt.transform == "D"
        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test', train_transform=train_trans,
                                    test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val', train_transform=train_trans,
                                    test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))

    # For cross-domain - we evaluate on a new dataset / domain
    elif opt.cross_dataset in ['cub', 'cars', 'places', 'plantae'] and opt.cross_domain:
        # train_classes = {'cub': 100, 'cars': 98, 'places': 183, 'plantae': 100}

        # assert opt.transform == "C"
        assert not opt.use_trainval, f"Train val option not possible for dataset {opt.dataset}"
        opt.transform = 'C'

        meta_testloader = DataLoader(MetaCUB(args=opt, partition='novel',
                                    train_transform=train_trans, test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCUB(args=opt, partition='val', train_transform=train_trans,
                                    test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        # n_cls = train_classes[opt.dataset]
        n_cls = 100

    else:
        raise NotImplementedError(opt.dataset)

    return meta_testloader, meta_valloader, n_cls

def get_aux_dataloader(opt):
    train_partition = 'trainval' if opt.use_trainval else 'train'

    if opt.dataset == 'miniImageNet' :
        train_trans, test_trans = transforms_options[opt.transform]
        dataset = ImageNet(args=opt, partition=train_partition, transform=test_trans)
        train_loader = DataLoader(dataset,
                                  batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                  num_workers=opt.num_workers)
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        dataset = TieredImageNet(args=opt, partition=train_partition, transform=test_trans)
        train_loader = DataLoader(dataset,
                                  batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                  num_workers=opt.num_workers)

    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options[opt.transform]
        dataset = CIFAR100(args=opt, partition=train_partition, transform=test_trans)
        train_loader = DataLoader(dataset,
                                  batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                  num_workers=opt.num_workers)

    elif opt.cross_domain and opt.dataset == 'miniImageNet':
        assert opt.transform == "A"
        train_trans, test_trans = transforms_options[opt.transform]
        train_dataset = ImageNet(args=opt, partition='train', transform=test_trans)
        val_dataset = ImageNet(args=opt, partition='val', transform=test_trans)
        test_dataset = ImageNet(args=opt, partition='test', transform=test_trans)
        dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

        train_loader = DataLoader(dataset,
                                  batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                  num_workers=opt.num_workers,
                                  )
    else:
        raise NotImplementedError(opt.dataset)
    return dataset,train_loader


