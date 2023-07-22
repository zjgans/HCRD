from __future__ import print_function

import io
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
normalize = transforms.Normalize(mean=mean, std=std)

transform_A = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize
    ])
]

transform_A_test = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize
    ])
]


# CIFAR style transformation
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
normalize_cifar100 = transforms.Normalize(mean=mean, std=std)

transform_D = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#         transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ])
]

transform_D_test = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ])
]

"""
CUB / CARS / places / plantae style transformation
"""

mean_cub = [0.485, 0.456, 0.406]
std_cub = [0.229, 0.224, 0.225]
normalize_cub = transforms.Normalize(mean=mean_cub, std=std_cub)

### -> Supervised transforms

transform_C = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x).copy(),
        transforms.ToTensor(),
        # normalize_cub
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        # normalize_cub
        normalize
    ])
]

transform_C_test = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x).copy(),
        transforms.ToTensor(),
        normalize_cub
        # normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        # normalize_cub
        normalize
    ])
]

# cub_normalize = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
#                                      np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
#
#
# transform_E=[transforms.Compose([
#         lambda x: Image.fromarray(x),
#         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#         transforms.RandomHorizontalFlip(),
#         lambda x: np.asarray(x).copy(),
#         transforms.ToTensor(),
#         # normalize_cub
#         normalize_imagenet
#     ]),
#
#     transforms.Compose([
#         lambda x: Image.fromarray(x),
#         transforms.ToTensor(),
#         # normalize_cub
#         normalize_imagenet
#     ])]
#
# transform_E_test = [transforms.Compose([
#         lambda x: Image.fromarray(x),
#         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#         transforms.RandomHorizontalFlip(),
#         lambda x: np.asarray(x).copy(),
#         transforms.ToTensor(),
#         # normalize_cub
#         normalize_imagenet
#     ]),
#
#     transforms.Compose([
#         lambda x: Image.fromarray(x),
#         transforms.ToTensor(),
#         # normalize_cub
#         normalize_imagenet
#     ])]

transforms_list = ['A','D','C']

transforms_options = {
    'A': transform_A,
    'D': transform_D,
    'C': transform_C
}

transforms_test_options = {
    'A': transform_A_test,
    'D': transform_D_test,
    'C': transform_C_test
}
