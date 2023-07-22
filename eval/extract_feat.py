import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
from eval.util import AverageMeter

def extract_cnn_feature(model,inputs):
    inputs = inputs.cuda()
    outputs = model(inputs,is_feat=True)
    return outputs

def extract_features(model, data_loader, print_freq=50,types='proto'):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, label, fnames,_) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for index, output, pid in zip(fnames, outputs, label):
                features[index] = output
                labels[index] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels