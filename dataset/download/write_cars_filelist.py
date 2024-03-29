import numpy as np
from os.path import join
import os, sys
import random
from scipy.io import loadmat

# DATASET_PATH = sys.argv[1]
DATASET_PATH = '/data/lxj/odata/dataset/cars/'
data_path = join(DATASET_PATH, 'cars_train')
savedir = DATASET_PATH
dataset_list = ['base','val','novel']

path_data_list = join(DATASET_PATH, 'devkit/cars_train_annos.mat')
path_class_list = join(DATASET_PATH, 'devkit/cars_meta.mat')
data_list = np.array(loadmat(path_data_list)['annotations'][0])
class_list = np.array(loadmat(path_class_list)['class_names'][0])
classfile_list_all = [[] for i in range(len(class_list))]

for i in range(len(data_list)):
  folder_path = join(data_path, data_list[i][-1][0])
  classfile_list_all[data_list[i][-2][0][0] - 1].append(folder_path)

for i in range(len(classfile_list_all)):
  random.shuffle(classfile_list_all[i])

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            if (i%2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(join(savedir, dataset+".json"), "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item[0]  for item in class_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
