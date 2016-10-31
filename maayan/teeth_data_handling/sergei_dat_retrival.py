# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:55:55 2016

@author: maayan
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def get_training_data_from_folder(folder):
    
    train_imgs = list()
    labels = list()
    for filename in os.listdir(folder):
        append_data(filename, folder, train_imgs, labels)
    data = {'images': np.array(train_imgs), 'labels': np.array(labels)}
    return data
    
def append_data(filename, folder, train_imgs, labels):
    
    if filename.endswith('.png'):
        im_dat = get_training_data_from_file(os.path.join(folder, filename))
        train_imgs.append(im_dat['train_img'])
        labels.append(im_dat['labels'])

def get_training_data_from_file(path):
    
    img = plt.imread(path)
    data = get_training_data_from_figure(img)
    return data

def get_training_data_from_figure(img):
    
    train_img = img[:, :, 1] - 0.5
    (values,counts) = np.unique(img[:,:,2]*32,return_counts=True)
    labels = np.zeros(32)
    labels[int(values[1])] = 1
    return {'train_img': train_img, 'labels': labels}