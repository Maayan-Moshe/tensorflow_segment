# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def calc_filtered_image_size(image_size, filter_size, stride, padding = 'SAME'):
    zero_padding = 0
    if padding == 'SAME':
        zero_padding = (filter_size - 1) // 2
    filtered_size = 1 + (image_size - filter_size + 2*zero_padding) // stride
    return filtered_size
    
    
def weight_variable(shape, stddev=-1):
    if stddev <= 0:
        num_inputs = calc_flat_size(shape[:-1]) # last element is output dimention
        stddev = calc_optimal_weights_stdev(num_inputs)
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(num_out_channels, init_val=0.0):
    initial = tf.constant(init_val, shape=[num_out_channels])
    return tf.Variable(initial)


def conv2d(x, W, stride=1, padding='SAME'):
    ''' x input tensor of shape [batch, in_height, in_width, in_channels]
        W filter shape - [filter_height, filter_width, in_channels, out_channels]
        1-D of length 4. The stride of the sliding window for each dimension of input.
        padding ='SAME' -  zero padded so that the output is the same size as the input'''
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, block_size=2, stride=2, padding='SAME'):
    ''' x A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
        ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.'''
    return tf.nn.max_pool(x, ksize=[1, block_size, block_size, 1], strides=[1, stride, stride, 1], padding=padding)


def avg_pool(x, block_size=2, stride=2, padding='SAME'):
    ''' x A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
        ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.'''
    return tf.nn.avg_pool(x, ksize=[1, block_size, block_size, 1], strides=[1, stride, stride, 1], padding=padding)


def activate(logits, activation_func_name='relu'):
    if activation_func_name == '':
        return logits
    if activation_func_name == 'elu':
        return tf.nn.elu(logits) 
    return tf.nn.relu(logits) 


def calc_optimal_weights_stdev(num_prev_layer_params):
    '''source:  http://arxiv.org/pdf/1502.01852v1.pdf'''
    return np.sqrt(2.0 / num_prev_layer_params)


def calc_flat_size(shape):
    if len(shape) ==0 :
        return 0
    size = 1
    for dim in shape:
        size *= dim
    return size


def accuracy(predictions, labels):
    num_samples = predictions.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / num_samples)
    
    
def reformat(dataset, labels, params):
    ''' 
    Reformat into a TensorFlow-friendly shape:
    convolutions need the image data formatted as a cube (width by height by #channels)
    labels as float 1-hot encodings.
    '''
    img_size = params['image_size']
    num_chan = params['num_channels']
    dataset = dataset.reshape((-1, img_size, img_size, num_chan)).astype(np.float32)
    labels = convert_labels_to_onehot_vectors(labels, params['num_labels'])
    return dataset, labels
    
    
def convert_labels_to_onehot_vectors(labels, num_labels):
    '''
    Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    '''
    label_classes = np.arange(num_labels)
    labels = (label_classes == labels[:,None]).astype(np.float32)    
    return labels
    
    
    
def print_params(hyperparams):
    print('----Hyperparameters----')
    for param_name in hyperparams.keys():
        value = hyperparams[param_name]
        if type(value) is list:
            print('%s values:' % param_name)
            for i, val in enumerate(value):
                print('%d: %s' % (i+1, val))
        else:
            print(param_name, value)
    print('-----------------------')    
    
    
def print_to_log(data, text='', dbg_log_fn = 'dbg_log.txt', mode='a'):    
    if dbg_log_fn:
        dbg_log = open(dbg_log_fn, mode)    
        print(text, data, file=dbg_log)
        dbg_log.close()