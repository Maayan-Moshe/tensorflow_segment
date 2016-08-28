
# coding: utf-8

# Deep Learning - Convolutional Neural Network
# =============
# 
# 
# Purpose - classification of [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
# 

# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
import time
import math

# In[3]:


def load_data(pickle_file = 'notMNIST.pickle'):

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        
        return train_dataset, train_labels, valid_dataset, valid_labels, valid_dataset, valid_labels 


# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

# In[4]:

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def convert_labels_to_onehot_vectors(labels):
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    label_classes = np.arange(num_labels)
    labels = (label_classes == labels[:,None]).astype(np.float32)    
    return labels

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = convert_labels_to_onehot_vectors(labels)
    return dataset, labels

# Let's build a small network with two convolutional layers, followed by one fully connected layer. Convolutional networks are more expensive computationally, so we'll limit its depth and number of fully connected nodes.

# Different parameters sets
# ---

# In[5]:

def get_default_params():
    # define hyperparameters
    hyperparams = dict()
    hyperparams['image_size'] = 28 # input layer
    hyperparams['num_channels'] = 1 # input layer
    hyperparams['num_labels'] = 10 # output layer
    num_labels = hyperparams['num_labels']
    hyperparams['layers_info'] = [{'type': 'conv', 'kernel_width' : 5, 'depth': 16, 'stride': 2, 'padding' : 'SAME'}, 
                                  {'type': 'conv', 'kernel_width' : 5, 'depth': 16, 'stride': 2, 'padding' : 'SAME'}, 
                                  {'type': 'fc', 'depth': 64},
                                  {'type': 'fc', 'depth': num_labels}]
    
    hyperparams['weight_stddev'] = 0.1
    hyperparams['bias_init_val'] = 0.0
    hyperparams['activation_func'] = 'relu'
    
    hyperparams['optimizer_alg_name'] = 'GradientDescentOptimizer'
    hyperparams['momentum_term'] = 0
    hyperparams['learning_rate'] = 0.05
    hyperparams['learning_decay_rate'] = 1.0 
    hyperparams['learning_decay_steps'] =1000
    hyperparams['dropout_keep_prob'] = 1.0  

    hyperparams['batch_size'] = 16
    hyperparams['max_num_samples'] = -1 # all
    
    hyperparams['num_steps'] = 1001 
    hyperparams['num_logs'] = 30
    
    return hyperparams
    
    
def get_optimal_params():
    # define hyperparameters
    hyperparams = dict()
    hyperparams['image_size'] = 28 # input layer
    hyperparams['num_channels'] = 1 # input layer
    hyperparams['num_labels'] = 10 # output layer
    num_labels = hyperparams['num_labels']
    hyperparams['layers_info'] = [{'type': 'conv', 'kernel_width' : 5, 'depth': 16, 'stride': 1, 'padding' : 'SAME'}, 
                                 {'type': 'pool', 'kernel_width' : 2, 'stride' : 2, 'padding' : 'SAME', 'pool_type' : 'maxpool'},  
                                 {'type': 'conv', 'kernel_width' : 5, 'depth': 16, 'stride': 1, 'padding' : 'SAME'}, 
                                 {'type': 'pool', 'kernel_width' : 2, 'stride' : 2, 'padding' : 'SAME', 'pool_type' : 'maxpool'}, 
                                 {'type': 'fc', 'depth': 64},
                                 {'type': 'fc', 'depth': num_labels}]
    
    hyperparams['weight_stddev'] = 0.1
    hyperparams['bias_init_val'] = 0.0
    hyperparams['activation_func'] = 'elu'
    
    hyperparams['optimizer_alg_name'] = 'GradientDescentOptimizer'
    hyperparams['momentum_term'] = 0.9
    hyperparams['learning_rate'] = 0.05
    hyperparams['learning_decay_rate'] = 0.98 
    hyperparams['learning_decay_steps'] =1000
    hyperparams['dropout_keep_prob'] = 1.0 

    hyperparams['batch_size'] = 64
    hyperparams['max_num_samples'] = -1 # all
    
    hyperparams['num_steps'] = 5001    
    hyperparams['num_logs'] = 50
    
    return hyperparams


def get_lenet_params():
    # define hyperparameters
    hyperparams = dict()
    hyperparams['image_size'] = 28 # input layer
    hyperparams['num_channels'] = 1 # input layer
    hyperparams['num_labels'] = 10 # output layer
    num_labels = hyperparams['num_labels']
    hyperparams['layers_info'] = [{'type': 'conv', 'kernel_width' : 5, 'depth': 8, 'stride': 1, 'padding' : 'SAME'}, 
                                 {'type': 'pool', 'kernel_width' : 2, 'stride' : 2, 'padding' : 'SAME', 'pool_type' : 'maxpool'},  
                                 {'type': 'conv', 'kernel_width' : 5, 'depth': 32, 'stride': 1, 'padding' : 'SAME'}, 
                                 {'type': 'pool', 'kernel_width' : 2, 'stride' : 2, 'padding' : 'SAME', 'pool_type' : 'maxpool'}, 
                                 {'type': 'conv', 'kernel_width' : 5, 'depth': 64, 'stride': 1, 'padding' : 'SAME'}, 
                                 {'type': 'fc', 'depth': 128},
                                 {'type': 'fc', 'depth': 84},
                                 {'type': 'fc', 'depth': num_labels}]
    
    hyperparams['weight_stddev'] = 0.1
    hyperparams['bias_init_val'] = 0.0
    hyperparams['activation_func'] = 'elu'
    
    hyperparams['optimizer_alg_name'] = 'GradientDescentOptimizer'
    hyperparams['momentum_term'] = 0.0
    hyperparams['learning_rate'] = 0.1
    hyperparams['learning_decay_rate'] = 0.95 
    hyperparams['learning_decay_steps'] = 1000
    hyperparams['dropout_keep_prob'] = 0.7

    hyperparams['batch_size'] = 64
    hyperparams['max_num_samples'] = -1 # all
    
    hyperparams['num_steps'] = 50001    
    hyperparams['num_logs'] = 50
    
    return hyperparams
    
    
def get_inception_params():
    # define hyperparameters
    hyperparams = dict()
    hyperparams['image_size'] = 28 # input layer
    hyperparams['num_channels'] = 1 # input layer
    hyperparams['num_labels'] = 10 # output layer
    num_labels = hyperparams['num_labels']
    hyperparams['layers_info'] = [
        {'type': 'conv', 'kernel_width' : 5, 'depth': 16, 'stride': 1, 'padding' : 'SAME'}, 
        {'type': 'pool', 'kernel_width' : 2, 'stride' : 2, 'padding' : 'SAME', 'pool_type' : 'maxpool'}, 
        {'type': 'conv', 'kernel_width' : 5, 'depth': 32, 'stride': 1, 'padding' : 'SAME'}, 
        {'type': 'inception', '1x1':12, '3x3_reduced':10, '3x3':20, '5x5_reduced':4, '5x5':10, 'pool_reduced':6}, 
        {'type': 'inception', '1x1':12, '3x3_reduced':12, '3x3':24, '5x5_reduced':4, '5x5':12, 'pool_reduced':8}, 
        {'type': 'pool', 'kernel_width' : 2, 'stride' : 2, 'padding' : 'SAME', 'pool_type' : 'maxpool'}, 
        {'type': 'inception', '1x1':16, '3x3_reduced':14, '3x3':28, '5x5_reduced':4, '5x5':12, 'pool_reduced':8}, 
        {'type': 'inception', '1x1':24, '3x3_reduced':20, '3x3':40, '5x5_reduced':12, '5x5':20, 'pool_reduced':12},
        {'type': 'inception', '1x1':32, '3x3_reduced':20, '3x3':56, '5x5_reduced':16, '5x5':24, 'pool_reduced':16},        
        {'type': 'pool', 'kernel_width' : 7, 'stride' : 1, 'padding' : 'VALID', 'pool_type' : 'avgpool'}, 
        {'type': 'fc', 'depth': num_labels}]
    
    hyperparams['weight_stddev'] = -1 # will calculate optimal stdev
    hyperparams['bias_init_val'] = 0.01
    hyperparams['activation_func'] = 'elu'
                
    #GradientDescentOptimizer, MomentumOptimizer, AdagradOptimizer, AdamOptimizer
    hyperparams['optimizer_alg_name'] = 'GradientDescentOptimizer'
    hyperparams['momentum_term'] = 0.8
    hyperparams['learning_rate'] = 0.1
    hyperparams['learning_decay_rate'] = 0.95 
    hyperparams['learning_decay_steps'] = 1000
    hyperparams['dropout_keep_prob'] = 0.7

    hyperparams['batch_size'] = 128
    hyperparams['max_num_samples'] = -1 # all

    hyperparams['num_full_epochs'] = 20        
    hyperparams['num_steps'] = 50001 # if num_full_epochs is defined then num_steps = num_full_epochs* num_sampled /batch_size  
    hyperparams['num_logs'] = hyperparams['num_full_epochs'] 
    
    return hyperparams    
 
    
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


# Classes definitions for different layer types
# ---

# In[6]:

def create_layer_module(layer_info, data_shape, activation_func):
    # class factory
    layer_obj = None             
    
    layer_type = layer_info['type']
    if layer_type == 'conv':
        layer_obj = conv_layer(layer_info, data_shape, activation_func)  
    elif layer_type == 'pool':
        layer_obj = pooling_layer(layer_info, data_shape, layer_info['pool_type'])    
    elif layer_type == 'inception':
        layer_obj = inception_module(layer_info, data_shape, activation_func)     
    elif layer_type == 'fc':           
        layer_obj = fc_layer(layer_info, data_shape, activation_func)  
        
    return layer_obj


#---------------------------------------------------------------------------

class layer_module_base:    
    def __init__(self, params, data_shape, activation_func = 'relu'): 
        self.activation_func = activation_func
        self.params = params        
        self.data_shape = data_shape
        self.out_data_shape = data_shape
        self.depth = 0
        self.weights_shape = []
    
    def build(self, bias_init_val=0, weights_stddev=-1):
        if len(self.weights_shape)==0:
            return
        self.weights = weight_variable(self.weights_shape, stddev=weights_stddev)
        self.biases = bias_variable(self.depth, init_val=bias_init_val)   
        
    def eval(self, data, dropout_keep_prob=1.0): 
        return data
    
    def calc_params_count(self):
        return 0
    
    def calc_num_input_nodes(self):
        return 0

    
#---------------------------------------------------------------------------
    
class conv_layer(layer_module_base):
    def __init__(self, params, data_shape, activation_func = 'relu'):
        layer_module_base.__init__(self, params, data_shape, activation_func)
        self.stride = self.params['stride']
        self.padding = self.params['padding']
        self.filter_size = self.params['kernel_width']
        self.depth = self.params['depth']
        prev_layer_depth = self.data_shape[-1] 
        self.weights_shape = [self.filter_size, self.filter_size, prev_layer_depth, self.depth]
        self.out_data_shape = self.__calc_out_data_shape()
                
    def eval(self, data, dropout_keep_prob=1.0): 
        logits = conv2d(data, self.weights, stride=self.stride , padding=self.padding) + self.biases
        activations = activate(logits, self.activation_func)
        return activations
    
    def calc_params_count(self):
        prev_layer_depth = self.data_shape[-1] 
        filter_size = self.filter_size * self.filter_size * prev_layer_depth
        weights_size = filter_size * self.depth
        bias_size = self.depth
        return weights_size + bias_size
    
    def __calc_out_data_shape(self):
        h, w, depth = self.data_shape
        out_h = calc_filtered_image_size(h, self.filter_size, self.stride, self.padding)
        out_w = calc_filtered_image_size(w, self.filter_size, self.stride, self.padding)
        return [out_h, out_w, self.depth]  
    



#---------------------------------------------------------------------------


class fc_layer(layer_module_base):
    def __init__(self, params, data_shape, activation_func = 'relu'):
        layer_module_base.__init__(self, params, data_shape, activation_func)
        self.depth = self.params['depth']
        self.data_shape = [calc_flat_size(self.data_shape)]
        prev_layer_depth = self.data_shape[-1] 
        self.weights_shape = [prev_layer_depth, self.depth] 
        self.out_data_shape = self.__calc_out_data_shape()
        
    def eval(self, data, dropout_keep_prob=1.0): 
        data = self.__flatten_data_shape(data)     
        logits = tf.matmul(data, self.weights) + self.biases
        if self.activation_func == '':
            return logits
        activations = activate(logits, self.activation_func)  
        activations = tf.nn.dropout(activations, dropout_keep_prob)
        return activations     
    
    def calc_params_count(self):
        prev_layer_depth = self.data_shape[-1] 
        weights_size = prev_layer_depth * self.depth
        bias_size = self.depth
        return weights_size + bias_size
    
    def __calc_out_data_shape(self):
        return [self.depth]   
            
    def __flatten_data_shape(self, data):
        shape = data.get_shape().as_list()
        if len(shape) <= 2:
            return data # already flat
        batch = shape[0]
        size = calc_flat_size(shape[1:])
        data = tf.reshape(data, [batch, size])
        return data

    
#---------------------------------------------------------------------------    

    
class pooling_layer(layer_module_base):
    def __init__(self, params, data_shape, activation_func = 'maxpool'):
        layer_module_base.__init__(self, params, data_shape, activation_func) 
        self.stride = self.params['stride']
        self.padding = self.params['padding']
        self.filter_size = self.params['kernel_width']
        self.out_data_shape = self.__calc_out_data_shape()

    def build(self, weights_stddev=0.1, bias_init_val=0): pass
        
    def eval(self, data, dropout_keep_prob=1.0): 
        if self.activation_func == 'maxpool':
            pooled_data = max_pool(data, block_size=self.filter_size , stride=self.stride, padding=self.padding) 
        else:
            pooled_data = avg_pool(data, block_size=self.filter_size , stride=self.stride, padding=self.padding) 
        return pooled_data
 
    def __calc_out_data_shape(self):
        h, w, depth = self.data_shape
        out_h = calc_filtered_image_size(h, self.filter_size, self.stride, self.padding)
        out_w = calc_filtered_image_size(w, self.filter_size, self.stride, self.padding)
        return [out_h, out_w, depth]         
    
        
#---------------------------------------------------------------------------        
        
    
class inception_module(layer_module_base):
    def __init__(self, params, data_shape, activation_func = 'relu'):
        layer_module_base.__init__(self, params, data_shape, activation_func)  
        self.prev_layer_depth = self.data_shape[-1] 
        self.out_data_shape = self.calc_out_data_shape()
        self.stride = 1
        self.padding = 'SAME'
        # params is a dictionary of the following structure:
        #{
        #    'type':'inception',
        #     '1x1':<depth>, 
        #     '3x3_reduced':<depth>, '3x3':<depth>, 
        #     '5x5_reduced':<depth>, '5x5':<depth>, 
        #     'pool_reduced':<depth>
        #}      

     
    def build(self, bias_init_val=0, weights_stddev=-1):
        self.weights_1x1 = weight_variable([1, 1, self.prev_layer_depth, self.params['1x1']], stddev=weights_stddev)
        self.biases_1x1 = bias_variable(self.params['1x1'], init_val=bias_init_val)

        self.weights_3x3_reduced = weight_variable([1, 1, self.prev_layer_depth, self.params['3x3_reduced']], stddev=weights_stddev)
        self.biases_3x3_reduced = bias_variable(self.params['3x3_reduced'], init_val=bias_init_val)
        self.weights_3x3 = weight_variable([3, 3, self.params['3x3_reduced'], self.params['3x3']], stddev=weights_stddev)
        self.biases_3x3 = bias_variable(self.params['3x3'], init_val=bias_init_val)

        self.weights_5x5_reduced = weight_variable([1, 1, self.prev_layer_depth, self.params['5x5_reduced']], stddev=weights_stddev)
        self.biases_5x5_reduced = bias_variable(self.params['5x5_reduced'], init_val=bias_init_val)
        self.weights_5x5 = weight_variable([5, 5, self.params['5x5_reduced'], self.params['5x5']], stddev=weights_stddev)
        self.biases_5x5 = bias_variable(self.params['5x5'], init_val=bias_init_val)    

        self.weights_pool_reduced = weight_variable([1, 1, self.prev_layer_depth, self.params['pool_reduced']], stddev=weights_stddev)
        self.biasespool_reduced= bias_variable(self.params['pool_reduced'], init_val=bias_init_val)  
        
        
    def eval(self, data, dropout_keep_prob=1.0):
        activations_1x1 = self.calc_activations(data, self.weights_1x1, self.biases_1x1)
        
        activations_3x3_reduced = self.calc_activations(data, self.weights_3x3_reduced, self.biases_3x3_reduced)
        activations_3x3 = self.calc_activations(activations_3x3_reduced, self.weights_3x3, self.biases_3x3)
        
        activations_5x5_reduced = self.calc_activations(data, self.weights_5x5_reduced, self.biases_5x5_reduced)
        activations_5x5 = self.calc_activations(activations_5x5_reduced, self.weights_5x5, self.biases_5x5)        
        
        activations_max_pool_3x3 = max_pool(data, block_size=3, stride=1, padding='SAME')  
        activations_pool_reduced = self.calc_activations(activations_max_pool_3x3, self.weights_pool_reduced, self.biasespool_reduced)
        
        depth_concat = tf.concat(3, [activations_3x3, activations_1x1, activations_5x5, activations_pool_reduced])
        return depth_concat
    
    def calc_activations(self, data, weights, biases):
        logits = conv2d(data, weights, stride=self.stride, padding=self.padding) + biases
        activations = activate(logits, self.activation_func)
        return activations    
    
    def calc_params_count(self):
        size_1x1 = self.__calc_conv_params_count(1, 1, self.prev_layer_depth, self.params['1x1'])
        size_3x3_reduced = self.__calc_conv_params_count(3, 3, self.prev_layer_depth, self.params['3x3_reduced'])
        size_3x3 = self.__calc_conv_params_count(3, 3, self.params['3x3_reduced'], self.params['3x3']) 
        size_5x5_reduced = self.__calc_conv_params_count(5, 5, self.prev_layer_depth, self.params['5x5_reduced'])
        size_5x5 = self.__calc_conv_params_count(5, 5, self.params['5x5_reduced'], self.params['5x5']) 
        size_pool_reduced = self.__calc_conv_params_count(1, 1, self.prev_layer_depth, self.params['pool_reduced'])
        total_size = size_1x1 + size_3x3_reduced + size_3x3 + size_5x5_reduced + size_5x5 + size_pool_reduced
        return total_size   
    
    def __calc_conv_params_count(self, width, height, depth, num_filters):
        weights_size = width * height * depth * num_filters
        bias_size = num_filters
        return weights_size + bias_size
        

    def calc_out_data_shape(self):
        h, w, depth = self.data_shape
        out_depth = self.params['1x1'] + self.params['3x3'] + self.params['5x5'] + self.params['pool_reduced']
        return [h, w, out_depth]    

    
#---------------------------------------------------------------------------    

    


# Helper functions
# ---

# In[7]:

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
    # x input tensor of shape [batch, in_height, in_width, in_channels]
    # W filter shape - [filter_height, filter_width, in_channels, out_channels]
    # 1-D of length 4. The stride of the sliding window for each dimension of input.
    #padding ='SAME' -  zero padded so that the output is the same size as the input
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

def max_pool(x, block_size=2, stride=2, padding='SAME'):
    # x A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
    # ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    return tf.nn.max_pool(x, ksize=[1, block_size, block_size, 1], strides=[1, stride, stride, 1], padding=padding)

def avg_pool(x, block_size=2, stride=2, padding='SAME'):
    # x A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
    # ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    return tf.nn.avg_pool(x, ksize=[1, block_size, block_size, 1], strides=[1, stride, stride, 1], padding=padding)

def activate(logits, activation_func_name='relu'):
    if activation_func_name == '':
        return logits
    if activation_func_name == 'elu':
        return tf.nn.elu(logits) 
    return tf.nn.relu(logits) 

def calc_optimal_weights_stdev(num_prev_layer_params):
    #source:  http://arxiv.org/pdf/1502.01852v1.pdf
    return np.sqrt(2.0 / num_prev_layer_params)

def calc_flat_size(shape):
    if len(shape) ==0 :
        return 0
    size = 1
    for dim in shape:
        size *= dim
    return size


def calc_filtered_image_size(image_size, filter_size, stride, padding = 'SAME'):
    zero_padding = 0
    if padding == 'SAME':
        zero_padding = (filter_size - 1) // 2
    filtered_size = 1 + (image_size - filter_size + 2*zero_padding) // stride
    return filtered_size


def accuracy(predictions, labels):
    num_samples = predictions.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / num_samples)


# Building computational graph model functionality
# ---

# In[8]:

def build_model(hyperparams, valid_dataset, test_dataset):    
    batch_size = hyperparams['batch_size']
    image_size = hyperparams['image_size']
    num_channels = hyperparams['num_channels']
    num_labels = hyperparams['num_labels'] 
    
    # build computational graph flow model
    graph = tf.Graph()
    with graph.as_default():
        
        # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)             
        
        # add layers variables - weight/biases
        layers_objects = build_network_layers(hyperparams)     

        # Optimisation --------------------------------------
        tf_dropout_keep_prob = tf.placeholder(tf.float32)  
        train_logits = forward_prop(tf_train_dataset, layers_objects, tf_dropout_keep_prob)
        loss, optimizer = add_optimizer(train_logits, tf_train_labels, hyperparams)

        # Predictions  ------------------------------------------
        train_prediction = tf.nn.softmax(train_logits)
        # no dropout on validation and test predictions
        valid_prediction = tf.nn.softmax(forward_prop(tf_valid_dataset, layers_objects)) 
        test_prediction = tf.nn.softmax(forward_prop(tf_test_dataset, layers_objects))        
        
        graph_elements = {}
        graph_elements['graph'] = graph
        graph_elements['tf_train_dataset'] = tf_train_dataset
        graph_elements['tf_train_labels'] = tf_train_labels
        graph_elements['optimizer'] = optimizer
        graph_elements['loss'] = loss
        graph_elements['train_prediction'] = train_prediction
        graph_elements['valid_prediction'] = valid_prediction
        graph_elements['test_prediction'] = test_prediction
        graph_elements['tf_dropout_keep_prob'] = tf_dropout_keep_prob
        return graph_elements
    
    

def build_network_layers(hyperparams):
    layers_objects = []
       
    layers_info = hyperparams['layers_info']
    num_layers = len(layers_info) # excluding input layer
    assert(num_layers > 0)
   
    activation_func = hyperparams['activation_func']
    total_params_count = 0
    
    data_shape = [hyperparams['image_size'], hyperparams['image_size'], hyperparams['num_channels']]    
    for layer in range(num_layers):
        layer_info = layers_info[layer]       
        if layer == (num_layers-1): # last layer
            activation_func = ''
        
        layer_obj = create_layer_module(layer_info, data_shape, activation_func)      
        if (layer_obj == None):
            continue
        
        layer_obj.build(bias_init_val=hyperparams['bias_init_val'], weights_stddev=hyperparams['weight_stddev'])
        layers_objects.append(layer_obj)     
        data_shape = layer_obj.out_data_shape
        
        params_count = layer_obj.calc_params_count()
        print('Layer % d %s : input_data_shape %s out_data_shape %s - total parameters = %d' % 
              (layer+1, layer_info['type'], layer_obj.data_shape, layer_obj.out_data_shape, params_count))
        total_params_count += params_count
    
    print('Total layers %d' % num_layers)
    print('Total parameters number = %d' % total_params_count)
    return layers_objects

    
def forward_prop(dataset, layers_objects, dropout_keep_prob=1.0):   
    num_layers = len(layers_objects)
    assert(num_layers > 0)
                       
    activations = dataset
    for layers_obj in layers_objects:  
        activations = layers_obj.eval(activations, dropout_keep_prob)

    return activations    


def add_optimizer(train_logits, tf_train_labels, hyperparams):
    learning_rate = hyperparams['learning_rate']
    learning_decay_steps = hyperparams['learning_decay_steps']
    learning_decay_rate = hyperparams['learning_decay_rate']
    momentum_term = hyperparams['momentum_term']
    optimizer_alg_name = hyperparams['optimizer_alg_name']
    
    # define loss function as cross-entropy between softmax of last layer predicted scores and labels.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))   
    
    # add variable for holding interations steps
    global_step = tf.Variable(0)  # count the number of steps taken.
    
    # Learning rate decay - for GradientDescentOptimizer or MomentumOptimizer
    # reference http://sebastianruder.com/optimizing-gradient-descent/
    if (learning_decay_rate < 1.0 and 
        (optimizer_alg_name == 'GradientDescentOptimizer' or optimizer_alg_name == 'MomentumOptimizer')) :
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, 
                                                   learning_decay_steps, learning_decay_rate, staircase=True)  
        
    if optimizer_alg_name == 'GradientDescentOptimizer':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer_alg_name == 'MomentumOptimizer':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum_term)        
    elif optimizer_alg_name == 'AdagradOptimizer':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer_alg_name == 'AdamOptimizer':
        optimizer = tf.train.AdamOptimizer(1e-4)
    else:
        hyperparams['optimizer_alg_name'] = 'GradientDescentOptimizer'
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    optimizer = optimizer.minimize(loss, global_step=global_step) 
        
    return loss, optimizer 


# Running and evaluating model functionality
# --

# In[9]:

def train_model(graph_elements, hyperparams, train_dataset, train_labels, \
                            valid_labels, test_labels):

    graph = graph_elements['graph']
    tf_train_dataset = graph_elements['tf_train_dataset']
    tf_train_labels = graph_elements['tf_train_labels']
    optimizer = graph_elements['optimizer']
    loss = graph_elements['loss']
    train_prediction = graph_elements['train_prediction']
    valid_prediction = graph_elements['valid_prediction']
    test_prediction = graph_elements['test_prediction']
    tf_dropout_keep_prob = graph_elements['tf_dropout_keep_prob']    
    
    t0 = time.clock()
    
    max_num_samples = hyperparams['max_num_samples']    
    num_samples = train_labels.shape[0]
    if max_num_samples > 0:
        num_samples = min(max_num_samples, num_samples)
 
    batch_size = hyperparams['batch_size']
    if 'num_full_epochs' in hyperparams:
        num_full_epochs = hyperparams['num_full_epochs']
        num_steps = num_full_epochs * (num_samples // batch_size)
    else:
        num_steps = hyperparams['num_steps'] # number of runs - minimization iterations
        num_full_epochs = num_steps / float(batch_size)
    num_steps_per_epoch = num_samples / batch_size
       
    log_step = num_steps // hyperparams['num_logs']
    print('log_step', log_step)
    
    dropout_keep_prob = hyperparams['dropout_keep_prob']
    
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("\nSession Initialized, running total %d steps %1.2f epochs" % (num_steps, num_full_epochs))
        
        train_accuracy_list = []
        validation_accuracy_list = []
        train_loss_vals = []
        plot_steps = []
                        
        # each step is optimization run on a single batch. 
        # num_steps *  batch_size can be more than num_samples to sloe multiple runs on full data
        for step in range(num_steps):

            # Generate a minibatch of already randomized training data
            sample_start_offset = (step * batch_size) % (num_samples - batch_size)
            batch_data = train_dataset[sample_start_offset : (sample_start_offset + batch_size), :, :, :]
            batch_labels = train_labels[sample_start_offset : (sample_start_offset + batch_size), :]

            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_dropout_keep_prob: dropout_keep_prob}
            _, train_loss = session.run([optimizer, loss], feed_dict=feed_dict)
            if math.isnan(train_loss):
                print('Breaking: Model divergence - train Loss is nan on step', step)
                break

            # log evaluations every log_step
            if (step % log_step == 0):
                t_eval0 = time.clock()
                print("--Train Minibatch step %d epoch %1.2f, time from start %1.3f min:" % 
                      (step, step/num_steps_per_epoch, ((t_eval0-t0)/60.0)))
                
                # all evaluations should be deterministic - without dropout
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_dropout_keep_prob: 1.0}
                train_predictions = train_prediction.eval(feed_dict=feed_dict)
                train_accuracy = accuracy(train_predictions, batch_labels)                
             
                valid_predictions = valid_prediction.eval(feed_dict = {tf_dropout_keep_prob:1.0})
                valid_accuracy = accuracy(valid_predictions, valid_labels)
     
                plot_steps.append(step)
                train_accuracy_list.append(train_accuracy)
                train_loss_vals.append(train_loss)
                validation_accuracy_list.append(valid_accuracy)
                
                t_eval1 = time.clock()
                print("Train loss %1.4f, Train/Validation accuracy (%1.2f , %1.2f) percent, eval_time %1.3f min" %
                      (train_loss, train_accuracy, valid_accuracy, (t_eval1-t_eval0)/60.0))
                
                test_accuracy = accuracy(test_prediction.eval(feed_dict = {tf_dropout_keep_prob:1.0}), test_labels)
                t_eval2 = time.clock()
                print('Test accuracy %1.2f, eval_time %1.3f min' % (test_accuracy, (t_eval2-t_eval1)/60.0))


        # draw learning curves        
        visualize_learning_curves(plot_steps, train_loss_vals, train_accuracy_list, validation_accuracy_list)       
        
        # calc test accuracy
        test_accuracy = accuracy(test_prediction.eval(feed_dict = {tf_dropout_keep_prob:1.0}), test_labels)
         
    t1 = time.clock()
    sec = t1-t0
    print('Total run time %s seconds = %1.2f minutes %1.3f hours' % (sec, sec/60.0, sec/60.0/60.0))
    
    return [train_accuracy_list[-1],  validation_accuracy_list[-1], test_accuracy, train_loss_vals[-1]]

         
def visualize_learning_curves(plot_steps, train_loss_vals, train_accuracy_list, validation_accuracy_list):
    plt.figure()
    plt.plot(plot_steps, train_loss_vals, label='train_loss')
    plt.legend(loc = 'best')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Train loss - final  train_loss %f' % train_loss_vals[-1])

    plt.figure()
    plt.plot(plot_steps, train_accuracy_list, label='train_accuracy')
    plt.plot(plot_steps, validation_accuracy_list, label='validation_accuracy')
    plt.xlabel('step')
    plt.ylabel('accuracy')
    plt.legend(loc = 'best')
    plt.title('Train/validation accuracy, final train_accuracy %f valid_accuracy %f' % 
                            (train_accuracy_list[-1], validation_accuracy_list[-1]))


# In[10]:

def run_model(hyperparams, train_dataset, train_labels, valid_dataset, \
                valid_labels, test_dataset, test_labels):
    print_params(hyperparams)
    
    # build graph flow model
    graph_elements = build_model(hyperparams, valid_dataset, test_dataset)

    # run graph flow model 
    run_res = train_model(graph_elements, hyperparams, train_dataset, train_labels, \
                                            valid_labels, test_labels)       
    train_accuracy, valid_accuracy, test_accuracy, train_loss = run_res
       
    # report final results
    print('------Final results-------')
    print('Final train loss %f' % train_loss)
    print('Final train accuracy %f, validation accuracy %f' % (train_accuracy, valid_accuracy))         
    print("Test accuracy: %.12f%%" % test_accuracy)
    
    return run_res


# Testing model functionality
# ---

# In[11]:

def cross_validation_for_param(cv_param_name, cv_param_values, hyperparams, 
                               valid_dataset, test_dataset, valid_labels, test_labels):

    # build and run the model for different hyperparams
    validation_accuracy_list = []
    train_accuracy_list = []
    validation_loss_list = []
    train_loss_list = []
    run_res_list = []

    print('\nRunning cross-validation for %s' % cv_param_name)
    for param_val in cv_param_values:
        hyperparams[cv_param_name] = param_val
        print('\n%s %f' % (cv_param_name, param_val))

        run_res = run_model(hyperparams, valid_dataset, test_dataset, valid_labels, test_labels)
  
        validation_accuracy_list.append(run_res[1])
        train_accuracy_list.append(run_res[2])
        validation_loss_list.append(run_res[4])
        train_loss_list.append(run_res[3])

        run_res.append(param_val)
        run_res_list.append(run_res)    

     
    # find best param_val by max validation accuracy
    best_validation_accuracy_idx = np.argmax(validation_accuracy_list)
    best_res = run_res_list[best_validation_accuracy_idx]
         
    train_accuracy, valid_accuracy, test_accuracy, train_loss, valid_loss, param_val = best_res
    
    plt.figure()
    plt.plot(cv_param_values, validation_accuracy_list, label='validation_accuracy')
    plt.plot(cv_param_values, train_accuracy_list, label='train_accuracy')
    plt.xlabel(cv_param_name)
    plt.ylabel('accuracy')
    plt.legend(loc = 'best')    
    plt.title('Accuracy %f for %s, best value %f' % (valid_accuracy, cv_param_name, param_val))
    
    plt.figure()
    plt.plot(cv_param_values, validation_loss_list, label='validation_loss')
    plt.plot(cv_param_values, train_loss_list, label='train_loss')
    plt.xlabel(cv_param_name)
    plt.ylabel('loss')
    plt.legend(loc = 'best') 
    plt.title('Loss %f for %s, best value %f' % (np.min(validation_loss_list), cv_param_name, param_val))
    
    hyperparams[cv_param_name] = param_val
    print('\nBest model %s value %f, train_accuracy %f, validation_accuracy %f, test_accuracy %f' % 
           (cv_param_name, param_val, train_accuracy, valid_accuracy, test_accuracy))
    print('train_loss, %f, valid_loss %f' % (train_loss, valid_loss))
    
    return [param_val, valid_loss, valid_accuracy]

def main(path):
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, \
            test_labels  = load_data(path)
    
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    # Test trials
    hyperparams = get_default_params()
    hyperparams['num_steps'] = 10001
    run_model(hyperparams, train_dataset, train_labels, valid_dataset,  \
                    valid_labels, test_dataset, test_labels)

if __name__ == '__main__':
    path = '/media/sf_teeth_segmentation/notMNIST.pickle'
    main(path)

