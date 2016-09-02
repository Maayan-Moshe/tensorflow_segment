# -*- coding: utf-8 -*-

import utils
import tensorflow as tf


# In[1]:
def create_layer_module(layer_info, data_shape, activation_func):
    ''' class factory '''
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
 
 
 # In[1]:
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
        self.weights = utils.weight_variable(self.weights_shape, stddev=weights_stddev)
        self.biases = utils.bias_variable(self.depth, init_val=bias_init_val)   
        
    def eval(self, data, dropout_keep_prob=1.0): 
        return data
    
    def calc_params_count(self):
        return 0
    
    def calc_num_input_nodes(self):
        return 0
       
       
# In[1]:   
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
        logits = utils.conv2d(data, self.weights, stride=self.stride , padding=self.padding) + self.biases
        activations = utils.activate(logits, self.activation_func)   
        return activations
    
    def calc_params_count(self):
        prev_layer_depth = self.data_shape[-1] 
        filter_size = self.filter_size * self.filter_size * prev_layer_depth
        weights_size = filter_size * self.depth
        bias_size = self.depth
        return weights_size + bias_size
    
    def __calc_out_data_shape(self):
        h, w, depth = self.data_shape
        out_h = utils.calc_filtered_image_size(h, self.filter_size, self.stride, self.padding)
        out_w = utils.calc_filtered_image_size(w, self.filter_size, self.stride, self.padding)
        return [out_h, out_w, self.depth]  


# In[1]:
class fc_layer(layer_module_base):
    def __init__(self, params, data_shape, activation_func = 'relu'):
        layer_module_base.__init__(self, params, data_shape, activation_func)
        self.depth = self.params['depth']
        self.data_shape = [utils.calc_flat_size(self.data_shape)]
        prev_layer_depth = self.data_shape[-1] 
        self.weights_shape = [prev_layer_depth, self.depth] 
        self.out_data_shape = self.__calc_out_data_shape()
        
    def eval(self, data, dropout_keep_prob=1.0): 
        data = self.__flatten_data_shape(data)     
        logits = tf.matmul(data, self.weights) + self.biases
        if self.activation_func == '':
            return logits
        activations = utils.activate(logits, self.activation_func)  
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
        size = utils.calc_flat_size(shape[1:])
        data = tf.reshape(data, [batch, size])
        return data
 
 
# In[1]: 
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
            pooled_data = utils.max_pool(data, block_size=self.filter_size , stride=self.stride, padding=self.padding) 
        else:
            pooled_data = utils.avg_pool(data, block_size=self.filter_size , stride=self.stride, padding=self.padding) 
        return pooled_data
 
    def __calc_out_data_shape(self):
        h, w, depth = self.data_shape
        out_h = utils.calc_filtered_image_size(h, self.filter_size, self.stride, self.padding)
        out_w = utils.calc_filtered_image_size(w, self.filter_size, self.stride, self.padding)
        return [out_h, out_w, depth]            
        

# In[1]:    
class inception_module(layer_module_base):
    def __init__(self, params, data_shape, activation_func = 'relu'):
        ''' params is a dictionary of the following structure:
        {
            'type':'inception',
             '1x1':<depth>, 
             '3x3_reduced':<depth>, '3x3':<depth>, 
             '5x5_reduced':<depth>, '5x5':<depth>, 
             'pool_reduced':<depth>
        } '''   
        layer_module_base.__init__(self, params, data_shape, activation_func)  
        self.prev_layer_depth = self.data_shape[-1] 
        self.out_data_shape = self.calc_out_data_shape()
        self.stride = 1
        self.padding = 'SAME'
     
    def build(self, bias_init_val=0, weights_stddev=-1):
        self.weights_1x1 = utils.weight_variable([1, 1, self.prev_layer_depth, self.params['1x1']], stddev=weights_stddev)
        self.biases_1x1 = utils.bias_variable(self.params['1x1'], init_val=bias_init_val)

        self.weights_3x3_reduced = utils.weight_variable([1, 1, self.prev_layer_depth, self.params['3x3_reduced']], stddev=weights_stddev)
        self.biases_3x3_reduced = utils.bias_variable(self.params['3x3_reduced'], init_val=bias_init_val)
        self.weights_3x3 = utils.weight_variable([3, 3, self.params['3x3_reduced'], self.params['3x3']], stddev=weights_stddev)
        self.biases_3x3 = utils.bias_variable(self.params['3x3'], init_val=bias_init_val)

        self.weights_5x5_reduced = utils.weight_variable([1, 1, self.prev_layer_depth, self.params['5x5_reduced']], stddev=weights_stddev)
        self.biases_5x5_reduced = utils.bias_variable(self.params['5x5_reduced'], init_val=bias_init_val)
        self.weights_5x5 = utils.weight_variable([5, 5, self.params['5x5_reduced'], self.params['5x5']], stddev=weights_stddev)
        self.biases_5x5 = utils.bias_variable(self.params['5x5'], init_val=bias_init_val)    

        self.weights_pool_reduced = utils.weight_variable([1, 1, self.prev_layer_depth, self.params['pool_reduced']], stddev=weights_stddev)
        self.biasespool_reduced= utils.bias_variable(self.params['pool_reduced'], init_val=bias_init_val)  
        
        
    def eval(self, data, dropout_keep_prob=1.0):
        activations_1x1 = self.calc_activations(data, self.weights_1x1, self.biases_1x1)
        
        activations_3x3_reduced = self.calc_activations(data, self.weights_3x3_reduced, self.biases_3x3_reduced)
        activations_3x3 = self.calc_activations(activations_3x3_reduced, self.weights_3x3, self.biases_3x3)
        
        activations_5x5_reduced = self.calc_activations(data, self.weights_5x5_reduced, self.biases_5x5_reduced)
        activations_5x5 = self.calc_activations(activations_5x5_reduced, self.weights_5x5, self.biases_5x5)        
        
        activations_max_pool_3x3 = utils.max_pool(data, block_size=3, stride=1, padding='SAME')  
        activations_pool_reduced = self.calc_activations(activations_max_pool_3x3, self.weights_pool_reduced, self.biasespool_reduced)
        
        depth_concat = tf.concat(3, [activations_3x3, activations_1x1, activations_5x5, activations_pool_reduced])
        return depth_concat
    
    def calc_activations(self, data, weights, biases):
        logits = utils.conv2d(data, weights, stride=self.stride, padding=self.padding) + biases
        activations = utils.activate(logits, self.activation_func)
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