# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:08:18 2016

@author: maayan
"""

import tensorflow as tf


def build_graph(hyperparams, valid_dataset, test_dataset):
    '''
    Building computational graph model functionality
    '''    
    batch_size = hyperparams['batch_size']
    image_size = hyperparams['image_size']
    num_channels = hyperparams['num_channels']
    num_labels = hyperparams['num_labels'] 
      
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
    graph_elements['layers_objects'] = layers_objects
    graph_elements['tf_train_dataset'] = tf_train_dataset
    graph_elements['tf_train_labels'] = tf_train_labels
    graph_elements['optimizer'] = optimizer
    graph_elements['loss'] = loss
    graph_elements['train_prediction'] = train_prediction
    graph_elements['valid_prediction'] = valid_prediction
    graph_elements['test_prediction'] = test_prediction
    graph_elements['tf_dropout_keep_prob'] = tf_dropout_keep_prob
    
    # Properly initialize all graph variables in current session.
    tf.initialize_all_variables().run()      
    print("\nSession Initialized")  

    return graph_elements
    
    
    
def build_network_layers(hyperparams):
    layers_objects = []
       
    layers_info = hyperparams['layers_info']
    num_layers = len(layers_info) # excluding input layer
    assert(num_layers > 0)
   
    activation_func = hyperparams['activation_func']
    total_params_count = 0
    
    data_shape = [hyperparams['image_size'], hyperparams['image_size'], hyperparams['num_channels']]
    from layer_modules import create_layer_module  
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
        activations = layers_obj.eval(activations, dropout_keep_prob, dbg_log_fn=dbg_log_fn)
     
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
