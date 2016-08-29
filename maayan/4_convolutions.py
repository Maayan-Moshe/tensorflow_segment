
# coding: utf-8

'''
Deep Learning - Convolutional Neural Network
============= 
Purpose - classification of [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
''' 

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
import utils


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

'''
Let's build a small network with two convolutional layers, 
followed by one fully connected layer. 
Convolutional networks are more expensive computationally, 
so we'll limit its depth and number of fully connected nodes.
'''

def build_model(hyperparams, valid_dataset, test_dataset):
    '''
    Building computational graph model functionality
    '''    
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

def run_model(hyperparams, train_dataset, train_labels, valid_dataset, \
                valid_labels, test_dataset, test_labels):
    
    # build graph flow model
    graph_elements = build_model(hyperparams, valid_dataset, test_dataset)
    # run graph flow model 
    from model_training import train_model
    run_res = train_model(graph_elements, hyperparams, train_dataset, train_labels, \
                                            valid_labels, test_labels)       
    train_accuracy, valid_accuracy, test_accuracy, train_loss = run_res
       
    # report final results
    print('------Final results-------')
    print('Final train loss %f' % train_loss)
    print('Final train utils.accuracy %f, validation utils.accuracy %f' % (train_accuracy, valid_accuracy))         
    print("Test utils.accuracy: %.12f%%" % test_accuracy)
    
    return run_res

def cross_validation_for_param(cv_param_name, cv_param_values, hyperparams, 
                               valid_dataset, test_dataset, valid_labels, test_labels):
    '''Testing model functionality'''
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

     
    # find best param_val by max validation utils.accuracy
    best_validation_accuracy_idx = np.argmax(validation_accuracy_list)
    best_res = run_res_list[best_validation_accuracy_idx]
         
    train_accuracy, valid_accuracy, test_accuracy, train_loss, valid_loss, param_val = best_res
    
    plt.figure()
    plt.plot(cv_param_values, validation_accuracy_list, label='validation_utils.accuracy')
    plt.plot(cv_param_values, train_accuracy_list, label='train_accuracy')
    plt.xlabel(cv_param_name)
    plt.ylabel('utils.accuracy')
    plt.legend(loc = 'best')    
    plt.title('utils.accuracy %f for %s, best value %f' % (valid_accuracy, cv_param_name, param_val))
    
    plt.figure()
    plt.plot(cv_param_values, validation_loss_list, label='validation_loss')
    plt.plot(cv_param_values, train_loss_list, label='train_loss')
    plt.xlabel(cv_param_name)
    plt.ylabel('loss')
    plt.legend(loc = 'best') 
    plt.title('Loss %f for %s, best value %f' % (np.min(validation_loss_list), cv_param_name, param_val))
    
    hyperparams[cv_param_name] = param_val
    print('\nBest model %s value %f, train_accuracy %f, validation_utils.accuracy %f, test_accuracy %f' % 
           (cv_param_name, param_val, train_accuracy, valid_accuracy, test_accuracy))
    print('train_loss, %f, valid_loss %f' % (train_loss, valid_loss))
    
    return [param_val, valid_loss, valid_accuracy]
    
def load_from_json(path):
    
    import json
    params = None
    with open(path) as in_f:
        params = json.load(in_f)
    return params

def main(path, params_path):
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, \
            test_labels  = load_data(path)
            
    params = load_from_json(params_path)
    
    train_dataset, train_labels = utils.reformat(train_dataset, train_labels, params)
    valid_dataset, valid_labels = utils.reformat(valid_dataset, valid_labels, params)
    test_dataset, test_labels = utils.reformat(test_dataset, test_labels, params)
    
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    # Test trials
    run_model(params, train_dataset, train_labels, valid_dataset,  \
                    valid_labels, test_dataset, test_labels)

if __name__ == '__main__':
    params_path = '/media/sf_teeth_segmentation/tensorflow_segment/default_sett.json'
    path = '/media/sf_teeth_segmentation/notMNIST.pickle'
    main(path, params_path)

