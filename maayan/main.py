
# coding: utf-8

'''
Deep Learning - Convolutional Neural Network
============= 
Purpose - classification of [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
''' 

from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import inference_graph
import tensorflow as tf
import utils




def run_all(data_path, params_path):
                
    hyperparams = load_hyperparams_from_json(params_path) 
    data = load_data(hyperparams, data_path)
    
    return run_model(hyperparams, data)



def run_model(hyperparams, data):
    
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data
    utils.print_params(hyperparams)
    
    with tf.Graph().as_default(), tf.Session() as session:    
    
        # build graph flow model
        graph_elements = inference_graph.build_graph(hyperparams, valid_dataset, test_dataset)
        # run graph flow model 
        from model_training import train_model
        run_res = train_model(session, graph_elements, hyperparams, train_dataset, train_labels, \
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
    
    
    
def load_data(params, dataset_pickle_file = 'notMNIST.pickle'):

    with open(dataset_pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        
        train_dataset, train_labels = utils.reformat(train_dataset, train_labels, params)
        valid_dataset, valid_labels = utils.reformat(valid_dataset, valid_labels, params)
        test_dataset, test_labels = utils.reformat(test_dataset, test_labels, params)
        
        print('---- Input data shapes: -----')
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        
        return [train_dataset, train_labels, valid_dataset, valid_labels, valid_dataset, valid_labels] 


def load_hyperparams_from_json(path):
    
    import json
    params = None
    with open(path) as in_f:
        params = json.load(in_f)
    return params


if __name__ == '__main__':
    params_path = '/media/sf_tensorflow_segment/optimal_sett.json'
    path = '/media/sf_tensorflow_segment/data/notMNIST.pickle'
    run_all(path, params_path)

