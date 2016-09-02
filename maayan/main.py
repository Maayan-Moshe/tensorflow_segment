
# coding: utf-8

'''
Deep Learning - Convolutional Neural Network
============= 
Purpose - classification of [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
''' 

#from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import computation_graph
import model_training
import tensorflow as tf
import utils
import serializer



# In[1]:
def run_all(data_path, params_path, save_to_folder=''):
                
    hyperparams = serializer.load_hyperparams(params_path) 
    data = load_data(hyperparams, data_path)
    return run_model(hyperparams, data, save_to_folder)


def run_all_from_restored(restore_from_folder, data_path, params_path, 
                          save_to_folder='', override_params = {}):
                
    hyperparams = serializer.load_hyperparams(params_path) 
    data = load_data(hyperparams, data_path)
    return run_model_from_restored(restore_from_folder, data, save_to_folder, 
                                   override_params = override_params)


# In[1]:
def run_model(hyperparams, data, save_to_folder):
    '''data: train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels]'''
    
    if save_to_folder != '':
        hyperparams['save_folder'] = save_to_folder
    utils.print_params(hyperparams)
    
    with tf.Graph().as_default(), tf.Session() as session:    
    
        # build graph flow model
        train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data
        graph = computation_graph.build_graph(hyperparams, valid_dataset, test_dataset)
        
        # run graph flow model 
        accuracy = model_training.train_model(session, graph, hyperparams, 
                                              train_dataset, train_labels, valid_labels, test_labels)
           
        # report final results
        print('------Final results-------')
        print('Final train accuracy %1.2f, validation accuracy %1.2f %%' % (accuracy[0], accuracy[1]))         
        print("Test accuracy: %1.2f%%" % accuracy[2])
        
        return accuracy


def run_model_from_restored(folder, data, save_to_folder='', override_params = {}):
    '''data: train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels]'''
    
    with tf.Graph().as_default(), tf.Session() as session:
        # restore model and create session
        train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data
        restored = serializer.restore_model(session, folder, train_dataset, valid_dataset, 
                                       test_dataset, override_params)
        if not restored:
            print('Restore failed')
            return []
            
        graph, hyperparams, last_epoch = restored
        if save_to_folder != '':
            hyperparams['save_folder'] = save_to_folder

        # run graph flow model 
        accuracy = model_training.train_model(session, graph, hyperparams, train_dataset, train_labels, 
                                              valid_labels, test_labels, start_epoch=last_epoch)

        # report final results
        print('------Final results-------')
        print('Final train accuracy %f, validation accuracy %f' % (accuracy[0], accuracy[1]))         
        print("Test accuracy: %.12f%%" % accuracy[2])


    return accuracy


def restore_model_and_stop(folder, data):
    '''data: train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels]'''
    override_params = {'num_full_epochs' : 0}
    return run_model_from_restored(folder, data, override_params=override_params)



# In[1]:
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
    
    
 # In[1]:   
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


# In[1]:
if __name__ == '__main__':
    who_am_i = 'dina' # :-)
    print('*** Hi, I am ', who_am_i)
    
    if who_am_i == 'dina':
        params_path = '/media/sf_tensorflow_segment/params/optimal_sett.json'
        data_path = '/media/sf_tensorflow_segment/data/notMNIST.pickle'
        
        restore_from_folder = '/media/sf_tensorflow_segment/save_restore/opt_save_model'
        save_to_folder = '/media/sf_tensorflow_segment/save_restore/opt_save_model1'
        restore_override_params = {}
    else:
        params_path = '/media/sf_teeth_segmentation/params/optimal_sett.json'
        data_path = '/media/sf_teeth_segmentation/data/notMNIST.pickle'
        
        restore_from_folder = ''
        save_to_folder = ''
        restore_override_params = {}        
    
    if restore_from_folder == '':
        run_all(data_path, params_path)
    else:
        run_all_from_restored(restore_from_folder, data_path, params_path, 
                          save_to_folder, override_params = restore_override_params)       

