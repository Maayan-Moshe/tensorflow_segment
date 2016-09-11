# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:30:16 2016

"""
import json
import os
import utils
import computation_graph
import tensorflow as tf

def restore_model(session, folder, train_data, validation_data, test_data, override_params = {}):
                      
    print('Restoring model from %s' % folder)
    
    if not os.path.exists(folder):
        print('Error: Folder does not exist', folder)
        return []
    
    # load saved parameters and override if needed
    params_path = folder + '/hyperparams.json' 
    hyperparams = load_hyperparams(params_path)
    last_num_steps_per_epoch = utils.calc_num_steps_per_epoch(train_data, hyperparams)
    for param in override_params:
        if param in hyperparams:
            hyperparams[param] = override_params[param]
    utils.print_params(hyperparams)

    # Add ops to save and restore all the variables.
    graph = computation_graph.build_graph(hyperparams, validation_data, test_data)   
        
    # Restore variables from disk. 
    model_path = tf.train.latest_checkpoint(folder)
    print('Restoring model %s' % model_path)
    saver = graph.saver
    saver.restore(session, model_path)
    print("Model restored.")

    global_step = graph.global_step
    last_step = global_step.eval()
    last_epoch = last_step // last_num_steps_per_epoch   
    print('Restored global_step %d last_epoch %d' % (last_step, last_epoch))
    
    return [graph, hyperparams, last_epoch]

def save_model(session, saver, hyperparams, global_step=0):
    print('Saving model and parameters')
    save_folder = hyperparams['save_folder']
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # saving the model at the end of each epoch    
    model_path = save_folder + '/' + hyperparams['save_model_file'] 
    saver.save(session, model_path, global_step = global_step)
        
    params_path = save_folder + '/' + hyperparams['save_params_file']    
    save_params(hyperparams, params_path)
 
def load_hyperparams(path):
    params = None
    with open(path) as in_f:
        params = json.load(in_f)
    return params
       
def save_params(hyperparams, save_path):
    f = open(save_path, 'w')
    json.dump(hyperparams, f)
    f.close()    