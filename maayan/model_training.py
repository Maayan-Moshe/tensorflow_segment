# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import utils
import math

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
                train_accuracy = utils.accuracy(train_predictions, batch_labels)                
             
                valid_predictions = valid_prediction.eval(feed_dict = {tf_dropout_keep_prob:1.0})
                valid_accuracy = utils.accuracy(valid_predictions, valid_labels)
     
                plot_steps.append(step)
                train_accuracy_list.append(train_accuracy)
                train_loss_vals.append(train_loss)
                validation_accuracy_list.append(valid_accuracy)
                
                t_eval1 = time.clock()
                print("Train loss %1.4f, Train/Validation utils.accuracy (%1.2f , %1.2f) percent, eval_time %1.3f min" %
                      (train_loss, train_accuracy, valid_accuracy, (t_eval1-t_eval0)/60.0))
                
                test_accuracy = utils.accuracy(test_prediction.eval(feed_dict = {tf_dropout_keep_prob:1.0}), test_labels)
                t_eval2 = time.clock()
                print('Test utils.accuracy %1.2f, eval_time %1.3f min' % (test_accuracy, (t_eval2-t_eval1)/60.0))


        # draw learning curves        
        visualize_learning_curves(plot_steps, train_loss_vals, train_accuracy_list, validation_accuracy_list)       
        
        # calc test utils.accuracy
        test_accuracy = utils.accuracy(test_prediction.eval(feed_dict = {tf_dropout_keep_prob:1.0}), test_labels)
         
    t1 = time.clock()
    sec = t1-t0
    print('Total run time %s seconds = %1.2f minutes %1.3f hours' % (sec, sec/60.0, sec/60.0/60.0))
    
    return [train_accuracy_list[-1],  validation_accuracy_list[-1], test_accuracy, train_loss_vals[-1]]
    
def visualize_learning_curves(plot_steps, train_loss_vals, train_accuracy_list, validation_accuracy_list):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(plot_steps, train_loss_vals, label='train_loss')
    plt.legend(loc = 'best')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Train loss - final  train_loss %f' % train_loss_vals[-1])

    plt.figure()
    plt.plot(plot_steps, train_accuracy_list, label='train_accuracy')
    plt.plot(plot_steps, validation_accuracy_list, label='validation_utils.accuracy')
    plt.xlabel('step')
    plt.ylabel('utils.accuracy')
    plt.legend(loc = 'best')
    plt.title('Train/validation utils.accuracy, final train_accuracy %f valid_accuracy %f' % 
                            (train_accuracy_list[-1], validation_accuracy_list[-1]))