# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import utils
import math

def train_model(graph_elements, hyperparams, train_dataset, train_labels, \
                            valid_labels, test_labels):
                                
    trainer = ModelTrainer(graph_elements, hyperparams, valid_labels)
    train_accuracy,  validation_accuracy, train_loss = trainer.train(train_dataset, train_labels)
    test_accuracy = trainer.test(test_labels)
    return train_accuracy,  validation_accuracy, test_accuracy, train_loss

class ModelTrainer:
    
    def __init__(self, graph_elements, hyperparams, valid_labels):
                                
        self.graph = graph_elements['graph']
        self.tf_train_dataset = graph_elements['tf_train_dataset']
        self.tf_train_labels = graph_elements['tf_train_labels']
        self.optimizer = graph_elements['optimizer']
        self.loss = graph_elements['loss']
        self.train_prediction = graph_elements['train_prediction']
        self.valid_prediction = graph_elements['valid_prediction']
        self.test_prediction = graph_elements['test_prediction']
        self.tf_dropout_keep_prob = graph_elements['tf_dropout_keep_prob']    
        
        self.max_num_samples = hyperparams['max_num_samples']    
             
        self.batch_size = hyperparams['batch_size']
        if 'num_full_epochs' in hyperparams:
            self.num_full_epochs = hyperparams['num_full_epochs']
            self.num_steps = self.num_full_epochs * (self.num_samples // self.batch_size)
        else:
            self.num_steps = hyperparams['num_steps'] # number of runs - minimization iterations
            self.num_full_epochs = self.num_steps / float(self.batch_size)
           
        self.log_step = self.num_steps // hyperparams['num_logs']
        print('log_step', self.log_step)
        
        self.dropout_keep_prob = hyperparams['dropout_keep_prob']
        
        self.valid_labels = valid_labels
        
    def train(self, train_dataset, train_labels):
        
        num_samples = train_labels.shape[0]
        if self.max_num_samples > 0:
            num_samples = min(self.max_num_samples, num_samples)
            
        num_steps_per_epoch = num_samples / self.batch_size
        
        t0 = time.clock()
        
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            print("\nSession Initialized, running total %d steps %1.2f epochs" % (self.num_steps, self.num_full_epochs))
            
            train_accuracy_list = []
            validation_accuracy_list = []
            train_loss_vals = []
            plot_steps = []
                            
            # each step is optimization run on a single batch. 
            # num_steps *  batch_size can be more than num_samples to sloe multiple runs on full data
            for step in range(self.num_steps):
    
                # Generate a minibatch of already randomized training data
                sample_start_offset = (step * self.batch_size) % (num_samples - self.batch_size)
                batch_data = train_dataset[sample_start_offset : (sample_start_offset + self.batch_size), :, :, :]
                batch_labels = train_labels[sample_start_offset : (sample_start_offset + self.batch_size), :]
    
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : \
                        batch_labels, self.tf_dropout_keep_prob: self.dropout_keep_prob}
                _, train_loss = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                if math.isnan(train_loss):
                    print('Breaking: Model divergence - train Loss is nan on step', step)
                    break
    
                # log evaluations every log_step
                if (step % self.log_step == 0):
                    t_eval0 = time.clock()
                    print("--Train Minibatch step %d epoch %1.2f, time from start %1.3f min:" % 
                          (step, step/num_steps_per_epoch, ((t_eval0-t0)/60.0)))
                    
                    # all evaluations should be deterministic - without dropout
                    feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : \
                                        batch_labels, self.tf_dropout_keep_prob: 1.0}
                    train_predictions = self.train_prediction.eval(feed_dict=feed_dict)
                    train_accuracy = utils.accuracy(train_predictions, batch_labels)                
                 
                    valid_predictions = self.valid_prediction.eval(feed_dict = {self.tf_dropout_keep_prob:1.0})
                    valid_accuracy = utils.accuracy(valid_predictions, self.valid_labels)
         
                    plot_steps.append(step)
                    train_accuracy_list.append(train_accuracy)
                    train_loss_vals.append(train_loss)
                    validation_accuracy_list.append(valid_accuracy)
                    
                    t_eval1 = time.clock()
                    print("Train loss %1.4f, Train/Validation utils.accuracy (%1.2f , %1.2f) percent, eval_time %1.3f min" %
                          (train_loss, train_accuracy, valid_accuracy, (t_eval1-t_eval0)/60.0))
                    
        # draw learning curves        
        visualize_learning_curves(plot_steps, train_loss_vals, train_accuracy_list, validation_accuracy_list)       
        
        t1 = time.clock()
        sec = t1-t0
        print('Total run time %s seconds = %1.2f minutes %1.3f hours' % (sec, sec/60.0, sec/60.0/60.0))
        
        return train_accuracy_list[-1],  validation_accuracy_list[-1], train_loss_vals[-1]
        
    def test(self):
        '''calc test utils.accuracy'''
        
        test_accuracy = utils.accuracy(self.test_prediction.eval(feed_dict = {self.tf_dropout_keep_prob:1.0}), self.test_labels)
        return test_accuracy
    
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