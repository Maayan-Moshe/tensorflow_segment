# -*- coding: utf-8 -*-
import time
import utils
import math
import serializer


# In[1]:

def train_model(session, graph, hyperparams, 
                train_dataset, train_labels, 
                validation_labels, test_labels, start_epoch=0):
                                
    trainer = ModelTrainer(session, graph, hyperparams, validation_labels)
    train_accuracy, validation_accuracy = trainer.train_all(train_dataset, train_labels, start_epoch)
    test_accuracy = trainer.eval_test_accuracy(test_labels)
        
    return [train_accuracy, validation_accuracy, test_accuracy]

# In[1]:

class ModelTrainer:
    
    def __init__(self, session, graph, hyperparams, validation_labels):                               
        self.session = session    
        self.graph = graph                         
        self.hyperparams = hyperparams
        self.validation_labels = validation_labels
         
# In[1]:     
    def train_all(self, train_dataset, train_labels, start_epoch=0):  
        t0 = time.time()
        
        self.__set_epoch_num_steps(train_dataset)        
        self.train_accuracy_list = []
        self.validation_accuracy_list = []
        self.train_loss_vals = []
        self.evaluation_steps = []     
     
        # Training for all epochs (full data passes)
        for epoch in range(start_epoch, self.num_full_epochs):            
            t0_epoch = time.time()
            print('\n---- Training epoch %d of %d' % (epoch, self.num_full_epochs))
            if not self.train_single_epoch(epoch, train_dataset, train_labels):
                break
            t1_epoch = time.time()
            sec = t1_epoch - t0_epoch
            print('Time for epoch - %1.2f minutes %1.3f hours' % (sec/60.0, sec/60.0/60.0))      
            
        if len(self.train_accuracy_list) < 2:
            return -1, -1
        
        # draw learning curves        
        self.visualize_learning_curves()          

        t1 = time.time()  
        sec = t1-t0
        print('\n ---Done. Total training time - %1.2f minutes %1.3f hours' % (sec/60.0, sec/60.0/60.0))        
         
        train_accuracy = self.train_accuracy_list[-1]
        validation_accuracy = self.validation_accuracy_list[-1]
        
        return train_accuracy, validation_accuracy


    def train_single_epoch(self, epoch, train_dataset, train_labels):
        num_samples = train_dataset.shape[0]
            
        # each step is optimization run on a single batch. 
        for step in range(self.num_steps_per_epoch):

            # Generate a minibatch. Train data should be randomized ahead
            batch_size = self.hyperparams['batch_size'] 
            batch_start_offset = (step * batch_size) % (num_samples - batch_size)
            batch_data = train_dataset[batch_start_offset : (batch_start_offset + batch_size)]
            batch_labels = train_labels[batch_start_offset : (batch_start_offset + batch_size)]

            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder tensor of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {self.graph.tf_train_dataset : batch_data, 
                         self.graph.tf_train_labels : batch_labels, 
                         self.graph.tf_dropout_keep_prob: self.hyperparams['dropout_keep_prob'] }
            
            # Run optimization step and get loss value
            _, train_loss = self.session.run([self.graph.optimizer, self.graph.loss], feed_dict=feed_dict)
            
            if math.isnan(train_loss):
                print('\n !!! Stopping due to divergence - train Loss is nan on step', step)
                return False

            # log train/validation accuracy evaluations every evaluation_step
            if (step % self.evaluation_step == 0):
                print("--Train Minibatch local step %d epoch %d:" % (step, epoch))
                
                # all evaluations should be deterministic - without dropout
                train_accuracy = self.__eval_accuracy(self.graph.train_prediction, batch_labels, feed_dict)
                valid_accuracy = self.__eval_accuracy(self.graph.valid_prediction, self.validation_labels)
  
                global_step = epoch * self.num_steps_per_epoch + step
                self.evaluation_steps.append(global_step)
                self.train_accuracy_list.append(train_accuracy)
                self.train_loss_vals.append(train_loss)
                self.validation_accuracy_list.append(valid_accuracy)
                
                print("Train loss %1.4f, Train/Validation accuracy (%1.2f , %1.2f) %%" %
                      (train_loss, train_accuracy, valid_accuracy))
                      
        # saving the model at the end of each epoch   
        global_step = epoch * self.num_steps_per_epoch + step
        serializer.save_model(self.session, self.graph.saver, self.hyperparams, global_step)
        
        return True
                         
# In[1]:          
    def eval_test_accuracy(self, test_labels):
        return self.__eval_accuracy(self.graph.test_prediction, test_labels)
        
        
    def __eval_accuracy(self, tf_prediction, labels, feed_dict = {}):
        feed_dict[self.graph.tf_dropout_keep_prob] = 1.0
        prediction = tf_prediction.eval(feed_dict = feed_dict)
        accuracy = utils.accuracy(prediction, labels)
        return accuracy        
        
# In[1]:    
    def visualize_learning_curves(self):
        import matplotlib.pyplot as plt
        
        if len(self.evaluation_steps) > 1:    
            plt.figure()
            plt.plot(self.evaluation_steps, self.train_loss_vals, label='train_loss')
            plt.legend(loc = 'best')
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.title('Train loss - final  train_loss %f' % self.train_loss_vals[-1])
        
            plt.figure()
            plt.plot(self.evaluation_steps, self.train_accuracy_list, label='train_accuracy')
            plt.plot(self.evaluation_steps, self.validation_accuracy_list, label='validation_utils.accuracy')
            plt.xlabel('step')
            plt.ylabel('utils.accuracy')
            plt.legend(loc = 'best')
            plt.title('Train/validation utils.accuracy, final train_accuracy %f valid_accuracy %f' % 
                                    (self.train_accuracy_list[-1], self.validation_accuracy_list[-1]))
                                    
# In[1]:                                      
    def __set_epoch_num_steps(self, train_dataset):     
        # single epoch means a full pass over the whole data
        # in a single iteration step batch_size of data is processed
        self.num_steps_per_epoch =  utils.calc_num_steps_per_epoch(train_dataset, self.hyperparams)
        
        if 'num_full_epochs' in self.hyperparams:
            self.num_full_epochs = self.hyperparams['num_full_epochs']  
            
        if 'num_steps' in self.hyperparams:
            # num_steps is optional parameter. 
            # If defined bigger than num_steps_per_epoch, num_full_epochs is recalculated.
            # If defined smaller than num_steps_per_epoch, num_steps_per_epoch is recalculated.
            num_steps = self.hyperparams['num_steps']    
            if num_steps > self.num_steps_per_epoch:
                self.num_full_epochs = num_steps // self.num_steps_per_epoch 
            else:
                self.num_steps_per_epoch = num_steps
                self.num_full_epochs = 1
                
        self.evaluation_step = self.num_steps_per_epoch // self.hyperparams['num_evaluations_per_epoch']
        print('num_full_epochs %d, num_steps_per_epoch %d' % (self.num_full_epochs, self.num_steps_per_epoch))