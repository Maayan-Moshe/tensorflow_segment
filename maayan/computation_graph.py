# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:08:18 2016

"""

import tensorflow as tf

def build_graph(hyperparams, valid_dataset, test_dataset):
    graph = ComputationalGraph(hyperparams)
    graph.build_graph(valid_dataset, test_dataset)
    return graph

class ComputationalGraph:
    '''
    Computational graph model functionality
    Assumming that session and default graph exist
    '''    
    
    def __init__(self, hyperparams):  
        self.hyperparams = hyperparams
        # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
        batch_size = hyperparams['batch_size']        
        data_shape = (batch_size, hyperparams['image_height'], 
                      hyperparams['image_width'], hyperparams['num_channels'])
        self.tf_train_dataset = tf.placeholder(tf.float32, shape=data_shape)
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, hyperparams['num_labels'] ))
        self.tf_dropout_keep_prob = tf.placeholder(tf.float32)  
        
    def build_graph(self, valid_dataset, test_dataset):
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)             
        
        # add layers weight/biases variables
        self.layers_objects = self.build_network_layers()     
    
        # Optimisation --------------------------------------   
        self.train_logits = self.forward_prop(self.tf_train_dataset)
        self.add_optimizer(self.train_logits, self.tf_train_labels)
    
        # Predictions  probabilities ------------------------
        self.train_prediction = self.predict(self.train_logits)
        self.valid_prediction = self.predict(self.forward_prop(tf_valid_dataset)) 
        self.test_prediction = self.predict(self.forward_prop(tf_test_dataset))   

        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()           
        
        # Properly initialize all graph variables in existing session.
        tf.initialize_all_variables().run()      
        print("\nSession Initialized")  
              
    def build_network_layers(self):
        self.layers_objects = []
           
        layers_info = self.hyperparams['layers_info']
        num_layers = len(layers_info) # excluding input layer
        assert(num_layers > 0)
       
        activation_func = self.hyperparams['activation_func']
        total_params_count = 0
        
        data_shape = [self.hyperparams['image_height'], self.hyperparams['image_width'], 
                      self.hyperparams['num_channels']]
                      
        from layer_modules import create_layer_module  
        for layer in range(num_layers):
            layer_info = layers_info[layer]       
            if layer == (num_layers-1): # last layer
                activation_func = ''
            
            layer_obj = create_layer_module(layer_info, data_shape, activation_func)      
            if (layer_obj == None):
                continue
            
            layer_obj.build(bias_init_val=self.hyperparams['bias_init_val'], 
                            weights_stddev=self.hyperparams['weight_stddev'])
            self.layers_objects.append(layer_obj)     
            data_shape = layer_obj.out_data_shape
            
            params_count = layer_obj.calc_params_count()
            print('Layer % d %s : input_data_shape %s out_data_shape %s - total parameters = %d' % 
                  (layer+1, layer_info['type'], layer_obj.data_shape, layer_obj.out_data_shape, params_count))
            total_params_count += params_count
        
        print('Total layers %d' % num_layers)
        print('Total parameters number = %d' % total_params_count)
        return self.layers_objects
        
    def add_optimizer(self, train_logits, tf_train_labels):
        learning_rate = self.hyperparams['learning_rate']
        learning_decay_steps = self.hyperparams['learning_decay_steps']
        learning_decay_rate = self.hyperparams['learning_decay_rate']
        optimizer_alg_name = self.hyperparams['optimizer_alg_name']
        
        # define loss function as cross-entropy between softmax of last layer predicted scores and labels.
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))   
        
        # add variable for holding interations steps
        self.global_step = tf.Variable(0)  # count the number of running steps taken.
        
        # Learning rate decay - for GradientDescentOptimizer or MomentumOptimizer
        # reference http://sebastianruder.com/optimizing-gradient-descent/
        if (learning_decay_rate < 1.0 and 
            (optimizer_alg_name == 'GradientDescentOptimizer' or optimizer_alg_name == 'MomentumOptimizer')) :
            learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 
                                                       learning_decay_steps, 
                                                       learning_decay_rate, staircase=True)  
            
        if optimizer_alg_name == 'GradientDescentOptimizer':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer_alg_name == 'MomentumOptimizer':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, self.hyperparams['momentum_term'])        
        elif optimizer_alg_name == 'AdagradOptimizer':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer_alg_name == 'AdamOptimizer':
            self.optimizer = tf.train.AdamOptimizer(1e-4)
        else:
            self.hyperparams['optimizer_alg_name'] = 'GradientDescentOptimizer'
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    
        self.optimizer = self.optimizer.minimize(self.loss, global_step=self.global_step) 
            
    def forward_prop(self, dataset):   
        num_layers = len(self.layers_objects)
        assert(num_layers > 0)
                           
        activations = dataset        
        for layers_obj in self.layers_objects:  
            activations = layers_obj.eval(activations, self.tf_dropout_keep_prob)
         
        return activations    

    def predict(self, logits):
        prediction = tf.nn.softmax(logits)
        return prediction