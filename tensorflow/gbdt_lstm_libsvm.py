#coding:utf-8
'''
LSTM and GBDT implementation using TensorFlow and Sklearn library.
Author: guohui
email: guohui1029@foxmail.com
project:https://github.com/guohuiGH/graduation_design
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pandas as pd
import sys, random, time
from gbdt_net import ForestNet
sys.path.append("../script")
from convert_input import Convert
from validation import Validation
from collections import OrderedDict

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("../data/", one_hot=True)

class InputLibsvm:
    def __init__(self):
        c = Convert()
        self.x_train, self.y_train, self.x_test, self.y_test = c.get_libsvm_gbdt_data("recent")
        self.x_train = np.array(self.x_train)[2:,]
        self.x_test = np.array(self.x_test)[2:,]
        self.y_input = self.y_test
        self.y_train = pd.get_dummies(np.array(self.y_train),prefix="y")
        self.y_test = pd.get_dummies(np.array(self.y_test),prefix="y")
        #self.length = len(self.x_train.shape[0])

class LstmParam:
    def __init__(self,dataset):

        # Parameters
        self.learning_rate = 0.1 # lstm learning rate
        self.training_iters = 40  # iteation, or forest size
        self.batch_size = dataset.x_train[0].shape[0]  # forest-net train number, input size
        self.display_step = 10   # show size

        # Network Parameters
        self.n_input = 70 # feature number
        self.n_steps = 2 # timesteps
        self.n_hidden = 128 # hidden layer num of features
        self.n_classes = 2 # MNIST total classes (0-9 digits)
        self.forget_bias = 0.0 # forget bias value

        #ForestNet Parameters
        self.f_learning_rate = 0.1 # forestnet learning rate
        self.f_number = self.n_input # single forestnet size --> or self.n_input
        self.f_steps = self.n_steps # forestnet cluster size --> or self_n_steps
        self.f_input = 132 # input feature number
        self.thread_core = 1 # number of thread_core
        self.step = -100000

class LstmModel:
    def __init__(self, param):

        self.param = param

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.param.n_steps, self.param.n_input])
        self.y = tf.placeholder("float", [None, self.param.n_classes])
        self.xx = self.x
        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([self.param.n_hidden, self.param.n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.param.n_classes]))
        }

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        self.x = tf.transpose(self.x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        self.x = tf.reshape(self.x, [-1, self.param.n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        self.x = tf.split(0, self.param.n_steps, self.x)

        # Define a lstm cell with tensorflow
        #lstm_cell = rnn_cell.LSTMCell(self.param.n_hidden, forget_bias=self.param.forget_bias, activation=tf.nn.relu)
        lstm_cell = rnn_cell.BasicLSTMCell(self.param.n_hidden, forget_bias=self.param.forget_bias)
        lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=0.5, output_keep_prob=0.5)
        #cell = rnn_cell.MultiRNNCell([lstm_cell] * 2)
        # Get lstm cell output
        outputs, states = tf.nn.rnn(lstm_cell, self.x, dtype=tf.float32)
        #activate function
        pred = tf.matmul(outputs[-1], weights['out'] + biases['out'])


        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, self.y))
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.param.learning_rate).minimize(self.cost)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.param.learning_rate).minimize(self.cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #probability and classifer collection
        self.proba_collection = tf.nn.softmax(pred)#tf.argmax(pred,1)
        self.classifier_collection = tf.argmax(pred, 1)

        # Initializing the variables
        self.init = tf.global_variables_initializer()
        #self.init = tf.initialize_all_variables()

#generate random input and forest data
def generate_lstm_input(param,forest_net, dataset, redisual):
    #random data
    #x_train = mnist.train.images; y_train = mnist.train.labels
    #index = sorted(random.sample(range(len(x_train)), param.batch_size))
    #batch_x = [x_train[i] for i in index]
    #batch_y = [y_train[i] for i in index]
    #batch_x, batch_y = mnist.train.next_batch(param.batch_size)
    batch_x = dataset.x_train
    batch_y = dataset.y_train
    f_batch_x = forest_net.get_libsvm_train_data(batch_x, redisual, dataset.x_test)
    return np.array(f_batch_x), np.array(batch_y)
    pass


#train data and predict
def train(param, model, forest_net, dataset):

    # Launch the graph
    with tf.Session() as sess:
        sess.run(model.init)
        step = 0; redisual = list()
        # Keep training until reach max iterations
        sum_time = list()
        while step  < param.training_iters:
            s_time = int(time.time())
            #generate train data---mini batch
            #using the forest net generate hidden data : n_steps * batch_size * n_input
            batch_x, batch_y = generate_lstm_input(param, forest_net, dataset,redisual)
            #batch_x = forest_net.get_train_data(mnist.train.images, redisual, mnist.test.images)

            batch_x = np.transpose(batch_x,[1,0,2])
            #batch_y = np.reshape(batch_y,(-1,param.n_classes))

            fd = {model.xx:batch_x, model.y:batch_y}

            m_time = float(time.time())
            # Run optimization op (backprop)
            sess.run(model.optimizer, feed_dict=fd)
            #get the gradient or residual

            grad_val = np.array(sess.run(tf.gradients(model.cost, model.xx), feed_dict=fd)[0])
            m2_time = float(time.time())
            sum_time.append(m2_time-m_time)
            redisual = param.step*np.transpose(grad_val,[1,0,2])
            e_time = int(time.time())

            if step % param.display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(model.accuracy, feed_dict=fd)
                # Calculate batch loss
                loss = sess.run(model.cost, feed_dict=fd)
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
            batch_x = list(); batch_y = list()

        print("Optimization Finished!")


        #test data


        test_label = dataset.y_test#[:test_len]
        #test_data_temp = forest_net.forest_test_predict(test_d)
        test_data_temp = np.array(forest_net.get_test_data())
        test_data = sess.run(tf.transpose(test_data_temp,[1,0,2]))


        fd = {model.xx:test_data, model.y:test_label}
        loss = sess.run(model.cost, feed_dict=fd)

        print("{:.6f}".format(loss))
        print("Testing Accuracy:", \
            sess.run(model.accuracy, feed_dict=fd))

        #get the predict classifier and probability value
        classifier_value = sess.run([model.classifier_collection], feed_dict={model.xx:test_data})
        proba_value = sess.run([model.proba_collection], feed_dict={model.xx:test_data})

        #print (np.array(classifier_value).shape, np.array(proba_value).shape)

        v = Validation()
        v.calculateF1(dataset.y_input, classifier_value[0])
        v.allValidation(dataset.y_input, proba_value[0][:,1])

        v.top_accuacy(dataset.y_input, classifier_value[0], proba_value[0][:,0],[1000,10000,50000])

def main():
    dataset = InputLibsvm()
    #print (dataset.x_train.shape)
    print (dataset.x_train[0].shape)
    print (dataset.x_test[0].shape)
    param = LstmParam(dataset)
    model = LstmModel(param)
    forest_net = ForestNet(param.batch_size, param.f_number, param.f_learning_rate, param.f_steps, param.f_input, param.training_iters, param.thread_core)
    train(param, model, forest_net, dataset)

if __name__=="__main__":
    main()
