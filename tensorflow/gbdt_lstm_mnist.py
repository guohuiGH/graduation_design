#coding:utf-8
'''
LSTM and GBDT implementation using TensorFlow and Sklearn library.
Author: guohui
email: guohui1029@foxmail.com
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pandas as pd
import sys, random
from gbdt_net import ForestNet
sys.path.append("../script")
from convert_input import Convert
from validation import Validation

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/", one_hot=True)


class LstmParam:
    def __init__(self):

        '''
        To classify images using a recurrent neural network, we consider every image
        row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
        handle 28 sequences of 28 steps for every sample.
        '''

        # Parameters
        self.learning_rate = 0.01 # lstm learning rate
        self.training_iters = 100  # iteation, or forest size
        self.batch_size = 10000  # forest-net train number, input size
        self.display_step = 10   # show size

        # Network Parameters
        self.n_input = 10 # feature number
        self.n_steps = 28 # timesteps
        self.n_hidden = 128 # hidden layer num of features
        self.n_classes = 10 # MNIST total classes (0-9 digits)
        self.forget_bias = 1.0 # forget bias value

        #ForestNet Parameters
        self.f_learning_rate = 0.1 # forestnet learning rate
        self.f_number = 10 # single forestnet size --> or self.n_input
        self.f_steps = 28 # forestnet cluster size --> or self_n_steps
        self.f_input = 28 # input feature number

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
        lstm_cell = rnn_cell.BasicLSTMCell(self.param.n_hidden, forget_bias=self.param.forget_bias)
        #lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=0.5, output_keep_prob=0.5)
        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, self.x, dtype=tf.float32)
        #activate function
        pred = tf.matmul(outputs[-1], weights['out'] + biases['out'])


        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, self.y))
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.param.learning_rate).minimize(self.cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #probability and classifer collection
        self.proba_collection = tf.nn.softmax(pred)#tf.argmax(pred,1)
        self.classifier_collection = tf.argmax(pred, 1)

        # Initializing the variables
        self.init = tf.global_variables_initializer()


#generate random input and forest data
def generate_lstm_input(param,forest_net, redisual):
    #random data
    #index = sorted(random.sample(range(len(param.x_train)), param.batch_size))
    #batch_x = [param.x_train[i] for i in index]
    #batch_y = [param.y_train.ix[i] for i in index]
    #batch_x, batch_y = mnist.train.next_batch(param.batch_size)
    batch_x = mnist.train.images[:param.batch_size]
    batch_y = mnist.train.labels[:param.batch_size]
    f_batch_x = forest_net.generate_forest(batch_x, redisual)
    return np.array(f_batch_x), np.array(batch_y)


#train data and predict
def train(param, model, forest_net):

    # Launch the graph
    with tf.Session() as sess:
        sess.run(model.init)
        step = 0; redisual = list()
        # Keep training until reach max iterations
        while step  < param.training_iters:

            #generate train data---mini batch
            #using the forest net generate hidden data : n_steps * batch_size * n_input
            batch_x, batch_y = generate_lstm_input(param, forest_net, redisual)

            batch_x = sess.run(tf.transpose(batch_x,[1,0,2]))
            #batch_y = np.reshape(batch_y,(-1,param.n_classes))

            fd = {model.xx:batch_x, model.y:batch_y}

            # Run optimization op (backprop)
            sess.run(model.optimizer, feed_dict=fd)
            #get the gradient or residual
            grad_val = sess.run(tf.gradients(model.cost, model.xx), feed_dict=fd)[0]

            redisual = -100000*np.array(sess.run(tf.transpose(grad_val,[1,0,2])))

            if step % param.display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(model.accuracy, feed_dict=fd)
                # Calculate batch loss
                loss = sess.run(model.cost, feed_dict=fd)
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1

        print("Optimization Finished!")


        #test data
        test_len = 128
        test_d = mnist.test.images[:test_len]
        test_label = mnist.test.labels[:test_len]
        test_data_temp = forest_net.forest_test_predict(test_d)
        test_data = sess.run(tf.transpose(test_data_temp,[1,0,2]))
        #test_label = np.reshape(test_label, (-1,2))

        fd = {model.xx:test_data, model.y:test_label}
        loss = sess.run(model.cost, feed_dict=fd)

        print("{:.6f}".format(loss))
        print("Testing Accuracy:", \
            sess.run(model.accuracy, feed_dict=fd))

        #get the predict classifier and probability value
        classifier_value = sess.run([model.classifier_collection], feed_dict={model.xx:test_data})
        proba_value = sess.run([model.proba_collection], feed_dict={model.xx:test_data})

        #v = Validation()
        #v.calculateF1(param.label_value, classifier_value[0])
        #v.allValidation(param.laebel_value, proba_value[0][:,1])

def main():
    param = LstmParam()
    model = LstmModel(param)
    forest_net = ForestNet(param.batch_size, param.f_number, param.f_learning_rate, param.f_steps, param.f_input, param.training_iters)
    train(param, model, forest_net)

if __name__=="__main__":
    main()
