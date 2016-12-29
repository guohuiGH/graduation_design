#coding:utf-8
'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pandas as pd
import sys
import random
sys.path.append("../script")
from convert_input import Convert
from validation import Validation
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
class LstmParam:
    def __init__(self):
        c = Convert()
        x_train, y_train, x_test, y_test = c.getDTOneHotData()
        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = pd.get_dummies(np.array(y_train),prefix="y")
        self.label_value = y_test #list format of label
        self.y_test = pd.get_dummies(np.array(y_test),prefix="y")
        '''
        To classify images using a recurrent neural network, we consider every image
        row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
        handle 28 sequences of 28 steps for every sample.
        '''

        # Parameters
        self.learning_rate = 0.01
        self.training_iters = 300
        self.batch_size = len(self.x_train)  # forest-net must total data
        self.display_step = 10

        # Network Parameters
        self.n_input = 418 # feature number
        self.n_steps = 4 # timesteps
        self.n_hidden = 128 # hidden layer num of features
        self.n_classes = 2 # MNIST total classes (0-9 digits)
        self.forget_bias = 1.0 # forget bias value

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


#generate random input
def generate_random():
    index = sorted(random.sample(range(len(x_train)), batch_size))
    batch_x = [x_train[i] for i in index]
    #a line data
    batch_y = [y_train.ix[i] for i in index]
    return np.array(batch_x), np.array(batch_y)


#train data and predict
def train(param, model):

    # Launch the graph
    with tf.Session() as sess:
        sess.run(model.init)
        step = 0
        # Keep training until reach max iterations
        while step  < param.training_iters:

            #generate train data---total data
            batch_x = param.x_train
            batch_y = param.y_train
            batch_x = batch_x.reshape((param.batch_size, param.n_steps, param.n_input))
            batch_y = np.reshape(batch_y,(-1,param.n_classes))


            fd = {model.xx:batch_x, model.y:batch_y}

            # Run optimization op (backprop)
            sess.run(model.optimizer, feed_dict=fd)
            #get the gradient or residual
            grads_wrt_input = sess.run(tf.gradients(model.cost, model.x), feed_dict=fd)
            residual = sess.run(tf.concat(1, grads_wrt_input))
            print (residual[0,:])


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
        test_data = param.x_test.reshape((len(param.x_test), param.n_steps, param.n_input))
        test_label = np.reshape(param.y_test, (-1,2))

        fd = {model.xx:test_data, model.y:test_label}
        loss = sess.run(model.cost, feed_dict=fd)

        print("{:.6f}".format(loss))
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict=fd))

        #get the predict classifier and probability value
        classifier_value = sess.run([classifier_collection], feed_dict={model.xx:test_data})

        proba_value = sess.run([proba_collection], feed_dict={model.xx:test_data})

        v = Validation()
        v.calculateF1(param.label_value, classifier_value[0])
        v.allValidation(param.laebel_value, proba_value[0][:,1])
def main():
    param = LstmParam()
    model = LstmModel(param)
    train(param, model)
if __name__=="__main__":
    main()
