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
sys.path.append("../decision-tree")
from gbdt import GBDT
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
g = GBDT()
x_train, y_train, x_test, y_test = g.get_leaf()
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = pd.get_dummies(np.array(y_train),prefix="y")
input_value = y_test
y_test = pd.get_dummies(np.array(y_test),prefix="y")
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.01
training_iters = 300
batch_size = 17000
display_step = 10

# Network Parameters
#n_input = len(x_train[0]) # MNIST data input (img shape: 28*28)
n_input = len(x_train[0])/4

n_steps = 4 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'], states, outputs

pred, state, outputs = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#probability and classifer collection
proba_collection = tf.nn.softmax(pred)#tf.argmax(pred,1)
classifier_collection = tf.argmax(pred, 1)

# Initializing the variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()


#generate random input
def generate_random():
    index = sorted(random.sample(range(len(x_train)), batch_size))
    batch_x = [x_train[i] for i in index]
    #a line data
    batch_y = [y_train.ix[i] for i in index]
    return np.array(batch_x), np.array(batch_y)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    start = 0; end = start + batch_size
    # Keep training until reach max iterations
    while step  < training_iters:
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        #batch_x = x_train[start:end]
        #batch_y = y_train[start:end]

        batch_x, batch_y = generate_random()
        #batch_x = x_train; batch_y = y_train
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_y = np.reshape(batch_y,(-1,n_classes))


        # Run optimization op (backprop)
        #op = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        #grad_vals = sess.run([(g,v) for (g,v) in compute_gradients], feed_dict={x: batch_x, y: batch_y})
        #for gv in grad_vals:
        #    print (gv[0].shape, gv[1].shape)



        sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})

        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
        start = start + batch_size
        end = end + batch_size
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]
    test_data = x_test.reshape((-1, n_steps, n_input))
    test_label = np.reshape(y_test, (-1,2))

    loss = sess.run(cost, feed_dict={x: test_data, y: test_label})

    print("{:.6f}".format(loss))
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

    classifier_value = sess.run([classifier_collection], feed_dict={x:test_data})
    print (len(classifier_value))
    for value in classifier_value:
        print (value)
    proba_value = sess.run([proba_collection], feed_dict={x:test_data})
    for value in proba_value:
        print (value)

    v = Validation()
    v.calculateF1(input_value, classifier_value[0])
    v.allValidation(input_value, proba_value[0][:,1])

