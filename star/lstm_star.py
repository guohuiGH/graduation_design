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
import pandas as pd
import numpy as np
import sys
sys.path.append("../script")
from convert_input import Convert
from validation import Validation
from read_data import DataReader

reader = DataReader(1,3)

name = "StarLightCurves"

x_train, y_train = reader.read_train_data('../UCR_TS_Archive_2015/' + name + "/" + name + "_TEST")
x_test, y_test = reader.read_test_data("../UCR_TS_Archive_2015/" + name + "/" + name + "_TRAIN")
x_input, y_input = reader.temp_read_test_data("../UCR_TS_Archive_2015/" + name + "/" + name + "_TRAIN")

x_train = np.transpose(x_train, [1,0,2])


x_test = np.transpose(x_test, [1,0,2])



'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.2
training_iters = 100
batch_size = x_train[0].shape[0]

display_step = 10

# Network Parameters
n_input = 1024 # data input
n_steps =  1# timesteps
n_hidden = 80 # hidden layer num of features
n_classes = 3 #  total classes

# tf Graphiinput
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
    #lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.0)
    lstm_cell = rnn_cell.GRUCell(n_hidden)
    lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=0.8, output_keep_prob=0.6)
    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
#cost = tf.contrib.losses.mean_squared_error(pred,y)
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

proba_collection = tf.nn.softmax(pred)
classifier_collection = tf.argmax(pred, 1)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step < training_iters:
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        #batch_x = x_train
        batch_y = y_train
        # Reshape data to get 28 seq of 28 elements
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        #batch_x = np.transpose(x_train,[1,0,2])
        batch_x = x_train
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        #grad = sess.run(tf.gradients(cost, x),feed_dict={x: batch_x, y:batch_y})[0]

        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        #print(sess.run(tf.gradients(cost, x), feed_dict={x:batch_x, y:batch_y})[0])
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images

    #test_data = x_test.reshape((-1, n_steps, n_input))
    #test_data = np.transpose(x_test, [1,0,2])
    test_data = x_test
    test_label = y_test
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


    classifier_value = sess.run([classifier_collection], feed_dict={x:test_data})[0]
    proba_value = sess.run([proba_collection], feed_dict={x:test_data})
    v = Validation()


    v.calculateF1(y_input,classifier_value + 1)

