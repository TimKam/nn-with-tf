"""
Simple python script to train a 1-layer neural network to classify cifar10 images use the TensorFlow library
Code adapted from:
https://kth.instructure.com/courses/4962/files/806181/download?verifier=9keHpBCsp2CAtZtVSKE8F4XsLLvqOu1zwgWkuRw2&wrap=1
"""

import tensorflow as tf
# class written to replicate input_data from tensorflow.examples.tutorials.mnist for CIFAR-10
from examples import cifar10_read


def run_network(path, batch_size, iterations, learning_rate):
    # read in the dataset
    print('reading in the CIFAR10 dataset')
    dataset = cifar10_read.read_data_sets(path, one_hot=True, reshape=False)

    using_tensorboard = True

    ##################################################
    # PHASE 1  - ASSEMBLE THE GRAPH

    # 1.1) define the placeholders for the input data and the ground truth labels

    # x_input can handle an arbitrary number of input vectors of length input_dim = d
    # y_  are the labels (each label is a length 10 one-hot encoding) of the inputs in x_input
    # If x_input has shape [N, input_dim] then y_ will have shape [N, 10]

    input_dim = 32 * 32 * 3  # d
    x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    F = tf.Variable(tf.truncated_normal([5, 5, 3, 64]), stddev=0.1)
    b = tf.Variable(tf.constant(.1, shape=[64]))
    S = tf.nn.conv2d(x_input, F, strides=[1, 1, 1, 1], padding='SAME') + b
    X1 = tf.nn.relu(S)
    H = tf.nn.max_pool(X1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    vec_H = tf.reshape(H, [-1, int(16*16*64)])
    b1 = tf.Variable(tf.constant(.1, shape=[64]))
    s1 = tf.matmul(W1, vec_H) + b1
    """W1 = tf.Variable(tf.truncated_normal([m, int(16 * 16 * nF)], stddev=.01))
    W2 = tf.Variable(tf.truncated_normal([10, m], stddev=.01))
    H_flat = tf.reshape(H, [-1, int(16 * 16 * nF)])
    b1 = tf.Variable(tf.constant(.1, shape=[m, int(16 * 16 * nF)]))
    print('##################')
    print('W1: ' + str(W1.shape))
    print('H: ' + str(H.shape))
    print('H_flat: ' + str(H_flat.shape))
    print('H_flat transposed: ' + str(tf.transpose(H_flat).shape))
    s1 = tf.matmul(W1, H_flat) + b1
    print('s1: ' + str(s1.shape))
    x1 = tf.nn.relu(s1)
    b2 = tf.Variable(tf.constant(.1, shape=[10, int(16 * 16 * nF)]))
    s = tf.matmul(W2, x1) + b2
    print('s: ' + str(s.shape))
    y = tf.nn.softmax(s)
    print('y: ' + str(y.shape))
    print('##################')"""


    # 1.2) define the parameters of the network
    # W: 3072 x 10 weight matrix,  b: bias vector of length 10

    W = tf.Variable(tf.truncated_normal([input_dim, 10], stddev=.01))
    b = tf.Variable(tf.constant(0.1, shape=[10]))

    # 1.3) define the sequence of operations in the network to produce the output
    # y = W *  x_input + b
    # y will have size [N, 10]  if x_input has size [N, input_dim]
    y = tf.matmul(x_input, W) + b

    # 1.4) define the loss funtion
    # cross entropy loss:
    # Apply softmax to each output vector in y to give probabilities for each class then compare to the ground truth labels via the cross-entropy loss and then compute the average loss over all the input examples
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # (optional) definiton of performance measures
    # definition of accuracy, count the number of correct predictions where the predictions are made by choosing the class with highest score
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 1.6) Add an op to initialize the variables.
    init = tf.global_variables_initializer()

    ##################################################


    # If using TENSORBOARD
    if using_tensorboard:
        # keep track of the loss and accuracy for the training set
        tf.summary.scalar('training loss', cross_entropy, collections=['training'])
        tf.summary.scalar('training accuracy', accuracy, collections=['training'])
        # merge the two quantities
        tsummary = tf.summary.merge_all('training')

        # keep track of the loss and accuracy for the validation set
        tf.summary.scalar('validation loss', cross_entropy, collections=['validation'])
        tf.summary.scalar('validation accuracy', accuracy, collections=['validation'])
        # merge the two quantities
        vsummary = tf.summary.merge_all('validation')

    ##################################################


    ##################################################
    # PHASE 2  - PERFORM COMPUTATIONS ON THE GRAPH

    n_iter = iterations

    # 2.1) start a TensorFlow session
    with tf.Session() as sess:
        ##################################################
        # If using TENSORBOARD
        if using_tensorboard:
            # set up a file writer and directory to where it should write info +
            # attach the assembled graph
            summary_writer = tf.summary.FileWriter(
                '/Users/timotheuskampik/Desktop/github/sensing_perception/graphs/network1/results/good2', sess.graph)
        ##################################################

        # 2.2)  Initialize the network's parameter variables
        # Run the "init" op (do this when training from a random initialization)
        sess.run(init)

        # 2.3) loop for the mini-batch training of the network's parameters
        for i in range(n_iter):

            # grab a random batch (size nbatch) of labelled training examples
            nbatch = batch_size
            batch = dataset.train.next_batch(nbatch)

            # create a dictionary with the batch data
            # batch data will be fed to the placeholders for inputs "x_input" and labels "y_"
            batch_dict = {
                x_input: batch[0],  # input data
                y_: batch[1],  # corresponding labels
            }

            # run an update step of mini-batch by calling the "train_step" op
            # with the mini-batch data. The network's parameters will be updated after applying this operation
            sess.run(train_step, feed_dict=batch_dict)

            # periodically evaluate how well training is going
            if i % 50 == 0:

                # compute the performance measures on the training set by
                # calling the "cross_entropy" loss and "accuracy" ops with the training data fed to the placeholders "x_input" and "y_"

                tr = sess.run([cross_entropy, accuracy],
                              feed_dict={x_input: dataset.train.images, y_: dataset.train.labels})

                # compute the performance measures on the validation set by
                # calling the "cross_entropy" loss and "accuracy" ops with the validation data fed to the placeholders "x_input" and "y_"

                val = sess.run([cross_entropy, accuracy],
                               feed_dict={x_input: dataset.validation.images, y_: dataset.validation.labels})

                info = [i] + tr + val
                print(info)

                ##################################################
                # If using TENSORBOARD
                if using_tensorboard:
                    # compute the summary statistics and write to file
                    summary_str = sess.run(tsummary, feed_dict={x_input: dataset.train.images, y_: dataset.train.labels})
                    summary_writer.add_summary(summary_str, i)

                    summary_str1 = sess.run(vsummary,
                                            feed_dict={x_input: dataset.validation.images, y_: dataset.validation.labels})
                    summary_writer.add_summary(summary_str1, i)
                ##################################################

        # evaluate the accuracy of the final model on the test data
        test_acc = sess.run(accuracy, feed_dict={x_input: dataset.test.images, y_: dataset.test.labels})
        final_msg = 'test accuracy:' + str(test_acc)
        print(final_msg)

    ##################################################


# location of the CIFAR-10 dataset
# CHANGE THIS PATH TO THE LOCATION OF THE CIFAR-10 dataset on your local machine
data_dir = '../Datasets/cifar-10-batches-py/'

run_network(data_dir, 200, 1000, 0.01)
# run_network(data_dir, 200, 1000, 0.0001)
# run_network(data_dir, 200, 1000, 0.1)

# run_network(data_dir, 10, 1000, 0.01)
# run_network(data_dir, 20000, 1000, 0.01)"""

# run_network(data_dir, 300, 50000, 0.05)

