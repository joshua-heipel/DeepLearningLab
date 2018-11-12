from __future__ import print_function

import sys
import argparse
import gzip
import json
import os
import pickle
from CNN import *
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np


def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels

def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32')
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)

def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, batch_size, filter_size):

    # define convolutional neural network:
    # with tf.device('/gpu:0'):
    cnn = Network()
    l1 = InitialLayer([None,28,28,1])
    l2 = ConvolutionalLayer(l1, filter_size, num_filters, pooling=tf.nn.max_pool, poolingsize=2)
    l3 = ConvolutionalLayer(l2, filter_size, num_filters, pooling=tf.nn.max_pool, poolingsize=2)
    l4 = FullyConnectedLayer(l3.flattened(), 128, activation=None)
    l5 = OutputLayer(l4, 10)
    cnn.addLayers([l1, l2, l3, l4, l5])

    print("... network configuration: " + " ,".join("x".join(str(x) for x in l.shape()[1:]) for l in cnn.layers))

    model = Calculation(cnn, lr)

    print("... optimize network")

    learning_curve = []
    n = x_train.shape[0]
    n_batches = n // batch_size

    # optimize parameters with stochastic gradient descent
    for j in range(num_epochs):
        print("....... epoch number {0}".format(j), end='', flush=True)
        perm = np.random.permutation(n)
        for i in range(n_batches):
            x_batch = x_train[perm[(batch_size*i):(batch_size*(i+1))],:,:,:]
            y_batch = y_train[perm[(batch_size*i):(batch_size*(i+1))],:]
            model.optimize(x_batch, y_batch)
        learning_curve += [float(model.classError(x_valid, y_valid))]
        print(" ... error: {0:.2f}".format(learning_curve[-1]), flush=True)

    return learning_curve, model

def test(x_test, y_test, model):
    return float(model.classError(x_test, y_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=16, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")
    parser.add_argument("--filter_size", default=3, type=int, nargs="?",
                        help="Filter width and height")
    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = args.filter_size

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

    learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)

    plt.plot(learning_curve, label='training')
    plt.xlabel('epoche')
    plt.ylabel('validation error')
    plt.title("")
    plt.legend()
    plt.show()

    test_error = test(x_test, y_test, model)

    model.stop()

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["filter_size"] = filter_size
    results["learning_curve"] = learning_curve
    results["test_error"] = test_error

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()
