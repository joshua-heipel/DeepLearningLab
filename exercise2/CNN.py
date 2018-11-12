import tensorflow as tf

class Layer:

    def shape(self):
        return [x.value for x in self.layer.get_shape()]

    def dimension(self):
        return len(self.layer.get_shape())

    def flattened(self):
        return FlatteningLayer(self)

class InitialLayer(Layer):

    def __init__(self, dimensions):
        self.layer = tf.placeholder(shape=dimensions, dtype="float32")

class FullyConnectedLayer(Layer):

    def __init__(self, input, n_units, activation=tf.nn.relu):
        self.weights = tf.Variable(tf.random_normal(shape=[input.shape()[1],n_units], stddev=0.01))
        self.bias = tf.Variable(tf.random_normal(shape=[n_units], stddev=0.01))
        self.activation = activation
        self.layer = tf.matmul(input.layer, self.weights) + self.bias
        if self.activation is not None:
            self.layer = self.activation(self.layer)

class ConvolutionalLayer(Layer):

    def __init__(self, input, filtersize, n_filters, strides=1, padding="SAME", pooling=None, poolingsize=1, activation=tf.nn.relu):
        self.filters = tf.Variable(tf.random_normal([filtersize, filtersize, input.shape()[3], n_filters], stddev=0.01))
        self.bias = tf.Variable(tf.random_normal([n_filters], stddev=0.01))
        self.activation = activation
        self.layer = tf.nn.conv2d(input=input.layer, filter=self.filters, strides=[1, strides, strides, 1], padding=padding) + self.bias
        if pooling is not None:
            self.layer = pooling(self.layer, ksize=[1, poolingsize, poolingsize, 1], strides=[1, poolingsize, poolingsize, 1], padding="SAME")
        if self.activation is not None:
            self.layer = self.activation(self.layer)


class FlatteningLayer(Layer):

    def __init__(self, input):
        self.layer = tf.reshape(input.layer, [-1, input.layer.get_shape()[1:].num_elements()])

class OutputLayer(Layer):

    def __init__(self, input, n_outputs, activation=tf.nn.softmax):
        self.weights = tf.Variable(tf.random_normal(shape=[input.shape()[1],n_outputs], stddev=0.01))
        self.bias = tf.Variable(tf.random_normal(shape=[n_outputs], stddev=0.01))
        self.activation = activation
        self.layer = tf.matmul(input.layer, self.weights) + self.bias
        if self.activation is not None:
            self.layer = self.activation(self.layer)
        self.setY(tf.placeholder("int64", [None, self.shape()[1]]))
        self.setLoss(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.layer, labels=self.true_y)))
        self.setClassError(1-tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.layer, axis=1), tf.argmax(self.true_y, axis=1)), "float32")))

    def setY(self, true_y):
        self.true_y = true_y

    def setLoss(self, loss):
        self.loss = loss

    def setClassError(self, error):
        self.classError = error

class Network:

    def __init__(self):
        self.layers = []
        self.last = None

    def addLayer(self, layer):
        self.layers += [layer]
        self.first = self.layers[0]
        self.last = layer

    def addLayers(self, layers):
        for l in layers:
            self.addLayer(l)
        self.first = self.layers[0]
        self.last = layers[-1]

class Calculation:

    def __init__(self, network, learning_rate):
        self.network = network
        self.setOptimizer(tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.network.last.loss))
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session.run(tf.global_variables_initializer())

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def restart(self):
        self.session.close()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def predict(self, x):
        return self.session.run(self.network.last.layer, feed_dict={self.network.first.layer: x})

    def loss(self, x, y):
        return self.session.run(self.network.last.loss, feed_dict={self.network.first.layer: x, self.network.last.true_y: y})

    def classError(self, x, y):
        return self.session.run(self.network.last.classError, feed_dict={self.network.first.layer: x, self.network.last.true_y: y})

    def optimize(self, x, y):
        self.session.run(self.optimizer, feed_dict={self.network.first.layer: x, self.network.last.true_y: y})

    def stop(self):
        self.session.close()
