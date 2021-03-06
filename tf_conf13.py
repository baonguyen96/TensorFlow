import os
import warnings
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

tfe.enable_eager_execution()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

dim_hidden = 2048

layer_cnn0 = tf.layers.Conv2D(32, 5, activation=tf.nn.relu)
layer_pool2x2 = tf.layers.MaxPooling2D(2, 2)
layer_pool3x3 = tf.layers.MaxPooling2D(3, 3)
layer_pool4x4 = tf.layers.MaxPooling2D(4, 4)
layer_cnn1 = tf.layers.Conv2D(64, 5, activation=tf.nn.relu)
layer_flatten = tf.layers.Flatten()
layer_fc0 = tf.layers.Dense(dim_hidden, activation=tf.nn.relu)
layer_dropout = tf.layers.Dropout(rate=0.9)
layer_fc1 = tf.layers.Dense(10, activation=None)


# forward propagation
def prediction(X, training):
    values = tf.constant(X)
    values = layer_pool4x4(values)  # this must be the first layer
    values = layer_cnn0(values)
    values = layer_pool2x2(values)
    values = layer_flatten(values)
    values = layer_fc0(values)
    values = layer_dropout(values, training=training)
    values = layer_fc1(values)
    return values


def calculate_cross_entropy_loss(X, y, training):
    logits = prediction(X, training)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    loss = tf.reduce_mean(loss)
    return loss


def binary_accuracy(X, y):
    logits = prediction(X, training=False)
    predict = tf.argmax(logits, 1).numpy()
    target = np.argmax(y, 1)
    accuracy = np.sum(predict == target) / len(target)
    return accuracy


X_validation = mnist.validation.images
y_validation = mnist.validation.labels
X_validation = X_validation.reshape([-1, 28, 28, 1])


def v_binary_accuracy():
    return binary_accuracy(X_validation, y_validation)


optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
batch_size = 50
iters = 2000
limit = batch_size * iters

if limit > 100000:
    print("invalid size: {} x {} = {}".format(batch_size, iters, limit))
    exit(1)

for i in range(iters):
    X, y = mnist.train.next_batch(batch_size)
    X = X.reshape([-1, 28, 28, 1])
    optimizer.minimize(lambda: calculate_cross_entropy_loss(X, y, True))

    if i % 100 == 0:
        batch_accuracy = binary_accuracy(X, y)
        validation_accuracy = v_binary_accuracy()
        print("batch %d, batch accuracy %.3f validation accuracy %.3f" %
              (i, batch_accuracy, validation_accuracy))

# evaluate the result
X, y = mnist.test.images, mnist.test.labels
X = X.reshape([-1, 28, 28, 1])
test_accuracy = binary_accuracy(X, y)
print("test accuracy %g" % test_accuracy)
