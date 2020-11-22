# Load pickled data
import pickle
import sys
import random

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

import src.lenet as ln
from src.data_visualizer import DataVisualizer
from src.session_wrapper import SessionWrapper

training_file = "train.p"
validation_file = "test.p"
testing_file = "valid.p"
internet_file = "internet.p"

GRAYSCALE_IDX = 1
NORMALIZE_IDX = 2
DROP_OUTS_IDX = 3
VISUALIZE_IDX = 4
LOAD_SESS_IDX = 5
DEBUG_MODE = False

print(sys.argv)
if not DEBUG_MODE:
    grayscale = sys.argv[GRAYSCALE_IDX] == "True"
    normalize = sys.argv[NORMALIZE_IDX] == "True"
    drop_outs = sys.argv[DROP_OUTS_IDX] == "True"
    visualize = sys.argv[VISUALIZE_IDX] == "True"
    load_sess = sys.argv[LOAD_SESS_IDX] == "True"
else:
    print("DEBUG MODE")
    grayscale = True
    normalize = True
    drop_outs = True
    visualize = False
    load_sess = True

print("Pipeline running with:")
print("Grayscale transform: " + str(grayscale))
print("Normalize grayscale transform: " + str(normalize))
print("Apply two dropouts: " + str(drop_outs))
print("Visualize: " + str(visualize))
print("Load session: " + str(load_sess))

with open(training_file, mode="rb") as f:
    train = pickle.load(f)
with open(validation_file, mode="rb") as f:
    valid = pickle.load(f)
with open(testing_file, mode="rb") as f:
    test = pickle.load(f)

X_train, y_train = train["features"], train["labels"]
X_valid, y_valid = valid["features"], valid["labels"]
X_test, y_test = test["features"], test["labels"]

n_train = y_train.shape[0]

n_validation = y_valid.shape[0]

n_test = y_test.shape[0]

image_shape = X_train[0].shape

labels = set(y_test)
n_classes = len(labels)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

visualizer = DataVisualizer(X_test, y_test, y_valid, y_train, n_classes)
if visualize:
    visualizer.generate_histogram()
    visualizer.visualize()

### Import Tensorflow
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

### Shuffle the data
X_train, y_train = shuffle(X_train, y_train)

### Setup
EPOCHS = 20
BATCH_SIZE = 256

### Setup CNN
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
low_keep_prob = tf.placeholder(tf.float32)
high_keep_prob = tf.placeholder(tf.float32)
one_shot_y = tf.one_hot(y, 43)

learning_rate = 0.001

logits = ln.network(x, grayscale, normalize, low_keep_prob, high_keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=one_shot_y, logits=logits
)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)

### Set up accuracy computation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_shot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session_wrapper = SessionWrapper("./lenet_ba")

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = (
            X_data[offset : offset + BATCH_SIZE],
            y_data[offset : offset + BATCH_SIZE],
        )
        accuracy = sess.run(
            accuracy_operation,
            feed_dict={x: batch_x, y: batch_y, low_keep_prob: 1, high_keep_prob: 1},
        )
        total_accuracy += accuracy * len(batch_x)
    return total_accuracy / num_examples


### Train
low_keep_prob_v = 1
high_keep_prob_v = 1
if drop_outs:
    low_keep_prob_v = 0.5
    high_keep_prob_v = 0.7

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = X_train.shape[0]

    if not load_sess:
        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(
                    training_operation,
                    feed_dict={
                        x: batch_x,
                        y: batch_y,
                        low_keep_prob: low_keep_prob_v,
                        high_keep_prob: high_keep_prob_v,
                    },
                )

            validation_accuracy = evaluate(X_valid, y_valid)
            training_accuracy = evaluate(X_train, y_train)
            test_accuracy = evaluate(X_test, y_test)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print("Training Accuracy = {:.3f}".format(training_accuracy))
            print("Test Accuracy = {:.3f}".format(test_accuracy))
            print()
            visualizer.add_training_accuracy(training_accuracy)
            visualizer.add_validation_accuracy(validation_accuracy)
            visualizer.add_test_accuracy(test_accuracy)
        visualizer.visualize_accuracy()

        session_wrapper.write(sess)
    else:
        session_wrapper.read(sess)
        validation_accuracy = evaluate(X_valid, y_valid)
        test_accuracy = evaluate(X_test, y_test)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Test Accuracy = {:.3f}".format(test_accuracy))

    with open(internet_file, mode="rb") as f:
        internet = pickle.load(f)

        print("Labeling...")
        result = sess.run(
            tf.argmax(logits, 1),
            feed_dict={x: internet["images"][0:5], low_keep_prob: 1, high_keep_prob: 1},
        )
        accuracy = sess.run(
            accuracy_operation,
            feed_dict={
                x: internet["images"],
                y: internet["labels"],
                low_keep_prob: 1,
                high_keep_prob: 1,
            },
        )
        print("Internet Accuracy = {:.3f}".format(accuracy))

        for i in range(0, len(result)):
            print(
                "name: "
                + internet["file_names"][i]
                + " predicted label: "
                + str(result[i])
                + " label: "
                + str(internet["labels"][i])
            )

        softmax = sess.run(
            tf.nn.softmax(logits),
            feed_dict={x: internet["images"][0:5], low_keep_prob: 1, high_keep_prob: 1},
        )
        softmax_tuples = []
        for image_softmax in softmax:
            softmax_tuples.append([])
            for i in range(0, len(image_softmax)):
                softmax_tuples[-1].append({"id": i, "p": image_softmax[i]})
        output_strings = []
        for softmax_tuple in softmax_tuples:
            softmax_tuple.sort(key=lambda x: x["p"])
            string = ""
            for i in range(0, 5):
                idx = - 1 - i
                sm = softmax_tuple[idx]
                string += "| " + str(i) + " | " + str(sm["id"]) + " | " + internet["names"][sm["id"]] + " | " + str(sm["p"] * 100) + ' |\n'
            print("| n | id | name | probability |")
            print("|:-:|:-:|:-:|:-:|")
            print(string)
