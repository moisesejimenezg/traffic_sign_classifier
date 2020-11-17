# Load pickled data
import pickle
import sys
import random

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

import src.lenet as ln
from src.data_visualizer import DataVisualizer

training_file = 'train.p'
validation_file = 'test.p'
testing_file = 'valid.p'

GRAYSCALE_IDX = 1
NORMALIZE_IDX = 2
DROP_OUTS_IDX = 3
VISUALIZE_IDX = 4
DEBUG_MODE = False

print(sys.argv)
if not DEBUG_MODE:
    grayscale = sys.argv[GRAYSCALE_IDX] == "True"
    normalize = sys.argv[NORMALIZE_IDX] == "True"
    drop_outs = sys.argv[DROP_OUTS_IDX] == "True"
    visualize = sys.argv[VISUALIZE_IDX] == "True"
else:
    print("DEBUG MODE")
    grayscale = True
    normalize = True
    drop_outs = True
    visualize = True

print("Pipeline running with:")
print("Grayscale transform: " + str(grayscale))
print("Normalize grayscale transform: " + str(normalize))
print("Apply two dropouts: " + str(drop_outs))
print("Visualize: " + str(visualize))

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# Total number of images: 51839
n_train = 34799

n_validation = 12630

n_test = 4410

image_shape = [32, 32]

n_classes = 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.
# %matplotlib inline

if visualize:

    visualizer = DataVisualizer(X_test, y_test, y_valid, y_train, n_classes)
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
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_shot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)

### Set up accuracy computation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_shot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, low_keep_prob: 1, high_keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

### Train
low_keep_prob_v = 1
high_keep_prob_v = 1
if drop_outs:
    low_keep_prob_v = 0.6
    high_keep_prob_v = 0.7

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = X_train.shape[0]
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, low_keep_prob: low_keep_prob_v, high_keep_prob: high_keep_prob_v})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
