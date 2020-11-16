# Load pickled data
import pickle
import sys

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
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
# %matplotlib inline
import random
import numpy as np

if visualize:
    from data_visualizer import DataVisualizer

    y_joint = np.concatenate((y_test, y_valid, y_train))
    visualizer = DataVisualizer(X_test, y_joint, n_classes)
    visualizer.visualize()

### Import Tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

### Shuffle the data
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

### Normalize grayscale
def normalize_grayscale(image_data: tf.Tensor):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    max_value = 0.9
    min_value = 0.1
    X_std = (image_data - tf.reduce_min(image_data)) / (tf.reduce_max(image_data) - tf.reduce_min(image_data))
    X_scaled = X_std * (max_value - min_value) + min_value
    return X_scaled

### Setup
EPOCHS = 50
BATCH_SIZE = 256
mu = 0
sigma = 0.1

def linear_network(x_in, in_dim, out_dim, mu_in = mu, sigma_in = sigma):
    W = tf.Variable(tf.truncated_normal(shape=(in_dim, out_dim), mean=mu_in, stddev=sigma_in))
    B = tf.Variable(tf.zeros(shape=(1, out_dim)))
    y_out = tf.matmul(x_in, W) + B
    return y_out


def convolutional_network(x_in, in_h_w, in_depth, filter_h_w, out_depth):
    out_h_w = (in_h_w - filter_h_w) + 1 # no padding, stride = 1
    W = tf.Variable(tf.truncated_normal(shape=(filter_h_w, filter_h_w, in_depth, out_depth), mean=mu, stddev=sigma))
    B = tf.Variable(tf.zeros(shape=(1, out_h_w, out_h_w, out_depth)))
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    y_out = tf.nn.conv2d(x_in, W, strides=strides, padding=padding) + B
    return y_out

from tensorflow.compat.v1.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    depth = 3
    if grayscale:
        x = tf.image.rgb_to_grayscale(x)
        depth = 1
        if normalize:
            x = normalize_grayscale(x)

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    layer_1 = convolutional_network(x, 32, 1, 5, 6)

    # Activation.
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, high_keep_prob)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    k = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    layer_1 = tf.nn.max_pool(layer_1, k, strides, padding)

    # Layer 2: Convolutional. Output = 10x10x16.
    layer_2 = convolutional_network(layer_1, 14, 6, 5, 16)

    # Activation.
    layer_2 = tf.nn.relu(layer_2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    k = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    layer_2 = tf.nn.max_pool(layer_2, k, strides, padding)

    # Layer 2_1: Convolutional. Output = 1x1x412.
    layer_2_1 = convolutional_network(layer_2, 5, 16, 5, 412)

    # Flatten. Input = 5x5x16. Output = 400.
    fc = flatten(layer_2_1)
    fc = tf.nn.dropout(fc, high_keep_prob)

    # Layer 3: Fully Connected. Input = 412. Output = 122.
    layer_3 = linear_network(fc, 412, 122)

    # Activation.
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, low_keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    layer_4 = linear_network(layer_3, 122, 84)

    # Activation.
    layer_4 = tf.nn.relu(layer_4)
    layer_4 = tf.nn.dropout(layer_4, low_keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = linear_network(layer_4, 84, 43)

    return logits

### Setup CNN
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
low_keep_prob = tf.placeholder(tf.float32)
high_keep_prob = tf.placeholder(tf.float32)
one_shot_y = tf.one_hot(y, 43)

learning_rate = 0.001

logits = LeNet(x)
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
