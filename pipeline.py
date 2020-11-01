# Load pickled data
import pickle
import sys

training_file = 'train.p'
validation_file = 'test.p'
testing_file = 'valid.p'

GRAYSCALE_IDX = 1
NORMALIZE_IDX = 2
DROP_OUTS_IDX = 3

print(sys.argv)
grayscale = bool(sys.argv[GRAYSCALE_IDX])
normalize = bool(sys.argv[NORMALIZE_IDX])
drop_outs = bool(sys.argv[DROP_OUTS_IDX])

print("Pipeline running with:")
print("Grayscale transform: " + str(grayscale))
print("Normalize grayscale transform: " + str(normalize))
print("Apply two dropouts: " + str(drop_outs))

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

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.imshow(image)
plt.show()
print(y_train[index])

### Import Tensorflow
import tensorflow as tf

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
EPOCHS = 10
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    depth = 3
    if grayscale:
        x = tf.image.rgb_to_grayscale(x)
        depth = 1
        if normalize:
            x = normalize_grayscale(x)

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W_1 = tf.Variable(tf.truncated_normal(shape=(5, 5, depth, 6), mean=mu, stddev=sigma))
    B_1 = tf.Variable(tf.zeros(shape=(1, 28, 28, 6)))
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    layer_1 = tf.nn.conv2d(x, W_1, strides=strides, padding=padding) + B_1

    # Activation.
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, high_keep_prob)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    k = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    layer_1 = tf.nn.max_pool(layer_1, k, strides, padding)

    # Layer 2: Convolutional. Output = 10x10x16.
    W_2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    B_2 = tf.Variable(tf.zeros(shape=(1, 10, 10, 16)))
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    layer_2 = tf.nn.conv2d(layer_1, W_2, strides=strides, padding=padding) + B_2
    
    # Activation.
    layer_2 = tf.nn.relu(layer_2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    k = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    layer_2 = tf.nn.max_pool(layer_2, k, strides, padding)

    # Flatten. Input = 5x5x16. Output = 400.
    fc = flatten(layer_2)
    fc = tf.nn.dropout(fc, low_keep_prob)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    W_3 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    B_3 = tf.Variable(tf.zeros(shape=(1, 120)))
    layer_3 = tf.matmul(fc, W_3) + B_3
    
    # Activation.
    layer_3 = tf.nn.relu(layer_3)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    W_4 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    B_4 = tf.Variable(tf.zeros(shape=(1, 84)))
    layer_4 = tf.matmul(layer_3, W_4) + B_4
    
    # Activation.
    layer_4 = tf.nn.relu(layer_4)
    layer_4 = tf.nn.dropout(layer_4, low_keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    W_5 = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    B_5 = tf.Variable(tf.zeros(shape=(1, 43)))
    logits = tf.matmul(layer_4, W_5) + B_5
    
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
