import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

import layers as ly

training_file = 'train.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
x_gray = tf.placeholder(tf.float32, (None, 32, 32, 1))
grayscale = tf.image.rgb_to_grayscale(x)
normalize = ly.normalize_grayscale(x_gray)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_grayscale = sess.run(grayscale, feed_dict={x: X_train})
    x_normalize = sess.run(normalize, feed_dict={x_gray: x_grayscale})

f, axes = plt.subplots(1, 2)
axes[0].imshow(X_train[0])
axes[1].imshow(x_grayscale[0], cmap="gray")
plt.show()

f, axes = plt.subplots(1, 2)
axes[0].hist(x_grayscale[0].flatten())
axes[0].set_title("Grayscale")
axes[1].hist(x_normalize[0].flatten())
axes[1].set_title("Normalized")
plt.show()

