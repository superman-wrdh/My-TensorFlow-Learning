import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.multiply(a, b)


_a = np.linspace(-100, 100, 1000000).reshape((1000, 1000))
_b = np.linspace(-200, 400, 1000000).reshape((1000, 1000))
# print(_a, "\n")
# print(_b, "\n")
with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: _a, b: _b}))  # 把10赋给a，30赋给b
