import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

out = tf.multiply(input1, input2)

with tf.Session() as sess:
    sess.run(out, feed_dict={input1: [2.0], input2: [7.0]})
