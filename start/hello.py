import tensorflow as tf


if __name__ == '__main__':
    a = tf.random_normal((100, 100))
    b = tf.random_normal((100, 500))
    c = tf.matmul(a, b)
    sess = tf.InteractiveSession()
    sess.run(c)
