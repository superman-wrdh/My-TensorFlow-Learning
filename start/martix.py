import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1,m2)


def demo1():
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    sess.close()


def demo2():
    with tf.Session() as sess:
        result = sess.run(product)
        print(result)


if __name__ == '__main__':
    demo2()
