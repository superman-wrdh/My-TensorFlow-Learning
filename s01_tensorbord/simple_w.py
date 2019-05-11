import tensorflow as tf

# 构造图 Graph结构
W = tf.Variable(2.0, dtype=tf.float32, name="Weight")
b = tf.Variable(1.0, dtype=tf.float32, name="Bias")
x = tf.placeholder(dtype=tf.float32, name="input")
with tf.name_scope("Output"):
    y = W * x + b

const = tf.constant(2.0)

path = "./log"
init = tf.global_variables_initializer()
# 创建session
with tf.Session() as sess:
    sess.run(init)  # 实现初始化操作
    tf.summary.FileWriter(path, graph=sess.graph)
    result = sess.run(y, {x: 3.0})  # 为 x 赋值 3
    print("y = W * x + b，值为 {}".format(result))  # 打印 y = W * x + b 的值，就是 7

# tensorboard --logdir=log
