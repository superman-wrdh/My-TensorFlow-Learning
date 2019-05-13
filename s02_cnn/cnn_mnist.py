import tensorflow as tf
import numpy as np
# 下载并载入mnist手写数据库
from tensorflow.examples.tutorials.mnist import input_data

# 下载 并载入 MNIST手写数据库 （55000 * 28 *28）
mnist = input_data.read_data_sets("mnist_data", one_hot=True)

# one_hot 独热码编码（encoding 形式）
# 0,1,2,3,4,5,6,7,8,9
# 0:1000000000
# 1:0100000000
# 2:0010000000
# 3:0001000000
# 4:0000100000
# 5:0000010000
# 6:0000001000
# 7:0000000100
# 8:0000000010
# 9:0000010001

# None表示张量的第一个维度 可以是任何长度
input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255.  # 0-255灰度值

output_y = tf.placeholder(tf.int32, [None, 10])

input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])  # 改变形状之后的输入

# 从test数据集中选取3000个手写数据的图片和对应标签
test_x = mnist.test.images[:3000]  # 图片
test_y = mnist.test.labels[:3000]  # 标签

# 构建我们的神经卷积网络
convl = tf.layers.conv2d(inputs=input_x_images,  # input_x_images 形状 [28,28,1]
                         filters=32,  # 32个过滤器 输出深度(depth) 是 32
                         kernel_size=[5, 5],  # 过滤器在二维的大小是 (5 * 5)
                         strides=1,  # 步长是1
                         padding='same',
                         activation=tf.nn.relu  # 激活函数
                         )  # 形状 [28,28,32]

# 第一层 池 化
pool1 = tf.layers.max_pooling2d(
    inputs=convl,  # 形状 28 * 28 *32
    pool_size=[2, 2],  # 过滤器在二维大小是(2 *2)
    strides=2  # 步长是2
)  # 形状是 [14,14,32]

# 第2层卷积
conv2 = tf.layers.conv2d(inputs=pool1,  # input_x_images 形状 [14,14,32]
                         filters=64,  # 64个过滤器 输出深度(depth) 是 64
                         kernel_size=[5, 5],  # 过滤器在二维的大小是 (5 * 5)
                         strides=1,  # 步长是1
                         padding='same',
                         activation=tf.nn.relu  # 激活函数
                         )  # 形状 [14,14,64]

# 第二层 池 化
pool1 = tf.layers.max_pooling2d(
    inputs=conv2,  # 形状 14 * 14 *64
    pool_size=[2, 2],  # 过滤器在二维大小是(2 *2)
    strides=2  # 步长是2
)  # 形状是 [7,7,64]


