import tensorflow as tf
import numpy as np
# 下载并载入mnist手写数据库
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data", one_hot=True)

# one_hot 独热码编码（encoding 形式）
# 0,1,2,3,4,5,6,7,8,9
# 0:1000000000
# 1:0100000000
# 2:0010000000
# 3:0001000000
# 4:0000100000
# 5:0000010000

