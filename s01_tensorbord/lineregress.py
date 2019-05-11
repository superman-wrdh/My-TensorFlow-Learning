import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 构建数据
points_num = 100
vector = []
# 用numpy正太随机函数生产100个点
# 对应方程 y  = 0.1 *x +0.2

for i in range(points_num):
    x1 = np.random.normal(0.1, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vector.append([x1, y1])

x_data = [v[0] for v in vector]  # 真实的x坐标
y_data = [v[1] for v in vector]  # 真实的y招标
plt.plot(x_data, y_data, "r*", label="original data")
plt.title("line regress")
plt.show()

# 构建线性回归模型
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 初始胡weight
b = tf.Variable(tf.zeros([1]))  # 初始化 Bias
y = W * x_data + b

# 定义损失函数  lost function or cost function
# 对tensor 所以维度计算 (y - y_data)^2 / n
loss = tf.reduce_mean((tf.square(y - y_data)))

# 用梯度下降优化器 来 优化 loss function
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 设置学习率 0.5
train = optimizer.minimize(loss)

# 创建会话 运行
sess = tf.Session()

# 初始化数据流图
init = tf.global_variables_initializer()
sess.run(init)

# 训练200部
for step in range(100):
    # 对每一步进行优化
    sess.run(train)
    # 打印每一步损失 ，权重 和偏差
    print("Step = %d,Loss=%f,[weight=%f,bias=%f]" % (step, sess.run(loss), sess.run(W), sess.run(b)))

# 图像2 绘制回归数据
plt.plot(x_data, y_data, "r*", label="original data")
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="line regress")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("line regress")
plt.show()

sess.close()
