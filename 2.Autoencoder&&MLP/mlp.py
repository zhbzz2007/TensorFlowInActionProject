#coding:utf-8

# 使用多层感知机进行图像分类

# 载入TensorFlow并加载Mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("../Mnist_data",one_hot = True)
sess = tf.InteractiveSession()

# 输入节点数
in_units = 784
# 隐含层节点数
h1_units = 300
# 隐含层的权重和偏置
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev = 0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
# 输出层的权重和偏置
W2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

# x为输入
# keep_prob为Dropout的比率，训练时小于1，预测时等于1
x = tf.placeholder(tf.float32,[None,in_units])
keep_prob = tf.placeholder(tf.float32)

# 计算隐含层
hidden1 = tf.nn.relu(tf.matmul(x,W1) + b1)
# Dropout
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
# 计算输出
y = tf.nn.softmax(tf.matmul(hidden1_drop,W2) + b2)

# 使用交叉信息熵作为损失函数，优化器选择Adagrad
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()

# 开始训练
for i in range(3000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})

# 对模型进行准确率预测
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))