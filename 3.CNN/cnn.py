#coding:utf-8

# 实现简单的卷积神经网络

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("../Mnist_data",one_hot = True)
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return initial

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])

# 定义第一个卷积层
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_con1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_con1)

# 定义第二个卷积层
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 定义全连接层
W_fc1 = weight_variable([7 * 7 * 64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

# 使用Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# 连接softmax层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# 定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 评测准确率
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()
maxLoop = 20000
batch_size = 50
# 开始训练
for i in range(maxLoop):
    batch = mnist.train.next_batch(batch_size)
    if i % 100 == 0:
        train_accuracy = accuracy.eval( feed_dict = {x:batch[0], y_ : batch[1],keep_prob:1.0} )
        print ("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_ : batch[1],keep_prob:1.0})
# 在测试集上测试准确率
print("test accuracy %g"%accuracy.eval(feed_dict = {x:mnist.test.images,y_ : mnist.test.labels,keep_prob:1.0}))