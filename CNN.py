# coding=utf-8
# __author__ = "cyclone"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 导入MNIST数据集

sess = tf.InteractiveSession()  # 在运行图的时候，插入一些计算图

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])  # 占位符
# TODO 记录
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))  # 变量

sess.run(tf.initialize_all_variables())   # 变量被session初始化
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))      # tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))      # 计算出平均值
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# 多层卷积网络
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)     # 用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0


# 卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='a')  # 使用1步长（stride size），0边距（padding size）的模板


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')  # 2x2大小的模板做max pooling，然后缩放


# 1 层卷积
W_conv1 = weight_variable([5, 5, 1, 32])     # 前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。  5*5patch
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])   # -1为自动计算，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数（灰1彩3）

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 将patch与图片卷积，加入偏移量，形成特征，再用池化将每个patch的特征用最大值表示
h_pool1 = max_pool_2x2(h_conv1)
# 2层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 密集连接
W_fc1 = weight_variable([7 * 7 * 64, 1024])   # 图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")  # placeholder来代表一个神经元的输出在dropout中保持不变的概率
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 评估
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print ("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print ("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

path = 'Desktop\3pixel'
from PIL import Image
for t in range(10):
  im = Image.open("%s%s.bmp" % (path, str(t)))
  pix = im.load()
  [w, h] = im.size
  test_data = open("%stest_data.idx3-ubyte" %path,'w')
  count = 0
  for j in range(h):
    for k in range(w):
      count = count + 1
      string = (1-pix[k, j]/255.0)
      test_data.write("%f\t" % string)
      if count % 16 == 0:
        test_data.write("\n")
      tf.add_to_collection('losses'+str(t), string)
  a2 = tf.get_collection('losses'+str(t))
  print(a2)
  a3=[a2]
  print(a3)
  maxinize = tf.reduce_max(tf.argmax(y_conv, 1))
  print (sess.run(maxinize, feed_dict={x: a3,keep_prob: 1.0}))

