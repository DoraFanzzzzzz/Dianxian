# -- encoding:utf-8 --
# '''
# 用2个隐层的全连接DNN 手写数字识别：
# 数据：图片像素 28*28 拉伸后为 784
# 输出类别：10（深度学习的分类问题需要转化为 哑编码的格式）
# '''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 1.1 模型构建
input_size = 4  # 一张图片的像素为 28*28，输入特征
hidden1_size = 64  # 全连接网络 第一隐层的 神经元数
hidden2_size = 64
hidden3_size = 16
n_class = 4  # 输出层的类别，需要时压编码格式
# a 定义输入数据占位符
input_x = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='input_x')
input_y = tf.placdweholder(dtype=tf.float32, shape=[None, n_class], name='input_y')
# b 定义模型参数变量

layer_weights = {
    'w1': tf.Variable(initial_value=tf.truncated_normal(shape=[input_size, hidden1_size], stddev=0.1), name='w1'),
    'w2': tf.Variable(initial_value=tf.truncated_normal(shape=[hidden1_size, hidden2_size], stddev=0.1), name='w2'),
    'w3': tf.Variable(initial_value=tf.truncated_normal(shape=[hidden2_size, hidden3_size], stddev=0.1), name='w3'),
    'out_w': tf.Variable(initial_value=tf.truncated_normal(shape=[hidden2_size, n_class], stddev=0.1), name='out_w')
}
layer_biases = {'b1': tf.zeros(shape=[hidden1_size], name='b1'),
                'b2': tf.zeros(shape=[hidden2_size], name='b2'),
                'b3': tf.zeros(shape=[hidden3_size], name='b3'),
                'out_b': tf.zeros(shape=[n_class], name='out_b'),
                }


def create_neural_network():
    # 输入层到隐层1
    hidden_layer1 = tf.nn.relu(features=tf.add(tf.matmul(input_x, layer_weights['w1']), layer_biases['b1']))
    hidden_layer2 = tf.nn.relu(features=tf.add(tf.matmul(hidden_layer1, layer_weights['w2']), layer_biases['b2']))
    hidden_layer3 = tf.nn.relu(features=tf.add(tf.matmul(hidden_layer2, layer_weights['w3']), layer_biases['b3']))
    output_layer_decFun = tf.add(tf.matmul(hidden_layer2, layer_weights['out_w']), layer_biases['out_b'])
    return output_layer_decFun


# 计算模型预测值
dec_fun = create_neural_network()  # 输出层的决策函数值
y_prob = tf.nn.softmax(logits=dec_fun, axis=1)  # 输出层的softmax概率值
y_pre = tf.argmax(y_prob, axis=1)
'''将预测概率转化为
预测值，0、1 、 2....
9，一维结构'''

# 构建损失函数
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=input_y, logits=dec_fun, dim=-1, name=None))
# 构建参数优化器
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss=loss)
# 构建准确率的操作对象
acc_score_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(input_y, axis=1), y_pre), dtype=tf.float32))
# 图执行
with tf.Session() as sess:
    print('构建测试成功 ================================================================================= ')
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 加载数据
    mnist = input_data.read_data_sets('data/', one_hot=True)

    # 模型训练/模型持久
    batch_size = 60  # MBGD的 每次epoch小批次样本的数量，epoch 是迭代次数
    epoch_size = 10
    total_batch_num = mnist.train.num_examples // batch_size
    display = 4  # 定义先迭代次数的步长
    for epoch in range(epoch_size):
        for batch_idx in range(total_batch_num):
            # 获取x和y
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {input_x: batch_xs, input_y: batch_ys}
            sess.run(train_op, feed_dict=feeds)

        if (epoch + 1) % display == 0:
            # 获取损失函数值与准确率
            loss_train, acc_score_train = sess.run([loss, acc_score_op],
                                                   feed_dict={input_x: mnist.train.images, input_y: mnist.train.labels})
            loss_test, acc_score_test = sess.run([loss, acc_score_op],
                                                 feed_dict={input_x: mnist.test.images, input_y: mnist.test.labels})
            # 重新计算平均损失(相当于计算每batch样本的损失值)
            # avg_loss = avg_loss / total_batch_num
            print('训练集信息：第{}次迭代时，损失函数的值为{},训练集的准确率为{}'.format(epoch + 1, loss_train,
                                                                                       acc_score_train))
            print('测试集信息：第{}次迭代时，损失函数的值为{},测试集的准确率为{}'.format(epoch + 1, loss_test,
                                                                                       acc_score_test))
            print('=' * 100)
            if acc_score_train > 0.99 and acc_score_test > 0.975:
                print('最终训练的结果是训练集的准确率为：{}，测试集的真确率为{}'.format(acc_score_train, acc_score_test))
                break

    # 模型评估