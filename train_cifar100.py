import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, regularizers
from resnet import ResNet18
import numpy as np
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
batchsz = 256

# 1. 归一化函数实现；cifar100 均值和方差，自己计算的。
img_mean = tf.constant([0.50736203482434500, 0.4866895632914611, 0.4410885713465068])
img_std  = tf.constant([0.26748815488001604, 0.2565930997269337, 0.2763085095510783])
def normalize(x, mean=img_mean, std=img_std):
    x = (x - mean)/std
    return x

# 2. 数据预处理，仅仅是类型的转换。    [-1~1]
def preprocess(x, y):
    x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])    # 上下填充4个0，左右填充4个0，变为[40, 40, 3]
    x = tf.image.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    # x: [0,255]=> -1~1   其次：normalizaion
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1) 调用函数；
    x = normalize(x)
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# 3. 学习率调整测率200epoch
def lr_schedule_300ep(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 120:
        return 0.02
    if epoch < 160:
        return 0.004
    if epoch < 200:
        return 0.0008
    if epoch < 250:
        return 0.0003
    if epoch < 300:
        return 0.0001
    else:
        return 0.00006

# 数据集的加载
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y)            # 或者tf.squeeze(y, axis=1)把1维度的squeeze掉。
y_test = tf.squeeze(y_test)  # 或者tf.squeeze(y, axis=1)把1维度的squeeze掉。
print(x.shape, y.shape, x_test.shape, y_test.shape)

# 训练集和标签包装成Dataset对象
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(5000).map(preprocess).batch(batchsz)
# 测试集和标签包装成Dataset对象
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(batchsz)

# 我们来取一个样本，测试一下sample的形状。
sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]),
      tf.reduce_max(sample[0]))  # 值范围为[0,1]

def main():
    # 输入：[b, 32, 32, 3]
    model = ResNet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    mydense = layers.Dense(100, activation=None, kernel_regularizer=regularizers.l2(5e-4))
    fc_net = Sequential([mydense])
    fc_net.build(input_shape=(None, 512))
    fc_net.summary()

    optimizer = optimizers.SGD(lr=0.1, momentum=0.9, decay=5e-4)
    variables = model.trainable_variables + fc_net.trainable_variables
    for epoch in range(500):

        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x, training=True)
                avgpool = layers.GlobalAveragePooling2D()(out)
                logits = fc_net(avgpool)
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)

            # 梯度求解
            grads = tape.gradient(loss, variables)
            # 梯度更新
            optimizer.apply_gradients(zip(grads, variables))
            # 学习率动态调整
            optimizer.lr = lr_schedule_300ep(epoch)
            # 每100个step打印一次
            if step % 100 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', float(loss), 'lr:', optimizer.lr.numpy())

        # 做测试
        total_num = 0
        total_correct = 0
        for x, y in test_db:
            out = model(x, training=False)
            avgpool = layers.GlobalAveragePooling2D()(out)
            output = fc_net(avgpool)
            # 预测可能性。
            prob = tf.nn.softmax(output, axis=1)
            pred = tf.argmax(prob, axis=1)  # 还记得吗pred类型为int64,需要转换一下。
            pred = tf.cast(pred, dtype=tf.int32)
            # 拿到预测值pred和真实值比较。
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)  # 转换为numpy数据

        acc = total_correct / total_num
        print('epoch:', epoch, 'test_acc:', acc)
        print('====================================================')


if __name__ == '__main__':
    main()
