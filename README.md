## O. CBAM-TensorFlow2.0

- CBAM(Convolutional Block Attention Module) implementation on TensowFlow2.0
- <font color=black> 本论文来自ECCV2018，主要在传统的CNN上引入通道注意力机制和空间注意力机制提升性能。论文地址：[CBAM!](https://arxiv.org/abs/1807.06521)
- 欢迎各位朋友star,接下来还会继续更新项目!
- 公式显示有乱码的话，Google浏览器添加：[MathJax Plugin for Github!](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related)

## 一. 论文摘要
<font   color=black> 本文提出了卷积块注意模块(CBAM)，这是一个简单而有效的前馈卷积神经网络注意模块。在给定中间特征图的情况下，我们的模块按照通道和空间两个独立的维度依次推断注意图，然后将注意图乘入输入特征图进行自适应特征细化。因为CBAM是一个轻量级的通用模块，它可以无缝地集成到任何CNN架构中，开销可以忽略不计，并且可以与基本CNNs一起进行端到端的培训。我们通过在ImageNet-1K、MS COCO检测和VOC 2007检测数据集上的大量实验来验证我们的CBAM。我们的实验表明，在不同的模型下，分类和检测性能都得到了一致的提高，说明了CBAM的广泛适用性。

## 二. CBAM的网络结构
### 2.1. 总体的描述

- 对于一个中间层的$\mathbf F \in \mathbb{R}^{C \times H \times W}$，CBAM将会顺序推理出1维的channel attention map $\mathbf M_{c} \in \mathbb{R}^{C  \times 1  \times 1}$ 以及2维的spatial attention map $\mathbf M_{s} \in \mathbb{R}^{1 \times H  \times W}$，整个过程如下所示： 
$$\mathbf{F}^{\prime}=\mathbf{M}_{\mathbf{c}}(\mathbf{F}) \otimes \mathbf{F}\tag{1}$$
$$\mathbf{F}^{\prime  \prime}=\mathbf{M}_{\mathbf{s}}\left(\mathbf{F}^{\prime}\right) \otimes \mathbf{F}^{\prime}\tag{2}$$

- **其中：** $⊗$为element-wise multiplication，首先将channel attention map与输入的feature map相乘得到 $\mathbf{F}^{\prime}$， 之后计算 $\mathbf{F}^{\prime}$ 的spatial attention map，并将两者相乘得到最终的输出 $\mathbf{F}^{\prime \prime}$。

<center><image src="https://github.com/kobiso/CBAM-keras/blob/master/figures/overview.png?raw=true" width="100%">
     

### 2.2. 通道注意力机制

- 首先是通道注意力，我们知道一张图片经过几个卷积层会得到一个特征矩阵，这个矩阵的通道数就是卷积层核的个数。那么，一个常见的卷积核经常达到1024，2048个，并不是每个通道都对于信息传递非常有用了的。因此，通过对这些通道进行过滤，也就是注意，来得到优化后的特征。
<font   color=black>**主要思路就是：增大有效通道权重，减少无效通道的权重。** 公式表示为如下：

$$\begin{aligned} \mathbf{M}_{\mathbf{c}}(\mathbf{F}) &=\sigma(\text{MLP(AvgPool}(\mathbf{F}))+\text{MLP}(\operatorname{MaxPool} (\mathbf{F}))) \\ &= \sigma\left(\mathbf{W}_{\mathbf{1}}(\mathbf{W}_{\mathbf{0}}(\mathbf{F}_{\text {avg}}^{\mathbf{c}}))+\mathbf{W}_{\mathbf{1}}\left(\mathbf{W}_{\mathbf{0}}\left(\mathbf{F}_{\max }^{\mathbf{c}}\right)\right)\right)\tag{3}\end{aligned}$$ 

- **其中：** $\mathbf{F}_{\text {avg}}^\mathbf{c}$ 和 $\mathbf{F}_{\text {max}}^\mathbf{c}$ 表示对feature map在空间维度上使用**最大池化**和**平均池化**。$\mathbf{W}_{0} \in \mathbb{R}^{C / r  \times C}, \quad \mathbf{W}_{1} \in \mathbb{R}^{C  \times C / r}$，$\mathbf{W}_{0}$ 后使用了Relu作为激活函数，<font   color=blue>$\sigma$ 表示**Sigmoid**函数</font>。

- **此外：** 共享网络是由一个隐藏层和多层感知机(MLP)组成。为了减少参数开销，隐藏的激活大小设置为 $\mathbb{R}^{C / r \times 1 \times 1}$，其中 $r$ 是压缩率。在将共享网络应用于矢量之后，我们使用**逐元素求和**来合并输出特征向量。

  <center><image src="https://img-blog.csdnimg.cn/20191230145340134.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FiYzEzNTI2MjIyMTYw,size_16,color_FFFFFF,t_70" width="100%">
  
- **注意：** 这里非常像SENet，SENet在很多论文中都被证实对效果有提升，这里的区别是，SENet采用的是平均值的pooling，这篇论文又加入了最大值pooling。作者在论文中，通过对比实验，证实max pooling提高了效果。这里的mlp的中间层较小，这个可能有助于信息的整合。

- <font   color=black><font   color=blue>**通道注意力模块代码(方式1)**</font>，推荐使用这种，这样喂入数据可以是None，就是可以自适应。
```powershell
class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg= layers.GlobalAveragePooling2D()
        self.max= layers.GlobalMaxPooling2D()

        self.fc1 = layers.Dense(in_planes//ratio, kernel_initializer='he_normal', activation='relu',
                                kernel_regularizer=regularizers.l2(5e-4),
                                use_bias=True, bias_initializer='zeros')
        self.fc2 = layers.Dense(in_planes, kernel_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(5e-4),
                                use_bias=True, bias_initializer='zeros')

    def call(self, inputs):
        avg_out = self.fc2(self.fc1(self.avg(inputs)))
        max_out = self.fc2(self.fc1(self.max(inputs)))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)
        out = layers.Reshape((1, 1, out.shape[1]))(out)

        return out
```
- <font   color=black><font   color=blue>**通道注意力模块代码(方式2)**</font>，更推荐使用这种，这样喂入数据可以是None，就是可以自适应。

```powershell
class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()

        self.avg_out= layers.GlobalAveragePooling2D()
        self.max_out= layers.GlobalMaxPooling2D()

        self.fc1 = layers.Dense(in_planes//ratio, kernel_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(5e-4),
                                activation=tf.nn.relu,
                                use_bias=True, bias_initializer='zeros')
        self.fc2 = layers.Dense(in_planes, kernel_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(5e-4),
                                use_bias=True, bias_initializer='zeros')

    def call(self, inputs):
        avg_out = self.avg_out(inputs)
        max_out = self.max_out(inputs)
        out = tf.stack([avg_out, max_out], axis=1)  # shape=(None, 2, fea_num)
        out = self.fc2(self.fc1(out))
        out = tf.reduce_sum(out, axis=1)      		# shape=(256, 512)
        out = tf.nn.sigmoid(out)
        out = layers.Reshape((1, 1, out.shape[1]))(out)

        return out
```

 - <font   color=black><font   color=blue>**通道注意力模块代码(方式3)**</font>，喂入数据的时候时候需要指定具体的batchsz值。

```powershell
class ChannelAttention(layers.Layer):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()

        self.avg= layers.GlobalAveragePooling2D()
        self.max= layers.GlobalMaxPooling2D()

        self.fc1 = layers.Dense(in_planes//16, kernel_initializer='he_normal', activation='relu',
                                use_bias=True, bias_initializer='zeros')
        self.fc2 = layers.Dense(in_planes, kernel_initializer='he_normal', use_bias=True,                             
        						bias_initializer='zeros')

    def call(self, inputs):
        avg_out = self.fc2(self.fc1(self.avg(inputs)))
        max_out = self.fc2(self.fc1(self.max(inputs)))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)
        out = tf.reshape(out, [out.shape[0], 1, 1, out.shape[1]])
        out = tf.tile(out, [1, inputs.shape[1], inputs.shape[2], 1])

        return out
```

- <font   color=black><font   color=blue>**通道注意力模块代码(方式4)**</font>，使用 $1×1$ 卷积替换全连接层。
    
```powershell
class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg= layers.GlobalAveragePooling2D()
        self.max= layers.GlobalMaxPooling2D()
        self.conv1 = layers.Conv2D(in_planes//ratio, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   use_bias=True, activation=tf.nn.relu)
        self.conv2 = layers.Conv2D(in_planes, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   use_bias=True)

    def call(self, inputs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = layers.Reshape((1, 1, avg.shape[1]))(avg)   # shape (None, 1, 1 feature)
        max = layers.Reshape((1, 1, max.shape[1]))(max)   # shape (None, 1, 1 feature)
        avg_out = self.conv2(self.conv1(avg))
        max_out = self.conv2(self.conv1(max))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)

        return out
```

### 2.3. 空间注意力机制
- <font   color=black> 论文中，作者认为通道注意力关注的是：what，然而空间注意力关注的是：where。

<font   color=black>$$\begin{aligned}\mathbf{M}_{\mathbf{s}}(\mathbf{F}) &=\sigma\left(f^{7 \times 7}(\left[\text {AvgPool}(\mathbf{F}) ; \text {MaxPool}(\mathbf{F})]\right))\right) \\ &=\sigma\left(f^{7 \times 7}\left(\left[\mathbf{F}_{\text {avg }}^{\mathrm{s}} ; \mathbf{F}_{\text {max }}^{\mathrm{s}}\right]\right)\right)\tag{4} \end{aligned} $$


- <font   color=black>**注意：** 这里同样使用了avg-pooling和max-pooling来对信息进行评估，使用一个 $7×7$ 的卷积来进行提取。注意权重都通过<font   color=blue>**sigmoid**来进行归一化</font>。

<center><image src="https://img-blog.csdnimg.cn/2019123014591351.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FiYzEzNTI2MjIyMTYw,size_16,color_FFFFFF,t_70" width="100%">

- <font   color=black><font   color=blue>**空间注意力模块代码(方式1)**</font>，推荐使用这种，这样喂入数据可以是None，就是可以自适应。


```powershell
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = regularized_padded_conv(1, kernel_size=kernel_size, strides=1, activation='sigmoid')

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        out = tf.stack([avg_out, max_out], axis=-1)             # 创建一个维度,拼接到一起concat。
        out = self.conv1(out)

        return out
```

- <font   color=black><font   color=blue>**空间注意力模块代码(方式2)**</font>，喂入数据的时候时候需要指定具体的batchsz值。

```powershell
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = regularized_padded_conv(1, kernel_size=kernel_size, strides=1)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        avg_out = tf.reshape(avg_out, [avg_out.shape[0], avg_out.shape[1], avg_out.shape[2], 1])
        max_out = tf.reduce_max(inputs, axis=3)
        max_out = tf.reshape(max_out, [max_out.shape[0], max_out.shape[1], max_out.shape[2], 1])
        out = tf.concat([avg_out, max_out], axis=3)
        out = self.conv1(out)
        out = tf.nn.sigmoid(out)

        return out
```

## 三. Tensorflow2.0+ResNet18+CIFAR100实战
### 3.1. Biasblock结构图
- <font   color=black>将模型应用到每一个ResNet block的输出上。
<center><image src="https://img-blog.csdnimg.cn/20200102235102746.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FiYzEzNTI2MjIyMTYw,size_16,color_FFFFFF,t_70" width="100%">
     
     
### 3.2. 网络结构的搭建

>  - <font   color=black> **网络结构resnet.py**

```python
import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers
import tensorflow.keras as keras

""" 第2个版本  2019-12-30  @devinzhang  更接近最真实的resnet18 """

#  定义一个3x3卷积！kernel_initializer='he_normal','glorot_normal'
def regularized_padded_conv(*args, **kwargs):
    return layers.Conv2D(*args, **kwargs, padding='same', use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(5e-4))

############################### 通道注意力机制 ###############################
class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg= layers.GlobalAveragePooling2D()
        self.max= layers.GlobalMaxPooling2D()
        self.conv1 = layers.Conv2D(in_planes//ratio, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   use_bias=True, activation=tf.nn.relu)
        self.conv2 = layers.Conv2D(in_planes, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   use_bias=True)

    def call(self, inputs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = layers.Reshape((1, 1, avg.shape[1]))(avg)   # shape (None, 1, 1 feature)
        max = layers.Reshape((1, 1, max.shape[1]))(max)   # shape (None, 1, 1 feature)
        avg_out = self.conv2(self.conv1(avg))
        max_out = self.conv2(self.conv1(max))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)

        return out

############################### 空间注意力机制 ###############################
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = regularized_padded_conv(1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        out = tf.stack([avg_out, max_out], axis=3)             # 创建一个维度,拼接到一起concat。
        out = self.conv1(out)

        return out

# 1.定义 Basic Block 模块。对于Resnet18和Resnet34
class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 1. BasicBlock模块中的共有2个卷积;BasicBlock模块中的第1个卷积层;
        self.conv1 = regularized_padded_conv(out_channels, kernel_size=3, strides=stride)
        self.bn1 = layers.BatchNormalization()

        # 2. 第2个；第1个卷积如果做stride就会有一个下采样，在这个里面就不做下采样了。这一块始终保持size一致，把stride固定为1
        self.conv2 = regularized_padded_conv(out_channels, kernel_size=3, strides=1)
        self.bn2 = layers.BatchNormalization()
        ############################### 注意力机制 ###############################
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        # 3. 判断stride是否等于1,如果为1就是没有降采样。
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Sequential([regularized_padded_conv(self.expansion * out_channels,
                                                                kernel_size=1, strides=stride),
                                        layers.BatchNormalization()])
        else:
            self.shortcut = lambda x, _: x

    def call(self, inputs, training=False):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        ############################### 注意力机制 ###############################
        out = self.ca(out) * out
        out = self.sa(out) * out

        out = out + self.shortcut(inputs, training)
        out = tf.nn.relu(out)

        return out

##############################################################
# 2. ResBlock 模块。继承keras.Model或者keras.Layer都可以
class ResNet(keras.Model):

    # 第1个参数layer_dims：[2, 2, 2, 2] 4个Res Block，每个包含2个Basic Block，第3参数num_classes：我们的全连接输出，取决于输出有多少类。
    def __init__(self, blocks, layer_dims, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 0. 预处理卷积层；实现比较灵活可以加MAXPool2D，或者不加，这里没加。注意这里的channels需要和layer1的channels是一样的，不然能add。
        self.stem = Sequential([regularized_padded_conv(64, kernel_size=3, strides=1),
                                layers.BatchNormalization()])

        # 1. 创建4个ResBlock；注意第1项不一定以2倍形式扩张，都是比较随意的，这里都是经验值。
        self.layer1 = self.build_resblock(blocks, 64,   layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(blocks, 128,  layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(blocks, 256,  layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(blocks, 512,  layer_dims[3], stride=2)
        # self.final_bn  = layers.BatchNormalization()

        # self.avgpool = layers.GlobalAveragePooling2D()
        # self.fc = layers.Dense(num_classes)

    # 2. 创建ResBlock
    def build_resblock(self, blocks, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)                    # [1]*3 = [1, 1, 1]
        res_blocks = Sequential()

        for stride in strides:
            res_blocks.add(blocks(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return res_blocks

    def call(self,inputs, training=False):
        # __init__中准备工作完毕；下面完成前向运算过程。
        out = self.stem(inputs, training)
        out = tf.nn.relu(out)

        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)
        # out = self.final_bn(out, training=training)
        # out = tf.nn.relu(out)

        # 做一个global average pooling，得到之后只会得到一个channel，不需要做reshape操作了。 shape为 [batchsize, channel]
        # out = self.avgpool(out)
        # [b, 100]
        # out = self.fc(out)

        return out

##############################################################
""" Resnet18 """
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

```

### 3.3. 前馈以及反馈网络
>  - <font   color=black>**主程序my_resnet.py**

```python
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
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
                loss = loss + tf.add_n(model.losses) + tf.add_n(fc_net.losses)

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

```
### 3.4. CIFAR100测试结果

>  - <font   color=black> **运行结果**

```powershell
ssh://zhangkf@192.168.136.64:22/home/zhangkf/anaconda3/envs/tf2/bin/python -u /home/zhangkf/johnCodes/TF1/resnet_my_soft.py
(50000, 32, 32, 3) (50000,) (10000, 32, 32, 3) (10000,)
sample: (256, 32, 32, 3) (256,) tf.Tensor(-1.8967644, shape=(), dtype=float32) tf.Tensor(2.0227804, shape=(), dtype=float32)
Model: "res_net"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sequential (Sequential)      multiple                  1984      
_________________________________________________________________
sequential_1 (Sequential)    multiple                  148480    
_________________________________________________________________
sequential_2 (Sequential)    multiple                  526848    
_________________________________________________________________
sequential_4 (Sequential)    multiple                  2102272   
_________________________________________________________________
sequential_6 (Sequential)    multiple                  8398848   
=================================================================
Total params: 11,178,432
Trainable params: 11,168,832
Non-trainable params: 9,600
_________________________________________________________________
Model: "sequential_8"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                multiple                  51300     
=================================================================
Total params: 51,300
Trainable params: 51,300
Non-trainable params: 0
_________________________________________________________________
epoch: 0 step: 0 loss: 10.013138771057129 lr: 0.1
epoch: 0 step: 100 loss: 8.118321418762207 lr: 0.1
epoch: 0 test_acc: 0.1259
====================================================
epoch: 1 step: 0 loss: 7.027041912078857 lr: 0.1
epoch: 1 step: 100 loss: 6.452754020690918 lr: 0.1
epoch: 1 test_acc: 0.1276
====================================================
epoch: 2 step: 0 loss: 5.762089729309082 lr: 0.1
epoch: 2 step: 100 loss: 5.085740089416504 lr: 0.1
epoch: 2 test_acc: 0.2322
====================================================
epoch: 3 step: 0 loss: 4.7743000984191895 lr: 0.1
epoch: 3 step: 100 loss: 4.29296350479126 lr: 0.1
epoch: 3 test_acc: 0.3211
====================================================
epoch: 4 step: 0 loss: 3.8665976524353027 lr: 0.1
epoch: 4 step: 100 loss: 3.415522575378418 lr: 0.1
epoch: 4 test_acc: 0.3536
====================================================
epoch: 5 step: 0 loss: 3.4253320693969727 lr: 0.1
epoch: 5 step: 100 loss: 3.275803804397583 lr: 0.1
epoch: 5 test_acc: 0.418
====================================================
epoch: 6 step: 0 loss: 3.021216869354248 lr: 0.1
epoch: 6 step: 100 loss: 2.786943197250366 lr: 0.1
epoch: 6 test_acc: 0.4373
====================================================
epoch: 7 step: 0 loss: 2.837778091430664 lr: 0.1
epoch: 7 step: 100 loss: 2.605891704559326 lr: 0.1
epoch: 7 test_acc: 0.437
====================================================
epoch: 8 step: 0 loss: 2.5111215114593506 lr: 0.1
epoch: 8 step: 100 loss: 2.38578462600708 lr: 0.1
epoch: 8 test_acc: 0.4019
====================================================
epoch: 9 step: 0 loss: 2.338027238845825 lr: 0.1
epoch: 9 step: 100 loss: 2.3904590606689453 lr: 0.1
epoch: 9 test_acc: 0.462
====================================================
epoch: 10 step: 0 loss: 2.166916847229004 lr: 0.1
epoch: 10 step: 100 loss: 1.9691071510314941 lr: 0.1
epoch: 10 test_acc: 0.5102
====================================================
epoch: 11 step: 0 loss: 2.0595767498016357 lr: 0.1
epoch: 11 step: 100 loss: 1.9153989553451538 lr: 0.1
epoch: 11 test_acc: 0.532
====================================================
epoch: 12 step: 0 loss: 1.8501120805740356 lr: 0.1
epoch: 12 step: 100 loss: 1.9241153001785278 lr: 0.1
epoch: 12 test_acc: 0.4521
====================================================
epoch: 13 step: 0 loss: 1.916994571685791 lr: 0.1
epoch: 13 step: 100 loss: 1.8477200269699097 lr: 0.1
epoch: 13 test_acc: 0.5565
====================================================
epoch: 14 step: 0 loss: 1.711875081062317 lr: 0.1
epoch: 14 step: 100 loss: 1.9274563789367676 lr: 0.1
epoch: 14 test_acc: 0.5078
====================================================
epoch: 15 step: 0 loss: 1.6906111240386963 lr: 0.1
epoch: 15 step: 100 loss: 1.7410812377929688 lr: 0.1
epoch: 15 test_acc: 0.5457
====================================================
epoch: 16 step: 0 loss: 1.5904935598373413 lr: 0.1
epoch: 16 step: 100 loss: 1.562569260597229 lr: 0.1
epoch: 16 test_acc: 0.5644
====================================================
epoch: 17 step: 0 loss: 1.6902556419372559 lr: 0.1
epoch: 17 step: 100 loss: 1.6318185329437256 lr: 0.1
epoch: 17 test_acc: 0.5634
====================================================
epoch: 18 step: 0 loss: 1.6303765773773193 lr: 0.1
epoch: 18 step: 100 loss: 1.538644790649414 lr: 0.1
epoch: 18 test_acc: 0.5611
====================================================
epoch: 19 step: 0 loss: 1.4814107418060303 lr: 0.1
epoch: 19 step: 100 loss: 1.3926563262939453 lr: 0.1
epoch: 19 test_acc: 0.5662
====================================================
epoch: 20 step: 0 loss: 1.4413306713104248 lr: 0.1
epoch: 20 step: 100 loss: 1.4577772617340088 lr: 0.1
epoch: 20 test_acc: 0.5886
====================================================
......
......
====================================================
epoch: 80 step: 0 loss: 0.3856082856655121 lr: 0.02
epoch: 80 step: 100 loss: 0.3806978464126587 lr: 0.02
epoch: 80 test_acc: 0.7475
====================================================
epoch: 81 step: 0 loss: 0.3794938325881958 lr: 0.02
epoch: 81 step: 100 loss: 0.3772055208683014 lr: 0.02
epoch: 81 test_acc: 0.7509
====================================================
epoch: 82 step: 0 loss: 0.37524694204330444 lr: 0.02
epoch: 82 step: 100 loss: 0.37618422508239746 lr: 0.02
epoch: 82 test_acc: 0.7508
====================================================
epoch: 83 step: 0 loss: 0.3739137053489685 lr: 0.02
epoch: 83 step: 100 loss: 0.37356191873550415 lr: 0.02
epoch: 83 test_acc: 0.7486
====================================================
epoch: 84 step: 0 loss: 0.37090277671813965 lr: 0.02
epoch: 84 step: 100 loss: 0.3684435784816742 lr: 0.02
epoch: 84 test_acc: 0.7485
====================================================
epoch: 85 step: 0 loss: 0.36540552973747253 lr: 0.02
epoch: 85 step: 100 loss: 0.36936986446380615 lr: 0.02
epoch: 85 test_acc: 0.7497
====================================================
epoch: 86 step: 0 loss: 0.36824268102645874 lr: 0.02
epoch: 86 step: 100 loss: 0.36354953050613403 lr: 0.02
epoch: 86 test_acc: 0.7498
====================================================
epoch: 87 step: 0 loss: 0.3597506880760193 lr: 0.02
epoch: 87 step: 100 loss: 0.36188676953315735 lr: 0.02
epoch: 87 test_acc: 0.7494
====================================================
epoch: 88 step: 0 loss: 0.3578110635280609 lr: 0.02
epoch: 88 step: 100 loss: 0.366219162940979 lr: 0.02
epoch: 88 test_acc: 0.7495
====================================================
epoch: 89 step: 0 loss: 0.35592514276504517 lr: 0.02
epoch: 89 step: 100 loss: 0.3578605651855469 lr: 0.02
epoch: 89 test_acc: 0.7497
====================================================
epoch: 90 step: 0 loss: 0.3519844710826874 lr: 0.02
epoch: 90 step: 100 loss: 0.3538321852684021 lr: 0.02
epoch: 90 test_acc: 0.75
====================================================
epoch: 91 step: 0 loss: 0.35011327266693115 lr: 0.02
epoch: 91 step: 100 loss: 0.3646775484085083 lr: 0.02
epoch: 91 test_acc: 0.7506
====================================================
epoch: 92 step: 0 loss: 0.34734296798706055 lr: 0.02
epoch: 92 step: 100 loss: 0.3485969305038452 lr: 0.02
epoch: 92 test_acc: 0.7509
====================================================
epoch: 93 step: 0 loss: 0.34550538659095764 lr: 0.02
epoch: 93 step: 100 loss: 0.34283071756362915 lr: 0.02
epoch: 93 test_acc: 0.7506
====================================================
epoch: 94 step: 0 loss: 0.3442630171775818 lr: 0.02
epoch: 94 step: 100 loss: 0.3417758047580719 lr: 0.02
epoch: 94 test_acc: 0.7516
====================================================
epoch: 95 step: 0 loss: 0.3394615352153778 lr: 0.02
epoch: 95 step: 100 loss: 0.33877307176589966 lr: 0.02
epoch: 95 test_acc: 0.7513
====================================================
epoch: 96 step: 0 loss: 0.33859947323799133 lr: 0.02
epoch: 96 step: 100 loss: 0.33696863055229187 lr: 0.02
epoch: 96 test_acc: 0.7494
====================================================
epoch: 97 step: 0 loss: 0.33495423197746277 lr: 0.02
epoch: 97 step: 100 loss: 0.33486634492874146 lr: 0.02
epoch: 97 test_acc: 0.7516
====================================================
epoch: 98 step: 0 loss: 0.3330698311328888 lr: 0.02
epoch: 98 step: 100 loss: 0.3340366780757904 lr: 0.02
epoch: 98 test_acc: 0.7501
====================================================
epoch: 99 step: 0 loss: 0.33288395404815674 lr: 0.02
epoch: 99 step: 100 loss: 0.329741895198822 lr: 0.02
epoch: 99 test_acc: 0.7515
====================================================
epoch: 100 step: 0 loss: 0.3289056420326233 lr: 0.02
epoch: 100 step: 100 loss: 0.32697969675064087 lr: 0.02
epoch: 100 test_acc: 0.7497
====================================================
......
......
====================================================
epoch: 496 step: 0 loss: 0.275225430727005 lr: 6e-05
epoch: 496 step: 100 loss: 0.27560004591941833 lr: 6e-05
epoch: 496 test_acc: 0.7525
====================================================
epoch: 497 step: 0 loss: 0.277438223361969 lr: 6e-05
epoch: 497 step: 100 loss: 0.2766727805137634 lr: 6e-05
epoch: 497 test_acc: 0.7524
====================================================
epoch: 498 step: 0 loss: 0.2758670747280121 lr: 6e-05
epoch: 498 step: 100 loss: 0.2739299535751343 lr: 6e-05
epoch: 498 test_acc: 0.7529
====================================================
epoch: 499 step: 0 loss: 0.27458423376083374 lr: 6e-05
epoch: 499 step: 100 loss: 0.27507463097572327 lr: 6e-05
epoch: 499 test_acc: 0.7523
====================================================

Process finished with exit code 0
```
     
     
## 参考文献

- [luuuyi/CBAM.PyTorch](https://github.com/luuuyi/CBAM.PyTorch)
- [kobiso/CBAM-keras](https://github.com/kobiso/CBAM-keras)
- [Jongchan/attention-module](https://github.com/Jongchan/attention-module/tree/master/MODELS)
