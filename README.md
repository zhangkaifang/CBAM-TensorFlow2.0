## CBAM-TensorFlow2.0

- CBAM(Convolutional Block Attention Module) implementation on TensowFlow2.0
- <font color=black> 本论文来自ECCV2018，主要在传统的CNN上引入通道注意力机制和空间注意力机制提升性能。论文地址：[CBAM！](https://arxiv.org/abs/1807.06521)

## 一. 论文摘要
<font   color=black> 本文提出了卷积块注意模块(CBAM)，这是一个简单而有效的前馈卷积神经网络注意模块。在给定中间特征图的情况下，我们的模块按照通道和空间两个独立的维度依次推断注意图，然后将注意图乘入输入特征图进行自适应特征细化。因为CBAM是一个轻量级的通用模块，它可以无缝地集成到任何CNN架构中，开销可以忽略不计，并且可以与基本CNNs一起进行端到端的培训。我们通过在ImageNet-1K、MS COCO检测和VOC 2007检测数据集上的大量实验来验证我们的CBAM。我们的实验表明，在不同的模型下，分类和检测性能都得到了一致的提高，说明了CBAM的广泛适用性。

## 二. CBAM的网络结构
### 2.1. 总体的描述
<font   color=black> 对于一个中间层的$\mathbf F \in \mathbb{R}^{C  \times H \times W}$，CBAM将会顺序推理出1维的channel attention map $\mathbf M_{c} \in \mathbb{R}^{C  \times 1  \times 1}$ 以及2维的spatial attention map $\mathbf M_{s} \in \mathbb{R}^{1 \times H  \times W}$，整个过程如下所示：
<font color=black> $$
\mathbf{F}^{\prime}=\mathbf{M}_{\mathbf{c}}(\mathbf{F}) \otimes \mathbf{F}\tag{1}
$$ $$
\mathbf{F}^{\prime \prime}=\mathbf{M}_{\mathbf{s}}\left(\mathbf{F}^{\prime}\right) \otimes \mathbf{F}^{\prime}\tag{2}
$$ **其中：** $⊗$为element-wise multiplication，首先将channel attention map与输入的feature map相乘得到 $\mathbf{F}^{\prime}$， 之后计算 $\mathbf{F}^{\prime}$ 的spatial attention map，并将两者相乘得到最终的输出 $\mathbf{F}^{\prime \prime}$。

<center><image src="https://github.com/kobiso/CBAM-keras/blob/master/figures/overview.png?raw=true" width="100%">
     

### 2.2. 通道注意力机制
<font   color=black> 首先是通道注意力，我们知道一张图片经过几个卷积层会得到一个特征矩阵，这个矩阵的通道数就是卷积层核的个数。那么，一个常见的卷积核经常达到1024，2048个，并不是每个通道都对于信息传递非常有用了的。因此，通过对这些通道进行过滤，也就是注意，来得到优化后的特征。
<font   color=black>**主要思路就是：增大有效通道权重，减少无效通道的权重。** 公式表示为如下：$$
\begin{aligned}
\mathbf{M}_{\mathbf{c}}(\mathbf{F}) &=\sigma(\text{MLP(AvgPool}(\mathbf{F}))+\text{MLP}(\operatorname{MaxPool} (\mathbf{F}))) \\
&=\sigma\left(\mathbf{W}_{\mathbf{1}}(\mathbf{W}_{\mathbf{0}}(\mathbf{F}_{\text {avg }}^{\mathbf{c}}))+\mathbf{W}_{\mathbf{1}}\left(\mathbf{W}_{\mathbf{0}}\left(\mathbf{F}_{\max }^{\mathbf{c}}\right)\right)\right)\tag{3}
\end{aligned}
$$ **其中：** $\mathbf{F}_{\text {avg}}^\mathbf{c}$ 和 $\mathbf{F}_{\text {max}}^\mathbf{c}$ 表示对feature map在空间维度上使用**最大池化**和**平均池化**。$\mathbf{W}_{0} \in \mathbb{R}^{C / r  \times C}, \quad \mathbf{W}_{1} \in \mathbb{R}^{C  \times C / r}$，$\mathbf{W}_{0}$ 后使用了Relu作为激活函数，<font   color=blue>$\sigma$ 表示**Sigmoid**函数</font>。
 **此外：** 共享网络是由一个隐藏层和多层感知机(MLP)组成。为了减少参数开销，隐藏的激活大小设置为 $\mathbb{R}^{C / r \times 1 \times 1}$，其中 $r$ 是压缩率。在将共享网络应用于矢量之后，我们使用**逐元素求和**来合并输出特征向量。

  <center><image src="https://img-blog.csdnimg.cn/20191230145340134.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FiYzEzNTI2MjIyMTYw,size_16,color_FFFFFF,t_70" width="100%">
  
<font   color=black>**注意：** 这里非常像SENet，SENet在很多论文中都被证实对效果有提升，这里的区别是，SENet采用的是平均值的pooling，这篇论文又加入了最大值pooling。作者在论文中，通过对比实验，证实max pooling提高了效果。这里的mlp的中间层较小，这个可能有助于信息的整合。

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
<font   color=black> 论文中，作者认为通道注意力关注的是：what，然而空间注意力关注的是：where。
<font   color=black>$$
\begin{aligned}
\mathbf{M}_{\mathbf{s}}(\mathbf{F}) &=\sigma\left(f^{7 \times 7}(\left[\text {AvgPool}(\mathbf{F}) ; \text {MaxPool}(\mathbf{F})]\right))\right) \\
&=\sigma\left(f^{7 \times 7}\left(\left[\mathbf{F}_{\text {avg }}^{\mathrm{s}} ; \mathbf{F}_{\text {max }}^{\mathrm{s}}\right]\right)\right)\tag{4}
\end{aligned}
$$


<font   color=black>**注意：** 这里同样使用了avg-pooling和max-pooling来对信息进行评估，使用一个 $7×7$ 的卷积来进行提取。注意权重都通过<font   color=blue>**sigmoid**来进行归一化</font>。

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
### 3.4. CIFAR100测试结果(CBAM)

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
epoch: 21 step: 0 loss: 1.4243451356887817 lr: 0.1
epoch: 21 step: 100 loss: 1.4382528066635132 lr: 0.1
epoch: 21 test_acc: 0.5575
====================================================
epoch: 22 step: 0 loss: 1.343896508216858 lr: 0.1
epoch: 22 step: 100 loss: 1.2740988731384277 lr: 0.1
epoch: 22 test_acc: 0.5836
====================================================
epoch: 23 step: 0 loss: 1.467741847038269 lr: 0.1
epoch: 23 step: 100 loss: 1.2149549722671509 lr: 0.1
epoch: 23 test_acc: 0.5978
====================================================
epoch: 24 step: 0 loss: 1.2722827196121216 lr: 0.1
epoch: 24 step: 100 loss: 1.2558863162994385 lr: 0.1
epoch: 24 test_acc: 0.588
====================================================
epoch: 25 step: 0 loss: 1.249694585800171 lr: 0.1
epoch: 25 step: 100 loss: 1.2405654191970825 lr: 0.1
epoch: 25 test_acc: 0.6004
====================================================
epoch: 26 step: 0 loss: 1.285285472869873 lr: 0.1
epoch: 26 step: 100 loss: 1.10854172706604 lr: 0.1
epoch: 26 test_acc: 0.5974
====================================================
epoch: 27 step: 0 loss: 1.208528757095337 lr: 0.1
epoch: 27 step: 100 loss: 1.078981876373291 lr: 0.1
epoch: 27 test_acc: 0.5564
====================================================
epoch: 28 step: 0 loss: 1.1439024209976196 lr: 0.1
epoch: 28 step: 100 loss: 1.125194787979126 lr: 0.1
epoch: 28 test_acc: 0.614
====================================================
epoch: 29 step: 0 loss: 1.165439248085022 lr: 0.1
epoch: 29 step: 100 loss: 1.063378930091858 lr: 0.1
epoch: 29 test_acc: 0.5963
====================================================
epoch: 30 step: 0 loss: 1.119661569595337 lr: 0.1
epoch: 30 step: 100 loss: 1.0098509788513184 lr: 0.1
epoch: 30 test_acc: 0.6236
====================================================
epoch: 31 step: 0 loss: 0.9726322889328003 lr: 0.1
epoch: 31 step: 100 loss: 0.9754623174667358 lr: 0.1
epoch: 31 test_acc: 0.6082
====================================================
epoch: 32 step: 0 loss: 1.071475625038147 lr: 0.1
epoch: 32 step: 100 loss: 0.9686889052391052 lr: 0.1
epoch: 32 test_acc: 0.6259
====================================================
epoch: 33 step: 0 loss: 1.0011951923370361 lr: 0.1
epoch: 33 step: 100 loss: 0.9600554704666138 lr: 0.1
epoch: 33 test_acc: 0.6268
====================================================
epoch: 34 step: 0 loss: 1.0018489360809326 lr: 0.1
epoch: 34 step: 100 loss: 1.0467039346694946 lr: 0.1
epoch: 34 test_acc: 0.5789
====================================================
epoch: 35 step: 0 loss: 0.9482053518295288 lr: 0.1
epoch: 35 step: 100 loss: 0.953992486000061 lr: 0.1
epoch: 35 test_acc: 0.6244
====================================================
epoch: 36 step: 0 loss: 0.8945754170417786 lr: 0.1
epoch: 36 step: 100 loss: 0.9210041761398315 lr: 0.1
epoch: 36 test_acc: 0.6191
====================================================
epoch: 37 step: 0 loss: 0.8879575729370117 lr: 0.1
epoch: 37 step: 100 loss: 0.8896430730819702 lr: 0.1
epoch: 37 test_acc: 0.6297
====================================================
epoch: 38 step: 0 loss: 0.8314822316169739 lr: 0.1
epoch: 38 step: 100 loss: 0.8576794862747192 lr: 0.1
epoch: 38 test_acc: 0.6224
====================================================
epoch: 39 step: 0 loss: 0.8786734342575073 lr: 0.1
epoch: 39 step: 100 loss: 0.8197131156921387 lr: 0.1
epoch: 39 test_acc: 0.6321
====================================================
epoch: 40 step: 0 loss: 0.8550319671630859 lr: 0.1
epoch: 40 step: 100 loss: 0.8329888582229614 lr: 0.1
epoch: 40 test_acc: 0.6246
====================================================
epoch: 41 step: 0 loss: 0.7935174703598022 lr: 0.1
epoch: 41 step: 100 loss: 0.8616297841072083 lr: 0.1
epoch: 41 test_acc: 0.6559
====================================================
epoch: 42 step: 0 loss: 0.8157148957252502 lr: 0.1
epoch: 42 step: 100 loss: 0.8170034289360046 lr: 0.1
epoch: 42 test_acc: 0.6566
====================================================
epoch: 43 step: 0 loss: 0.7272849678993225 lr: 0.1
epoch: 43 step: 100 loss: 0.7370797395706177 lr: 0.1
epoch: 43 test_acc: 0.6441
====================================================
epoch: 44 step: 0 loss: 0.7465075254440308 lr: 0.1
epoch: 44 step: 100 loss: 0.7382514476776123 lr: 0.1
epoch: 44 test_acc: 0.6369
====================================================
epoch: 45 step: 0 loss: 0.8375297784805298 lr: 0.1
epoch: 45 step: 100 loss: 0.7834970951080322 lr: 0.1
epoch: 45 test_acc: 0.6439
====================================================
epoch: 46 step: 0 loss: 0.6721677184104919 lr: 0.1
epoch: 46 step: 100 loss: 0.708148717880249 lr: 0.1
epoch: 46 test_acc: 0.6638
====================================================
epoch: 47 step: 0 loss: 0.685032308101654 lr: 0.1
epoch: 47 step: 100 loss: 0.7191689610481262 lr: 0.1
epoch: 47 test_acc: 0.6448
====================================================
epoch: 48 step: 0 loss: 0.7170189619064331 lr: 0.1
epoch: 48 step: 100 loss: 0.6866345405578613 lr: 0.1
epoch: 48 test_acc: 0.6579
====================================================
epoch: 49 step: 0 loss: 0.7102372646331787 lr: 0.1
epoch: 49 step: 100 loss: 0.7244817018508911 lr: 0.1
epoch: 49 test_acc: 0.6497
====================================================
epoch: 50 step: 0 loss: 0.6929795742034912 lr: 0.1
epoch: 50 step: 100 loss: 0.6529719829559326 lr: 0.1
epoch: 50 test_acc: 0.6522
====================================================
epoch: 51 step: 0 loss: 0.6795995235443115 lr: 0.1
epoch: 51 step: 100 loss: 0.6251389980316162 lr: 0.1
epoch: 51 test_acc: 0.6584
====================================================
epoch: 52 step: 0 loss: 0.6453678607940674 lr: 0.1
epoch: 52 step: 100 loss: 0.6200178265571594 lr: 0.1
epoch: 52 test_acc: 0.6605
====================================================
epoch: 53 step: 0 loss: 0.5963259339332581 lr: 0.1
epoch: 53 step: 100 loss: 0.62004554271698 lr: 0.1
epoch: 53 test_acc: 0.6529
====================================================
epoch: 54 step: 0 loss: 0.6910814046859741 lr: 0.1
epoch: 54 step: 100 loss: 0.5933144092559814 lr: 0.1
epoch: 54 test_acc: 0.6579
====================================================
epoch: 55 step: 0 loss: 0.6343454122543335 lr: 0.1
epoch: 55 step: 100 loss: 0.6011928915977478 lr: 0.1
epoch: 55 test_acc: 0.6615
====================================================
epoch: 56 step: 0 loss: 0.6087968945503235 lr: 0.1
epoch: 56 step: 100 loss: 0.6308256983757019 lr: 0.1
epoch: 56 test_acc: 0.6609
====================================================
epoch: 57 step: 0 loss: 0.629458487033844 lr: 0.1
epoch: 57 step: 100 loss: 0.561354398727417 lr: 0.1
epoch: 57 test_acc: 0.6615
====================================================
epoch: 58 step: 0 loss: 0.5572633743286133 lr: 0.1
epoch: 58 step: 100 loss: 0.560829758644104 lr: 0.1
epoch: 58 test_acc: 0.6423
====================================================
epoch: 59 step: 0 loss: 0.5535402894020081 lr: 0.1
epoch: 59 step: 100 loss: 0.5763798952102661 lr: 0.1
epoch: 59 test_acc: 0.6667
====================================================
epoch: 60 step: 0 loss: 0.5497308969497681 lr: 0.02
epoch: 60 step: 100 loss: 0.49834513664245605 lr: 0.02
epoch: 60 test_acc: 0.7317
====================================================
epoch: 61 step: 0 loss: 0.46500396728515625 lr: 0.02
epoch: 61 step: 100 loss: 0.4765871465206146 lr: 0.02
epoch: 61 test_acc: 0.7388
====================================================
epoch: 62 step: 0 loss: 0.45633265376091003 lr: 0.02
epoch: 62 step: 100 loss: 0.4682161509990692 lr: 0.02
epoch: 62 test_acc: 0.7409
====================================================
epoch: 63 step: 0 loss: 0.4585873782634735 lr: 0.02
epoch: 63 step: 100 loss: 0.46889346837997437 lr: 0.02
epoch: 63 test_acc: 0.7422
====================================================
epoch: 64 step: 0 loss: 0.4585932791233063 lr: 0.02
epoch: 64 step: 100 loss: 0.44502490758895874 lr: 0.02
epoch: 64 test_acc: 0.7402
====================================================
epoch: 65 step: 0 loss: 0.4441959261894226 lr: 0.02
epoch: 65 step: 100 loss: 0.4461227059364319 lr: 0.02
epoch: 65 test_acc: 0.7428
====================================================
epoch: 66 step: 0 loss: 0.43956106901168823 lr: 0.02
epoch: 66 step: 100 loss: 0.45719292759895325 lr: 0.02
epoch: 66 test_acc: 0.7425
====================================================
epoch: 67 step: 0 loss: 0.4408290982246399 lr: 0.02
epoch: 67 step: 100 loss: 0.42879435420036316 lr: 0.02
epoch: 67 test_acc: 0.7461
====================================================
epoch: 68 step: 0 loss: 0.4296460747718811 lr: 0.02
epoch: 68 step: 100 loss: 0.4287627637386322 lr: 0.02
epoch: 68 test_acc: 0.7455
====================================================
epoch: 69 step: 0 loss: 0.42484211921691895 lr: 0.02
epoch: 69 step: 100 loss: 0.4245947003364563 lr: 0.02
epoch: 69 test_acc: 0.7466
====================================================
epoch: 70 step: 0 loss: 0.4221042990684509 lr: 0.02
epoch: 70 step: 100 loss: 0.4168050289154053 lr: 0.02
epoch: 70 test_acc: 0.7454
====================================================
epoch: 71 step: 0 loss: 0.41682755947113037 lr: 0.02
epoch: 71 step: 100 loss: 0.4150591194629669 lr: 0.02
epoch: 71 test_acc: 0.7468
====================================================
epoch: 72 step: 0 loss: 0.4119566082954407 lr: 0.02
epoch: 72 step: 100 loss: 0.41274455189704895 lr: 0.02
epoch: 72 test_acc: 0.7455
====================================================
epoch: 73 step: 0 loss: 0.41112929582595825 lr: 0.02
epoch: 73 step: 100 loss: 0.4079146385192871 lr: 0.02
epoch: 73 test_acc: 0.7475
====================================================
epoch: 74 step: 0 loss: 0.4024839699268341 lr: 0.02
epoch: 74 step: 100 loss: 0.4047936797142029 lr: 0.02
epoch: 74 test_acc: 0.7482
====================================================
epoch: 75 step: 0 loss: 0.3998277485370636 lr: 0.02
epoch: 75 step: 100 loss: 0.4003147482872009 lr: 0.02
epoch: 75 test_acc: 0.7503
====================================================
epoch: 76 step: 0 loss: 0.39820605516433716 lr: 0.02
epoch: 76 step: 100 loss: 0.3948240578174591 lr: 0.02
epoch: 76 test_acc: 0.7489
====================================================
epoch: 77 step: 0 loss: 0.4108869135379791 lr: 0.02
epoch: 77 step: 100 loss: 0.39124470949172974 lr: 0.02
epoch: 77 test_acc: 0.7472
====================================================
epoch: 78 step: 0 loss: 0.38836920261383057 lr: 0.02
epoch: 78 step: 100 loss: 0.391308456659317 lr: 0.02
epoch: 78 test_acc: 0.7485
====================================================
epoch: 79 step: 0 loss: 0.3871617317199707 lr: 0.02
epoch: 79 step: 100 loss: 0.383708119392395 lr: 0.02
epoch: 79 test_acc: 0.7488
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
epoch: 101 step: 0 loss: 0.327077716588974 lr: 0.02
epoch: 101 step: 100 loss: 0.32603633403778076 lr: 0.02
epoch: 101 test_acc: 0.7514
====================================================
epoch: 102 step: 0 loss: 0.3257395923137665 lr: 0.02
epoch: 102 step: 100 loss: 0.3280397355556488 lr: 0.02
epoch: 102 test_acc: 0.7523
====================================================
epoch: 103 step: 0 loss: 0.32568296790122986 lr: 0.02
epoch: 103 step: 100 loss: 0.3250931203365326 lr: 0.02
epoch: 103 test_acc: 0.7517
====================================================
epoch: 104 step: 0 loss: 0.32200348377227783 lr: 0.02
epoch: 104 step: 100 loss: 0.3203653395175934 lr: 0.02
epoch: 104 test_acc: 0.7503
====================================================
epoch: 105 step: 0 loss: 0.32009026408195496 lr: 0.02
epoch: 105 step: 100 loss: 0.31660938262939453 lr: 0.02
epoch: 105 test_acc: 0.7511
====================================================
epoch: 106 step: 0 loss: 0.31745535135269165 lr: 0.02
epoch: 106 step: 100 loss: 0.314359188079834 lr: 0.02
epoch: 106 test_acc: 0.7512
====================================================
epoch: 107 step: 0 loss: 0.31396955251693726 lr: 0.02
epoch: 107 step: 100 loss: 0.3138728737831116 lr: 0.02
epoch: 107 test_acc: 0.7519
====================================================
epoch: 108 step: 0 loss: 0.3108638525009155 lr: 0.02
epoch: 108 step: 100 loss: 0.3098585307598114 lr: 0.02
epoch: 108 test_acc: 0.7533
====================================================
epoch: 109 step: 0 loss: 0.3136166036128998 lr: 0.02
epoch: 109 step: 100 loss: 0.310923308134079 lr: 0.02
epoch: 109 test_acc: 0.7518
====================================================
epoch: 110 step: 0 loss: 0.30647367238998413 lr: 0.02
epoch: 110 step: 100 loss: 0.3057536780834198 lr: 0.02
epoch: 110 test_acc: 0.7513
====================================================
epoch: 111 step: 0 loss: 0.30533990263938904 lr: 0.02
epoch: 111 step: 100 loss: 0.3045988380908966 lr: 0.02
epoch: 111 test_acc: 0.7508
====================================================
epoch: 112 step: 0 loss: 0.3043580949306488 lr: 0.02
epoch: 112 step: 100 loss: 0.3046347498893738 lr: 0.02
epoch: 112 test_acc: 0.7507
====================================================
epoch: 113 step: 0 loss: 0.3019658327102661 lr: 0.02
epoch: 113 step: 100 loss: 0.30040839314460754 lr: 0.02
epoch: 113 test_acc: 0.752
====================================================
epoch: 114 step: 0 loss: 0.2995662987232208 lr: 0.02
epoch: 114 step: 100 loss: 0.3107931315898895 lr: 0.02
epoch: 114 test_acc: 0.752
====================================================
epoch: 115 step: 0 loss: 0.2986544072628021 lr: 0.02
epoch: 115 step: 100 loss: 0.2987724542617798 lr: 0.02
epoch: 115 test_acc: 0.7541
====================================================
epoch: 116 step: 0 loss: 0.29660871624946594 lr: 0.02
epoch: 116 step: 100 loss: 0.30207592248916626 lr: 0.02
epoch: 116 test_acc: 0.7522
====================================================
epoch: 117 step: 0 loss: 0.2940964698791504 lr: 0.02
epoch: 117 step: 100 loss: 0.30191338062286377 lr: 0.02
epoch: 117 test_acc: 0.7541
====================================================
epoch: 118 step: 0 loss: 0.293407142162323 lr: 0.02
epoch: 118 step: 100 loss: 0.2927250266075134 lr: 0.02
epoch: 118 test_acc: 0.753
====================================================
epoch: 119 step: 0 loss: 0.29183921217918396 lr: 0.02
epoch: 119 step: 100 loss: 0.28980520367622375 lr: 0.02
epoch: 119 test_acc: 0.7523
====================================================
epoch: 120 step: 0 loss: 0.2893623411655426 lr: 0.004
epoch: 120 step: 100 loss: 0.28972864151000977 lr: 0.004
epoch: 120 test_acc: 0.7516
====================================================
epoch: 121 step: 0 loss: 0.2897017002105713 lr: 0.004
epoch: 121 step: 100 loss: 0.29064151644706726 lr: 0.004
epoch: 121 test_acc: 0.7523
====================================================
epoch: 122 step: 0 loss: 0.28871580958366394 lr: 0.004
epoch: 122 step: 100 loss: 0.28919002413749695 lr: 0.004
epoch: 122 test_acc: 0.7524
====================================================
epoch: 123 step: 0 loss: 0.2896999716758728 lr: 0.004
epoch: 123 step: 100 loss: 0.28755539655685425 lr: 0.004
epoch: 123 test_acc: 0.7531
====================================================
epoch: 124 step: 0 loss: 0.2878280282020569 lr: 0.004
epoch: 124 step: 100 loss: 0.2968134880065918 lr: 0.004
epoch: 124 test_acc: 0.7522
====================================================
epoch: 125 step: 0 loss: 0.2875559628009796 lr: 0.004
epoch: 125 step: 100 loss: 0.2917034327983856 lr: 0.004
epoch: 125 test_acc: 0.7526
====================================================
epoch: 126 step: 0 loss: 0.28771722316741943 lr: 0.004
epoch: 126 step: 100 loss: 0.2885883152484894 lr: 0.004
epoch: 126 test_acc: 0.7528
====================================================
epoch: 127 step: 0 loss: 0.2862972617149353 lr: 0.004
epoch: 127 step: 100 loss: 0.28680601716041565 lr: 0.004
epoch: 127 test_acc: 0.7527
====================================================
epoch: 128 step: 0 loss: 0.2872631251811981 lr: 0.004
epoch: 128 step: 100 loss: 0.2872883975505829 lr: 0.004
epoch: 128 test_acc: 0.7532
====================================================
epoch: 129 step: 0 loss: 0.28729361295700073 lr: 0.004
epoch: 129 step: 100 loss: 0.2894335389137268 lr: 0.004
epoch: 129 test_acc: 0.7532
====================================================
epoch: 130 step: 0 loss: 0.28647589683532715 lr: 0.004
epoch: 130 step: 100 loss: 0.2868848443031311 lr: 0.004
epoch: 130 test_acc: 0.7525
====================================================
epoch: 131 step: 0 loss: 0.2869456112384796 lr: 0.004
epoch: 131 step: 100 loss: 0.28505727648735046 lr: 0.004
epoch: 131 test_acc: 0.7527
====================================================
epoch: 132 step: 0 loss: 0.2880209684371948 lr: 0.004
epoch: 132 step: 100 loss: 0.2872430980205536 lr: 0.004
epoch: 132 test_acc: 0.752
====================================================
epoch: 133 step: 0 loss: 0.2860906720161438 lr: 0.004
epoch: 133 step: 100 loss: 0.2852298319339752 lr: 0.004
epoch: 133 test_acc: 0.7528
====================================================
epoch: 134 step: 0 loss: 0.28615301847457886 lr: 0.004
epoch: 134 step: 100 loss: 0.28449681401252747 lr: 0.004
epoch: 134 test_acc: 0.7524
====================================================
epoch: 135 step: 0 loss: 0.28529056906700134 lr: 0.004
epoch: 135 step: 100 loss: 0.28432750701904297 lr: 0.004
epoch: 135 test_acc: 0.7523
====================================================
epoch: 136 step: 0 loss: 0.28320664167404175 lr: 0.004
epoch: 136 step: 100 loss: 0.28349512815475464 lr: 0.004
epoch: 136 test_acc: 0.7527
====================================================
epoch: 137 step: 0 loss: 0.2843083143234253 lr: 0.004
epoch: 137 step: 100 loss: 0.2860974967479706 lr: 0.004
epoch: 137 test_acc: 0.7529
====================================================
epoch: 138 step: 0 loss: 0.28563129901885986 lr: 0.004
epoch: 138 step: 100 loss: 0.28444647789001465 lr: 0.004
epoch: 138 test_acc: 0.7521
====================================================
epoch: 139 step: 0 loss: 0.283224880695343 lr: 0.004
epoch: 139 step: 100 loss: 0.2842170000076294 lr: 0.004
epoch: 139 test_acc: 0.7523
====================================================
epoch: 140 step: 0 loss: 0.2838503122329712 lr: 0.004
epoch: 140 step: 100 loss: 0.2837461531162262 lr: 0.004
epoch: 140 test_acc: 0.7524
====================================================
epoch: 141 step: 0 loss: 0.2830551564693451 lr: 0.004
epoch: 141 step: 100 loss: 0.2828311324119568 lr: 0.004
epoch: 141 test_acc: 0.7527
====================================================
epoch: 142 step: 0 loss: 0.2824473977088928 lr: 0.004
epoch: 142 step: 100 loss: 0.28366684913635254 lr: 0.004
epoch: 142 test_acc: 0.7527
====================================================
epoch: 143 step: 0 loss: 0.28499048948287964 lr: 0.004
epoch: 143 step: 100 loss: 0.287776380777359 lr: 0.004
epoch: 143 test_acc: 0.752
====================================================
epoch: 144 step: 0 loss: 0.2817365527153015 lr: 0.004
epoch: 144 step: 100 loss: 0.28316423296928406 lr: 0.004
epoch: 144 test_acc: 0.7518
====================================================
epoch: 145 step: 0 loss: 0.28208139538764954 lr: 0.004
epoch: 145 step: 100 loss: 0.28433191776275635 lr: 0.004
epoch: 145 test_acc: 0.7525
====================================================
epoch: 146 step: 0 loss: 0.2819580137729645 lr: 0.004
epoch: 146 step: 100 loss: 0.2833653688430786 lr: 0.004
epoch: 146 test_acc: 0.7524
====================================================
epoch: 147 step: 0 loss: 0.2829534709453583 lr: 0.004
epoch: 147 step: 100 loss: 0.2803763449192047 lr: 0.004
epoch: 147 test_acc: 0.7527
====================================================
epoch: 148 step: 0 loss: 0.2825709581375122 lr: 0.004
epoch: 148 step: 100 loss: 0.2812015414237976 lr: 0.004
epoch: 148 test_acc: 0.7529
====================================================
epoch: 149 step: 0 loss: 0.281236857175827 lr: 0.004
epoch: 149 step: 100 loss: 0.28030893206596375 lr: 0.004
epoch: 149 test_acc: 0.7529
====================================================
epoch: 150 step: 0 loss: 0.2806352972984314 lr: 0.004
epoch: 150 step: 100 loss: 0.2805061638355255 lr: 0.004
epoch: 150 test_acc: 0.7535
====================================================
epoch: 151 step: 0 loss: 0.280410498380661 lr: 0.004
epoch: 151 step: 100 loss: 0.2804100811481476 lr: 0.004
epoch: 151 test_acc: 0.7536
====================================================
epoch: 152 step: 0 loss: 0.2804739773273468 lr: 0.004
epoch: 152 step: 100 loss: 0.2800332307815552 lr: 0.004
epoch: 152 test_acc: 0.7532
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
