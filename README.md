## CBAM-TensorFlow2.0

- CBAM(Convolutional Block Attention Module) implementation on TensowFlow2.0
- <font color=black> 本论文来自ECCV2018，主要在传统的CNN上引入通道注意力机制和空间注意力机制提升性能。论文地址：[CBAM！](https://arxiv.org/abs/1807.06521)

## 一. 论文摘要
<font   color=black> 本文提出了卷积块注意模块(CBAM)，这是一个简单而有效的前馈卷积神经网络注意模块。在给定中间特征图的情况下，我们的模块按照通道和空间两个独立的维度依次推断注意图，然后将注意图乘入输入特征图进行自适应特征细化。因为CBAM是一个轻量级的通用模块，它可以无缝地集成到任何CNN架构中，开销可以忽略不计，并且可以与基本CNNs一起进行端到端的培训。我们通过在ImageNet-1K、MS COCO检测和VOC 2007检测数据集上的大量实验来验证我们的CBAM。我们的实验表明，在不同的模型下，分类和检测性能都得到了一致的提高，说明了CBAM的广泛适用性。

## 二. CBAM的网络结构
### 2.1. 总体的描述
<font   color=black> 对于一个中间层的<a href="https://www.codecogs.com/eqnedit.php?latex=$\mathbf{F}&space;\in&space;\mathbb{R}^{C&space;\times&space;H&space;\times&space;W}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathbf{F}&space;\in&space;\mathbb{R}^{C&space;\times&space;H&space;\times&space;W}$" title="$\mathbf{F} \in \mathbb{R}^{C \times H \times W}$" /></a>，CBAM将会顺序推理出1维的channel attention map $\mathbf M_{c} \in \mathbb{R}^{C  \times 1  \times 1}$ 以及2维的spatial attention map $\mathbf M_{s} \in \mathbb{R}^{1 \times H  \times W}$，整个过程如下所示：
<font color=black> $$
\mathbf{F}^{\prime}=\mathbf{M}_{\mathbf{c}}(\mathbf{F}) \otimes \mathbf{F}\tag{1}
$$ $$
\mathbf{F}^{\prime \prime}=\mathbf{M}_{\mathbf{s}}\left(\mathbf{F}^{\prime}\right) \otimes \mathbf{F}^{\prime}\tag{2}
$$ **其中：** $⊗$为element-wise multiplication，首先将channel attention map与输入的feature map相乘得到 $\mathbf{F}^{\prime}$， 之后计算 $\mathbf{F}^{\prime}$ 的spatial attention map，并将两者相乘得到最终的输出 $\mathbf{F}^{\prime \prime}$。

<center><image src="https://img-blog.csdnimg.cn/20200102234830810.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FiYzEzNTI2MjIyMTYw,size_16,color_FFFFFF,t_70" width="100%">
     
    
### 2.2. 通道注意力机制
<font   color=black> 首先是通道注意力，我们知道一张图片经过几个卷积层会得到一个特征矩阵，这个矩阵的通道数就是卷积层核的个数。那么，一个常见的卷积核经常达到1024，2048个，并不是每个通道都对于信息传递非常有用了的。因此，通过对这些通道进行过滤，也就是注意，来得到优化后的特征。
<font   color=black>**主要思路就是：增大有效通道权重，减少无效通道的权重。** 公式表示为如下：$$
\begin{aligned}
\mathbf{M}_{\mathbf{c}}(\mathbf{F}) &=\sigma(\text{MLP(AvgPool}(\mathbf{F}))+\text{MLP}(\operatorname{MaxPool} (\mathbf{F}))) \\
&=\sigma\left(\mathbf{W}_{\mathbf{1}}(\mathbf{W}_{\mathbf{0}}(\mathbf{F}_{\text {avg }}^{\mathbf{c}}))+\mathbf{W}_{\mathbf{1}}\left(\mathbf{W}_{\mathbf{0}}\left(\mathbf{F}_{\max }^{\mathbf{c}}\right)\right)\right)\tag{3}
\end{aligned}
$$ **其中：** $\mathbf{F}_{\text {avg}}^\mathbf{c}$ 和 $\mathbf{F}_{\text {max}}^\mathbf{c}$ 表示对feature map在空间维度上使用**最大池化**和**平均池化**。$\mathbf{W}_{0} \in \mathbb{R}^{C / r  \times C}, \quad \mathbf{W}_{1} \in \mathbb{R}^{C  \times C / r}$，$\mathbf{W}_{0}$ 后使用了Relu作为激活函数，<font   color=blue>$\sigma$ 表示**Sigmoid**函数</font>。
 **此外：** 共享网络是由一个隐藏层和多层感知机(MLP)组成。为了减少参数开销，隐藏的激活大小设置为 $\mathbb{R}^{C / r \times 1 \times 1}$，其中 $r$ 是压缩率。在将共享网络应用于矢量之后，我们使用**逐元素求和**来合并输出特征向量。


<center><image src="https://img-blog.csdnimg.cn/2020010223493014.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FiYzEzNTI2MjIyMTYw,size_16,color_FFFFFF,t_70" width="100%">
   
  
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

<center><image src="https://img-blog.csdnimg.cn/20200102235023296.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FiYzEzNTI2MjIyMTYw,size_16,color_FFFFFF,t_70" width="100%">

<font   color=black>**注意：** 这里同样使用了avg-pooling和max-pooling来对信息进行评估，使用一个 $7×7$ 的卷积来进行提取。注意权重都通过<font   color=blue>**sigmoid**来进行归一化</font>。

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
