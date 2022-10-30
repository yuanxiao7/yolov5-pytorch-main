# 注意力集中机制

### 笔记出处   

-   https://blog.csdn.net/weixin_44791964/article/details/121371986       **B导，优质导师！**



## 理论理解

对于注意力集中机制的理论理解，建议去听B站范仁义的讲解，然后再来听B导讲课 (* ^_^ *)

---



## 空间注意力集中机制

空间注意力中机制，他会关注特征层上的像素点，如有一只鸟，他就会自适应关注鸟的区域，从而增加那一区域的权重，权重变得更大，后面网络训练时会更加关注那只鸟的区域。



## 通道注意力集中机制

高维高语义特征，可能特征层的一些明暗信息、色域信息的特征对检测不是很重要，然后自适应通道注意力集中机制就会自适应的决定那一部分的特征是重要的，那一部分是不重要的，然后再去关注重要的特征的高语义特征。

---



### SENet

SENet是通道注意力集中机制的典型实现，对于输入的特征层，关注的是每一个通道的权重，即SENet可以使网络关注网络最需要关注的通道。

实现步骤

1. 对输入进来的特征层进行全局平均池化，改变维度。
2. 进行两次全连接，第一次全连接神经元少于输入特征层（具体的个数是根据实践尝试出来的），第二次全连接神经元的个数和输入特征层相同，每一次全连接后都用relu激活函数（两个FC层的设计是为了捕捉非线性的跨通道交互，其中包括降维来控制模型的复杂性。）。
3. 完成全连接的特征提取后，用sigmoid归一化到0-1之间，改变维度，获得输入特征层每一个通道的权值。
4. 最后将这个权值点乘原输入特征层即可（个人认为加上比乘上的效果好，后面我会按我的想法将他改进）。

![img](https://img-blog.csdnimg.cn/20201124130209827.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)

具体代码实现

```python
import torch
import torch.nn as nn
import math

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):  # 论文实验得到的，有2， 8， 16， 32
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 1 表示宽高为1
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

```





### CBAM

CBAM将通道注意力机制和空间注意力机制进行一个结合，自然有两个注意力机制的优势，一般而言，比SENet可以取得更好的效果。CBAM会对输入进来的特征图进行通道注意力集中，在进行空间注意力集中，最后得出权值。

简单理解如下图

![img](https://img-blog.csdnimg.cn/20201124133821606.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)



细节理解内部实现

上图可分为分为前半部分和后半部分，而前半部分通道注意力集中机制又细分为两部分，即对输入进来的单个特征层，分别进行全局平均池化和全局最大池化，再对最大池化和平均池化的结果，利用共享全连接层进行处理，之后对分别处理得到的两个结果进行相加，最后利用sigmoid进行归一化（0-1之间）得到输入特征层的每一个通道的权值，再将权值和原输入特征层相乘即可。具体如下图所示。

![image-20220801095758050](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220801095758050.png)



后半部分为空间注意力集中机制，对于输入进来的特征层，在每一个特征点的通道上取得最大值和平均值（这并不是池化，只是找通道上所有特征点的最大值和平均值），两个结果宽高不变，通道为1，将两个结果堆叠，利用 f 为1的卷积核调整通道数（含一定的特征提取），然后利用sigmoid归一化（0-1之间），即可得到输入特征层每一个特征点的权值，最后将这个权值乘以原输入特征层即可。具体如下图所示。

![image-20220801100810216](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220801100810216.png)



代码实现如下

```python
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1   # 可以写为 padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

```





### ECA

ECANet可以看作SENet的改进版，ECANet的作者认为SENet对通道注意力机制的预测带来了副作用，捕获所有通道的依赖关系是低效且不必要的。在ECANet的论文中，作者认为卷积具有良好的跨通道信息获取能力（避免降维对于学习通道注意力非常重要，适当的跨信道交互可以在显著降低模型复杂度的同时保持性能）。**针对深度CNN的高效通道注意(ECA)模块，该模块避免了降维，有效捕获了跨通道交互的信息。**

ECA模块简洁，它去除了SE模块中的全连接层，在全局平均池化后的特征直接通过一个1D卷积进行学习。这样，1D 卷积核大小就变得非常重要了，1D 卷积核会影响注意力集中机制每个权重得计算要考虑得通道数量，即跨通道交互的覆盖率。

放图，左图是常规的SE模块，右图是ECA模块，改进处 ECA模块用1D卷积替换两次全连接处理。

![img](https://img-blog.csdnimg.cn/20201126210628785.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)



代码实现

```python
class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))  # 卷积核得计算公式  自适应通道数
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

```



### CAnet

CA原理是将输入的特征图分为宽度和高度两个 方向分别进行平均池化，分别获得在宽高和高度两个方向的特征图，接着将获得全局感受野的宽度和高度两个方向的特征图拼接在一起，将他们送入共享的 1 * 1 卷积核，将维度降为原来的C/r，然后经过批量归一化处理的特征图送进relu，再将特征图按原来的高度和宽度进行 1 * 1的卷分别得到与原来一样的特征图，在经过sigmoid分别得到特征图在高度和宽度上的注意力权重，最后通过矩阵乘法将两张特征图合成一张与输入一样的特征图，这样就获取到了图像高度和宽度的注意力并对精确位置信息进行编码。

结构图如下

![preview](https://pic4.zhimg.com/v2-5e48697e8685334a9cfe9c298b6aa6a7_r.jpg)



附代码

​		注：代码最后没有借助矩阵乘法。

```python

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        a1 = s_h.expand_as(x)
        a2 = s_w.expand_as(x)
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out



# model = CA_Block(512)
# print(model)
# map = torch.randn(2, 512, 26, 26)
# if __name__ == "__main__":
#     outputs = model(map)
#     print(outputs.shape)

```

