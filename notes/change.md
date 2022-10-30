7月22日

# 模型改动

---

### 注意

- **注意参数之间的对应关系，以及特征图的shape匹配问题**

---

### 改动思路：

- 尽可能少的代码，得到高效的结果
- 根据评价指标，看看哪里可以欠缺（补），那个指标影响不大（省），哪里可以更好（改）
- 由任务结果思考评价指标，可能有更合适的指标，是否能用到网络里，是否能根据这个指标是检测效果接近真实结果
- 网络的深度和广度是网络提取信息的重点

### 参考方向

以下是考虑改动的地方

1. 激活函数    关于求导

2. 损失函数    关于更新

3. 学习率lr      关于更新速度

4. 优化器        优化loss lr 

5. 数据增强     关于数据预处理

6. 添加模块     如注意力集中机制

7. 网络修改     注意结构合理融合

8. 新奇可用的小想法

   ......

   有余力加上deep sort

---

### 相关概念

#### 正则化

- Regularization，中文翻译过来可以称为**正则化**，或者是**规范化**。什么是规则？闭卷考试中不能查书，这就是规则，一个**限制**。同理，在这里，规则化就是说给**损失函数**加上一些限制，通过这种规则去规范他们再接下来的循环迭代中，不要自我膨胀。

#### 梯度弥散

- 由于导数的链式法则，连续多层小于1的梯度相乘会使梯度越来越小，最终导致某层梯度为0。

#### 梯度爆炸

- 由于导数的链式法则，连续多层大于1的梯度相乘会使梯度越来越大，最终导致梯度太大的问题

---



## 激活函数

- 注：这里的处理对象 x 是t ensor

#### SiLU

y = x*torch.sigmoid（x）

原主使用的是SiLU函数，也叫做swish函数，其图像如下：

![image-20220723130827426](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220723130827426.png)

SiLU激活函数的优点，无界性方面有利于防止慢速训练期间，梯度接近于0并导致饱和；导数恒大于0；平滑度在优化和泛化中起很大作用。SiLU 的激活大约等于ReLU的激活，SiLU 的一个很强特点是它具有自稳定特性。导数为零的全局最小值在权重上起到“软地板”的作用，作为隐式正则化器，抑制了大数量权重的学习。



#### 以下是常见的几个激活函数



![image-20220723130843685](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220723130843685.png)







#### hardswish_H_SiLU

![image-20220805084055473](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220805084055473.png)



![image-20220725145821689](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220725145821689.png)

因为swish中sigmoid的计算程度太高，对硬件配置要求高，就会在大模型部署上的运算速度，于是就出现了H_SiLU分段函数，在江都计算强调的同时尽可能的等效于swish函数。在相关实验中，发现所有这些功能的硬版本在精度上没有明显的差异，但从部署的角度来看，有多种优势。首先，在几乎所有的软件和硬件框架上都可以使用经过优化的ReLU6实现。其次，在量子化模式下，它消除了由于近似sigmoid的不同实现而造成的潜在数值精度损失，分段函数可以减少内存访问的数量，从而大幅降低延迟成本。



#### sigmoid（）

y = 1/(1 +torch.exp(-x) )

y' = y( 1 - y )

- 好：sigmoid可以对输出归一化，范围0到1；很适合以预测概率为输出的模型，比如二分类；（有界性也是有优势的，因为有界激活函数可以具有很强的正则化，并且较大的负输入问题也能解决）；梯度平滑，避免跳跃出值。
- 不好：函数输出并非以0为中心，权重更新较慢；指数运算，计算机运行较慢；容易梯度弥散，当 y 趋向于 1 和-1 时称为过饱和，处于饱和状态的激活函数意味着当 100 x = 和 1000 x = 时的反映是一样的，这样的特征转换会造成信息给丢失，Sigmoid 函数在 x 取值在-3 到 3 之间应该会有比较好的效果。



#### tanh（）

- tanh函数也是双切函数的一种，他补充了sigmoid函数不适是以0为中心输出的问题，然而，梯度消失的问题和幂运算问题仍旧存在。



#### relu（）

torch

- 好：线性整流函数，正线性单元relu，是网络模型中常用的激活函数；输入为非零时，不存在梯度饱和问题；计算速度快，模型收敛快，模型训练时间较短，且很少出现梯度弥散现象。
- 不好：输出不是以0为中心；在训练过程中，很可能会出现大量的神经元死亡失效，可以通过减小学习率来解决或其变体函数。



#### leaky relu（）

- 为解决dead relu的问题而设计的激活函数，通过一个a值来扩大relu的范围，在yolov3中发挥很好的效果。



#### elu（）

- 好：relu的另一个变体，elu有负值，通过减少偏置偏移的影响，使正常梯度更接近于单位自然梯度，从而使均值向零加速学习；同样没有dead relu问题，输出的平均值接近于零；函数在较小的输入下会饱和至负值，从而减少前向传播的变异和信息；
- 不好：计算强度高，实践中也不总是比relu好，应用不广泛。



#### prelu（）

- 大于0部分为relu，小于等于0部分为leaky relu，即取两者优点。



#### FReLU

-  https://arxiv.org/pdf/2007.11824.pdf       ECCV2020 | FReLU：旷视提出一种新的激活函数，实现像素级空间信息建模

- Funnel激活函数 (FReLU)，通过增加一个空间条件 (见图2)来扩展ReLU/PReLU函数，它是结合卷积实现的带参数的激活函数，增加了一个可以忽略不计的计算开销。 该激活函数的形式是**y=max (x,T (x))，**其中T (x)代表简单高效的空间上下文特征提取器。 由于使用了空间条件，FReLU简单地将ReLU和PReLU扩展为具有像素化建模能力的视觉参数化ReLU。比以前的激活函数更有效,实验环节中，在分类网络中替换了正常的ReLU.
- 空间条件spatial condition以简单的方式实现了像素级建模能力，并通过常规卷积捕获了复杂的视觉layouts。



#### 其他激活函数

- maxout 函数
- softplus 函数
- softsign 函数
- 高斯误差线性单元 GELUs

---



## 注意力集中机制

注意力集中机制已经写到attention_block.md文件里了。学会了4个比较简单的，也看了多头注意力集中机制，还有一些 transformer 的讲解，但因为对transformer理解不深，且transformer参数大，目前把学会的attention添加到网络里进行测试，简单做了个消融实验。

---



## 损失函数

### focus loss

- **改动：**将原来的交叉熵损失函数改为交叉熵损失函数的基础上的focus loss
- **理由：**再网上看到对于focus loss的评论不一，又说他可以较好的区别正负样本，可以长点，但也有的说没什么用。在yolov3的论文里，作者也常讲过，他试过了多次实验，并没有发现focus对yolo有很好的影响。别人说的再多还不如自己实验一遍，不就知道了，于是我就去找了相关视频博客等了解并修改。
- **结果：**跑完训练后，并没有得到好效果，掉点了，确实，在我这个yolo模型中并没有好的影响。

---



## 网络修改

1. 将focus改成两个标准卷积bn激活函数和一个深度可分离卷积构成的组件
2. 把中间及循环部分的卷积改为深度可分离卷积
3. 残差融合部分cat改为add
4. 并且稍稍减少了循环的次数

---



## 数据增强



### baseline已有

- **色域，缩放，水平变换，马赛克，Mixup**



### 可添加

#### 高斯模糊



- 高斯滤波将图像频域处理和时域处理相联系，作为低通滤波器使用，可以将低频能量（如噪声）滤去，起到图像平滑作用。

- 图像的高斯模糊过程就是图像与正态分布做卷积。

  

- 高斯模糊是一种图像模糊[滤波器](https://baike.baidu.com/item/滤波器)，它用[正态分布](https://baike.baidu.com/item/正态分布)计算图像中每个像素的[变换](https://baike.baidu.com/item/变换)。N维空间正态分布方程为

  ![img](https://bkimg.cdn.bcebos.com/formula/4227ca574304a3b92a08fd1bc10c6d10.svg)

  在二维空间定义为

  ![img](https://bkimg.cdn.bcebos.com/formula/af01dd8f4e5f1181a9c8f8570ff0bcf6.svg)

  

- 其中*r*是模糊半径，σ是正态分布的[标准偏差](https://baike.baidu.com/item/标准偏差)。在二维空间中，这个公式生成的曲面的[等高线](https://baike.baidu.com/item/等高线)是从中心开始呈正态分布的[同心圆](https://baike.baidu.com/item/同心圆)。分布不为零的像素组成的[卷积](https://baike.baidu.com/item/卷积)矩阵与原始图像做变换。每个像素的值都是周围相邻像素值的[加权平均](https://baike.baidu.com/item/加权平均)。原始像素的值有最大的高斯分布值，所以有最大的权重，相邻像素随着距离原始像素越来越远，其权重也越来越小。这样进行模糊处理比其它的均衡模糊滤波器更高地保留了边缘效果，如尺度空间实现。

- 模糊半径r越大，范围越大，图片越模糊。



代码实现如下，根目录为yolov5-pytorch-main

```python

file = 'img/1_2.jpg'
save_path = "1_21.jpg"

img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
sigma = 0
kernel_size = (5, 5)
img = cv2.GaussianBlur(img, kernel_size, sigma)
cv2.imwrite(save_path, img)

print("The picture is currently being processed")



```



![image-20220905195141795](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220905195141795.png)

![image-20220905195105962](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220905195105962.png)





#### 添加云雾

给图片添加云雾，类似于模拟现实多雾或阴雨蒙蒙的状态。



```python
import cv2, math
import numpy as np
 
def demo():
    img_path = 'test.png'
 
    img = cv2.imread(img_path)
    img_f = img / 255.0
    (row, col, chs) = img.shape
 
    A = 0.5                               # 亮度
    beta = 0.08                           # 雾的浓度
    size = math.sqrt(max(row, col))      # 雾化尺寸
    center = (row // 2, col // 2)        # 雾化中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j-center[0])**2 + (l-center[1])**2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
 
    cv2.imshow("src", img)
    cv2.imshow("dst", img_f)
    cv2.waitKey()
 
 
if __name__ == '__main__':
    demo()
```





#### **cutout随机擦除**

https://arxiv.org/pdf/1708.04896.pdf

1. 这是一个轻量级的数据增强，不需要任何额外的参数学习或内存消耗。它可以在不改变学习策略的情况下与各种CNN模型相结合。
2. 合理的和其他数据增强一起可以提高模型识别性能。
3. 持续改进最新最先进的深度模型在图像分类、目标检测和人的重新识别方面的性能。
4. 降低了背景对数据的贡献，在CNN的决策上，可以让他不仅仅只是关注目标的整体，多关注目标的局部特征，提高CNNa对部分遮挡样本的鲁棒性，可以降低模型过拟合，使模型具有更好的泛化能力。
5. 随机给图像添加一些小块遮挡，模拟现实遮挡场景，遮挡的形状不影响，

- 

```python
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, fill_value):
        self.n_holes = n_holes
        self.length = length
        self.fill_value= fill_value

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = self.fill_value

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
        
if __name__ == "__main__":
    # # 2、Cutout
    img = cv2.imread("test.jpeg")
    transform = Compose([
        transforms.ToTensor(),
        Cutout(n_holes=30, length=10, fill_value=0.)
    ])
    img2 = transform(img=img)
    img2 = img2.numpy().transpose([1, 2, 0])
    cv2.imshow("test", img2)
    cv2.waitKey(0)

```



paper

![image-20220905195849415](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220905195849415.png)

