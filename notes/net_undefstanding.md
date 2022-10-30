# 主干网络的构建



## 组件

### FReLU

- **torch.max（ x， bn( conv(x) )** ）
  1.  torch.max( 输入x  ,  x经过一个标准2d卷积 + bn)



### Conv

- **1 ：conv + bn + act**
- **2 ：conv + act**
  1. forward ：一个标准2d卷积 + bn + act(FReLU)
  2. fuseforward ： 一个标准2d卷积 + act



### focus

- **Conv（cat（x））**
  1. 将特征图进行隔行隔列提取处四个张量，再cat成一张特征图，此时高宽减半，通道数*4，即channel=12
  2. 将cat得到的特征图进行 一个Conv



### Bottleneck

- **add 为 true ：  输入和两个Conv的输出cat**
- **else  ：  两个Conv输出**
  1. cv1 ：Conv
  2. cv2 ：Conv
  3. add ： 当shortcut=true，输入通道=输出通道



### C3

- **Conv3（cat（m（Conv1（x)），Conv2（x），sdim=1））**
  1. cv1 ： Conv 
  2. cv2 ： Conv
  3. cv3 ： Conv  通道为1 2的2倍
  4. m代表Bottleneck net，base_depth=n为残差块的数量
  5. shortcut=Fals(不用残差块，直接往下卷积) ,



SPP

- **Conv（cat（ 3个m（Conv（x）） ， 1个 Conv（x）））**
  1. cv1 ：Conv
  2. cv2 ： Conv
  3. m ： k=（5， 9， 13）三层最大池化层

---



## 主干

### CSPDarcnet

#### **stem**

- **focus：   cat + Conv**
  1. wh 640 --》320 ， ch  3 --》64



#### **dark2**

- **Conv层降维 + C3层**
  1. Conv 降维，通道数增加   
  2. c3 调用3层 resblock net
  3. wh 320 --》160 ， ch  64 --》128



#### **dark3**

- **Conv层降维 + C3层**
  1. Conv 降维，通道数增加   
  2. c3 调用9层 resblock net
  3. wh 160 --》80 ， ch  128 --》256
- feat1



#### **dark4**

- **Conv层降维 + C3层**
  1. Conv 降维，通道数增加   
  2. c3 调用9层 resblock net
  3. wh 80 --》40 ， ch  256 --》512
- **feat2**





#### **dark5**

- **Conv层降维 + SPP + C3层（shortcut=False）**
  1. Conv 降维，通道数增加   
  2. k=5， 9， 13的最大池化层
  3. c3 调用3层 resblock net， bottleneck输出两个Conv，没有和x cat
  4. wh 40 --》20 ， ch  512 --》1024
- **feat3**

---



## YoloBody

- 注：这里是将batch_size看作1

### conv_for_feat3

- **Conv**
  1. 通道减半
  2. 1024，20，20 --》 512，20，20  =p5_1



### upsample

- **nn.Upsample(scale_factor=2, mode="nearest")**
  1. scale_factor=2 输出为输入的2倍
  2. 512，20，20 --》 512，40，40 = p5_upsample



### cat

- **cat（ [ p5_upsample, feat2 ] , dim=1 ）**
  1. p5上采样与p2 cat
  2.  两 个 512，40，40 --》 1024，40，40 = p4_1



### conv3_for_upsample1

- **C3（base_depth, shortcut=False）**
  1. base_depth, shortcut=False不用残差块，直接往下卷积，m为3层bottleneck 
  2. 1024，40，40 --》512，40，40 = p4_2



### conv_for_feat2

- **Conv**
  1. 通道减半
  2. 512，40，40 --》256，40，40 = p4_3



### upsample

- **nn.Upsample(scale_factor=2, mode="nearest")**
  1. scale_factor=2 输出为输入的2倍
  2. 256， 40， 40 --》256，80， 80  = p4_upsample



### cat

- **cat( [ P4_upsample, feat1 ] ，dim=1 )** 
  1. p4_3上采样与feat1 cat
  2.  两个 256，80，80 --》512，80，80 = p3_1



### conv3_for_upsample2

- **C3（base_depth, shortcut=False）**
  1. base_depth, shortcut=False不用残差块，直接往下卷积，m为3层bottleneck 
  2. 512，80，80 --》256，80，80  = p3_2



### down_sample1

- **Conv**
  1. k=3，s=2  p3_2下采样
  2. 256，80，80 --》256，40，40 = p3_downsample



### cat

- **cat( [ P3_downsample, P4_3 ] , 1 )**
  1. p3_2 的下采样和p4_3  cat
  2. 两个256，40，40 --》 512，40，40 = p4_4



### conv3_for_downsample1

- **C3（base_depth, shortcut=False）**
  1. base_depth, shortcut=False不用残差块，直接往下卷积，m为3层bottleneck 
  2. 512，40，40 --》512，40，40 = p4_5



### down_sample2

- **Conv**
  1. k=3，s=2 p4_5下采样
  2. 512，40，40 --》512，20，20



### cat

- **cat ( [ P4_downsample, P5_1 ] ，1 )**
- 两个512，20，20 --》1024，20，20 = p5_2



### conv3_for_downsample2

- **C3（base_depth, shortcut=False）**
  1. base_depth, shortcut=False不用残差块，直接往下卷积，m为3层bottleneck 
  2. 1024，20，20 --》1024，20，20 = p5_3



### yolo_head_P3(P3)

- **conv2d**
  1. 改变输出通道数，将张量转化为【坐标 宽高 置信度 类别】
  2. 256 ，80，80 --》25，80，80 = out2



### yolo_head_P4(P4)

- **conv2d**
  1. 改变输出通道数，将张量转化为【坐标 宽高 置信度 类别】
  2. 512 ，40，40 --》25，40，40 = out1



### yolo_head_P5(P5)

- **conv2d**
  1. 改变输出通道数，将张量转化为【坐标 宽高 置信度 类别】
  2. 1024 ，20，20 --》25，20，20 = out0





```python


```

---



8月18日

# 网络的构建



## 网络架构内部更改



### 改动一、

- **改动：**将之前的focusnet换成两个标准卷积和一个深度可分离卷积合成的组件。
- **理由：**因为focusnet的特征提取方式各一个像素点取一个然后再堆叠，让我觉的他在一定程度使特征图丢失了空间和通道上的信息，特别是对感受野的形成有很大的影响，当然这只是我个人的认为。
- **结论：**我用两个标准卷积和一个深度可分离卷积（均有bn和act层）进行替换，因为标准卷积可以更全面的提取特征，对感受野的形成友好。



### 改动二、

- **改动：**我把中间和循环中的标准卷积换成了深度可分离卷积
- **理由：**本来是想将他换成分组卷积的，后面发现，深度可分离卷积，可以以对精度有很小的影响很大的减少参数量，因为我看了mobilenet的论文，Mobilenet模型与其他相比的模型，mobilenet的mutil_add更低，精度相差不大，所以就进行尝试。
- **结果：**还以为将标准卷积替换成深度可分离卷积，模型的速度会变快，因为他的参数量确实减少了很多，但是并没有，后面查了才知道，也有点对应我的猜测，深度可分离卷积是由分组卷积和1*1的卷积组成的，而通道越深，分组卷积并行度越高，会占据很大的内存，推理也会受到影响。



### 改动三、

- **改动：**残差堆叠出的特征层融合方式由cat改为add，保持标准卷积提取特征。
- **理由：**我最初认为的特征融合就是add，因为循环特征提取的过程中，将每一个通道的特征都提取，通过对应的简单相加，将提取到明显的特征add到输入，这样明显的特征就可以在融合下得到很好的保留。后面查询cat的原理，发现，哦不，博客给了我另外的思考，cat在某些情况下理论上比add好，而add的可以认为是cat的一种特殊情况，也就是cat保持通道不变经过同一个卷积后得到的特征图。但也有的说，add对分类可能会更好，结合我自己的认知，我觉得我应该用add，并且add在计算成都上比cat的低。
- **结论：**add跟多的是在特征图的空间上提供更多的信息，而cat的话（默认其后面会有卷积操作）在通道上融合后再再空间上融合，相对而言，信息融合的更好，但所得的信息也没有这么明显了。



### 改动四、

- **改动：**减少resblock的数量

- **理由：**因为我在前面把focusnet换成了卷积组件，相对于原来加深了网络的深度，在一定程度上提高了感受野，所以我认为减少一些block可能会更好一些（也有认为，可能过多的循环有些冗余，没有很大的必要）。



