# 改动对比



## 改进一、激活函数



### 说明

此次改动训练了两次。

- 主要无关变量 batch_size、精度、num_works等

- 自变量  激活函数

- 指标为所有类的平均指标，指标是由测试脚本检测而来的。



### baseline_SiLU

baseline使用的是SiLU函数，即swish函数，除了计算量比较高以外，她所获得的激活效果是非常好的，以至于v6，v7的激活函数依旧是他，此次是从零开始训练的baseline模型，其中train的loss由2.7左右下降到0.057左右，val的loss由1.7几下降到0.056左右，召回率和精确率已经平均调和F1都比较高。总的来说，baseline的检测效果是比较好的，我跑了两次，本地跑一次，修改参数后在服务器由跑了一次。

最好的模型的map达到83.10%,以下是一些检测指标的展示，这个是在服务器上面跑的baseline。

| 指标     | map    | f1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 83.10% | 79.14% | 74.47% | 84.88%    |

下面是每个类的map条形图展示。

<img src="D:\shuqikaohe\yolov5-pytorch-main\saved_model\map_out_baseline_silu\results\mAP.png" style="zoom:80%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220729114610781.png" alt="image-20220729114610781" style="zoom: 67%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220729114948401.png" alt="image-20220729114948401" style="zoom: 50%;" />







### change_hardswish

- - 同一个函数两种不同的代码呈现形式，所得结果相差不大，他们只是在部署方面有区别有区别，理应来说，所得结果应该一样，不懂其内在原理，但实验结果差别甚微。

- hardswish激活函数是一个分段函数，在尽可能的模拟silu的效果的同时降低计算程度，

- hardswish激活函数是mobilenet V3在swish函数基础上的改进，主要是用于替换ReLU6的，作者认为swish函数平滑非单调，在深层模型学习优于ReLU函数，但是sigmoid指数计算成本高，而作者看中了大多数的软硬件框架上都可以使用ReLU6的优化实现，ReLU6还可以在特定的模型是消除由于近似sigmoid的不同实现而带来的潜在的数值精度损失。

#### x * F.hardsigmoid

这里我把silu改成x * F.hardsigmoid，理论上所得出的效果应该差不多，我还导入之前训练的30个epoch来训练的，训练时间确实短了不少，而且map也相对提升了一点点，精确率很高，但是其他召回率和平均调和F1相对就很低。检测的效果也没有baseline的效果好，我觉得是他的分段线性不能够像silu给网络带去一些非线性影响因素，网络相比于silu训出来的的简单。

此次训练得到的最好的模型map83.80%，指标展示如下。

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 83.80% | 53.82% | 39.16% | 97.42%    |

下面是每个类的map的条形图展示

<img src="D:\shuqikaohe\yolov5-pytorch-main\saved_model\map_out_server_Hsilu\results\mAP.png" style="zoom:80%;" />



<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220729115438983.png" alt="image-20220729115438983" style="zoom:67%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220729115810590.png" alt="image-20220729115810590" style="zoom:50%;" />







#### x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0

确实和baseline的map相差不大，但是召回率就低了很多，召回率低了精确率自然就上升，但是由于召回率过低，F1自然不高。

此次模型指标如下：

score = 0.5

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 83.40% | 53.49% | 38.72% | 97.97%    |

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804110134955.png" alt="image-20220804110134955" style="zoom:80%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804111225342.png" alt="image-20220804111225342" style="zoom:67%;" />



<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804111716766.png" alt="image-20220804111716766" style="zoom:50%;" />



### change_frelu

因为激活函数的非线性能够提升网络的复杂性，从而使网络有更好地拟合检测效果，看到了一篇paper https://arxiv.org/pdf/2007.11824.pdf  介绍frelu能使网络性能和鲁棒性有很大的提升，还给出了在coco和imagenet上frelu检测效果相对于swish prelu relu 这三个函数好很多，它扩展了ReLU和PReLU到一个2D视觉激活，只增加了可忽略的计算开销，简单但有效，正好符合我的需求，于是我从官网中把这个自定义的frelu换到我的网络中进行训练，由于之前改的那个激活函数的检测效果不是很好，以为改了之后应该和silu效果差不多，没想到实际检测出来召回率和平均调和F1比silu的低很多，这是让我没有想到的，果然，这个非线性的卷积激活函数消除了激活函数中空间的不敏感，真的取得了很好的效果，结合卷积的激活函数，他的参数是可以学习并保留下来的。美中稍稍差以，他的召回率没有baseline的高，有些极少图像baseline检测的出来，他检测不出来，也算是差强人意。（补充：召回率只是设置的门限问题，这个模型的检测效果比baseline的好）

此次模型指标如下：

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测效果 | 88.06% | 81.13% | 71.93% | 94.11%    |

以下是他的每个类的map展示

<img src="D:\shuqikaohe\yolov5-pytorch-main\saved_model\map_out_server2frelu\results\mAP.png" style="zoom:80%;" />



检测效果图

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220726133509813.png" alt="image-20220726133509813" style="zoom: 67%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220729113959756.png" alt="image-20220729113959756" style="zoom: 50%;" />



### 总结

今天激活函数告一段落，基于我自己的理解以及我的实践，可能很多人认为改个激活函数很容易，但是想改一个正面的激活函数还是很难的，激活函数看似简单但是他对网络的影响是非常显著的，激活函数的求导、线性非线性、含不含参都是需要思考能不能有助于网络性能的提升。激活函数，我花了很长时间去调整，光训练我就跑了四次，每一个得到的结果不经相同，根据消融实验不同指标，让我对不同激活函数同一网络有了进一步的理解。





**注：由于之前的召回率很低，我就在想如何提升召回率，后面查询发现，分数门限对召回率的影响非常大，适当降低分数，就可以显著提升召回率，虽然检测物体的分数很低，但不代表检测到他的框不准，只是因为分数低，就把他过滤掉而已，故在一下改经我会把分数门限改为0.4。**





## 改进二、添加注意力集中机制

- **注：检测展示图是score = 0.4，注意力模块是backbone的输出，即neck的输入添加的**



### HSiLU6_SE

此次实验是在HSiLU6（即上方HSiLU6）下添加SEnet，在调试时，但匹配shape时改动了一下计算方式，我觉得矩阵对应相称让特征值变小了，有些可能之前比较突出的特征就没有这么明显了，相对于没加注意力集中机制前，所得测试指标还行，测试效果也比之前的好。

此次模型指标如下：

score = 0.5

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 85.61% | 75.11% | 63.71% | 93.53%    |



score = 0.4

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 85.61% | 79.79% | 72.13% | 90.63%    |

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804150031094.png" alt="image-20220804150031094" style="zoom:67%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804150125112.png" alt="image-20220804150125112" style="zoom:50%;" />



### FReLU_SE

通道注意力集中机制，特征提取，重点关注通道特征，经过全连接，得到通道系数权重，再乘以原图，在一定程度上使网络关注图像得通道特征。把他加入frelu得网络上，得到得指标有微小得提升。

检测指标如下：

score = 0.5

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 88.57% | 82.08% | 73.73% | 93.65%    |



score = 0.4

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 88.41% | 84.41% | 79.54% | 90.30%    |

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220805162242859.png" alt="image-20220805162242859" style="zoom:80%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220805162105374.png" alt="image-20220805162105374" style="zoom:67%;" />



<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220805162149025.png" alt="image-20220805162149025" style="zoom:50%;" />





### FReLU_CBAM

CBAMnet是将通道注意力集中机制和空间注意力集中机制的结合，池化层特征提取，消参，在经过全连接整和，再经过维度不变，取最大和取平均后用1乘1的卷积进行特征层提取。在意料之中，在改动激活函数并加上注意力集中机制后，map会提升，这里提升了1个百分点，F1也提升了1个百分点，召回率提升了3个百分点，精确率降了0.6个百分点，检测效果还是不错的。调整门限分数以后，recall提升了5个百分点，当然精确率也相对下降一些。

此次模型指标如下：

score = 0.5

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 89.10% | 82.62% | 74.49% | 93.50%    |



score = 0.4

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 89.10% | 84.45% | 79.49% | 90.46%    |

![image-20220804100437529](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804100437529.png)

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804100236324.png" alt="image-20220804100236324" style="zoom: 67%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804100313653.png" alt="image-20220804100313653" style="zoom: 50%;" />





### FReLU_ECA

ECAnet可以认为是SEnet的改进版，作者表明SEnet中的降维给通道注意力集中机制带来副作用，并且捕获所有通道之间的依存关系低效且没有必要。而ECA直接在全局平均池化后1*1卷积提取信息，高效参数少，跨通道交互，利用函数控制卷积核大小通道自适应。

该公式为   K = | log2（C）/r + b/r | 其中 r = 2， b = 1； 得到通道权重再乘以原图。不过这个模块添加进去以后，得到的结果是负增长，基本上各项指标都负增长了1个百分点。检测效果好吧，并不说所有的trick添加进去就可以提升网络效果。

此次模型指标如下：

score = 0.5

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 87.64% | 80.43% | 71.28% | 93.49%    |



score = 0.4

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 87.64% | 83.19% | 76.98% | 90.98%    |

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804204211076.png" alt="image-20220804204211076" style="zoom: 67%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804210330909.png" alt="image-20220804210330909" style="zoom:67%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804210405345.png" alt="image-20220804210405345" style="zoom:50%;" />





### FReLU_CA

一般通道注意力对模型性能具有显著的效果，但他们通常会忽略位置信息，而位置信息对于生成空间选择性attention maps是非常重要的，所以作者通过将位置信息嵌入到通道注意力中提出了新的注意力机制，也就是CA，与通过2位全局池化将特征张量转换为单个特征向量的通道注意力不同，CA将通道注意力分解为两个1维特征编码的过程，分别可以沿2个空间方向聚合特征，一个沿空间方向捕获远程依赖，一个沿空间方向保留位置信息，特征互补，最后将两张特征图融合乘以原图。将之添加到网络后各项指标均有小幅度上，检测效果较没加之前好，但不明显。



此次模型指标如下：

score = 0.5

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 88.29% | 81.36% | 72.09% | 94.58%    |



score = 0.4

| 指标     | map    | F1     | recall | precision |
| -------- | ------ | ------ | ------ | --------- |
| 检测结果 | 88.29% | 84.08% | 78.40% | 91.24%    |

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220805093229774.png" alt="image-20220805093229774" style="zoom: 80%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804174321547.png" alt="image-20220804174321547" style="zoom:67%;" />



<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220804164436759.png" alt="image-20220804164436759" style="zoom:50%;" />





### 总结

注意力集中机制是网络常用的一种特征提取的trick，它可以有效地提升模型的检测效果，当然，网络添加注意力集中机制的地方有很多，我放在backbone的后面，主要是考虑到我想要加载预训练模型，所以就没有放到backbone里面，之所以没有放到neck后面，是觉得那里的网络比较深了，许多特征已经不是很明显了，在这里加的话，可能没有什么效果。此次消融实验跑了四次训练，训练完以后根据训练效果对比，发现，相对于我这个网络，CBAM模块和CA模块的效果相差不大，且网络性能相对较好，SE模块一般，ECA则是负增长，用了这些模块，我个人觉得，最后和原特征图融合的时候，shape一致的用加法会更好一些，因为我们的像素值是比较小的（经过归一化处理），用乘法的话会变得更小，会使得某些特征变得不那么明显，从而比较难以提取（当然你要是用softmax是使它特征更明显也行，不过我没有试过），加法的话，相对较好一些，让特征明显一些。



## 改进三、损失函数

### focal loss

这里是在FPeLU_CBAM的基础上修改的损失函数，focal loss这个损失函数看到过很多次，它是基于交叉熵损失函数上修改的一个新的损失函数，网上对他的评论褒贬不一，于是就去实验一下，看看这个focus loss的鲁棒性，跑完训练后，并没有明显的提升，反而精度还掉了一点点。说实话，理论上，这个loss给正负样本不同的加权，应该会有些许提升，不是很理解，但事实确实没有涨点。

此次模型指标如下

| 模型 \ 指标 | map    | F1     | recall | precision |
| ----------- | ------ | ------ | ------ | --------- |
| BCEloss     | 89.10% | 82.62% | 74.49% | 93.50%    |
| focus loss  | 88.46% | 73.69% | 61.18% | 95.48%    |



<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220809171521824.png" alt="image-20220809171521824" style="zoom: 80%;" />



<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220821211017341.png" alt="image-20220821211017341" style="zoom:67%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220821211204230.png" alt="image-20220821211204230" style="zoom:80%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220821210940923.png" alt="image-20220821210940923" style="zoom:67%;" />

### 总结

也许是我对focal loss的理解不够好，没有合理的运用它，只是简单理解原理，用在网络里。



### 



## 改进四、网络结构

因为小伙伴们太卷了，于是我也开始更改主干网络，我把focusnet换成了两个标准卷积+bn+act和一个深度可分离组成的小组件，把中间除了特征融合出的卷积，其他都换成深度可分离卷积，因为前面用了小组件，稍稍减少了resblock，并把resblock的特征融合改成add。跑了300个epoch，测试loss在100epoch左右开始平稳，而train loss缓慢下降，模型开始收敛，最好的模型是在140个epoch左右。模型的速度比baseline慢了10个fps，但是模型的参数量几乎减少一半，到比baseline好很多的检测效果。



| 模型 \ 指标 | map    | F1     | recall | precision |
| ----------- | ------ | ------ | ------ | --------- |
| baseline    | 83.10% | 79.14% | 74.47% | 84.88%    |
| change_net  | 90.51% | 85.13% | 78.33% | 93.96%    |

![image-20220820113123397](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220820113123397.png)



<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220821215531935.png" alt="image-20220821215531935" style="zoom:80%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220821215636305.png" alt="image-20220821215636305" style="zoom: 50%;" />



### 中间调参实验

- 这个实验可算是检验参数调整吧，我把马赛克数据增强的比例调整到0.5，并且不加注意力集中机制，并引入之前训练了60-70代的模型，在训练200次，得到的训练结果比调整降了2点几，这说明，0.7比例的马赛克数据增强在加上注意力模块对模型的检测更加有效，不过值得注意的是测试的精度效果明显好于没有调整之前的，所以说，若是有时间，调整好参数，可能会得到更好的效果。

模型检测如下：

| 模型 \ 指标 | map    | F1     | recall | precision |
| ----------- | ------ | ------ | ------ | --------- |
| baseline    | 83.10% | 79.14% | 74.47% | 84.88%    |
| 调整        | 87.89% | 81.43% | 72.82% | 93.35%    |



<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220829135931406.png" alt="image-20220829135931406" style="zoom:67%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220829140018377.png" alt="image-20220829140018377" style="zoom:50%;" />



## 改进点五、随机擦除

- 这是一个轻量级的数据增强，也就是不需要参数，他降低了背景对数据的贡献，可以让CNN不仅仅只是关注目标的整体，多关注目标的局部特征，提高CNNa对部分遮挡样本的鲁棒性，可以降低模型过拟合，使模型具有更好的泛化能力，可以更好模拟现实场景。我想着，提高模型识别遮挡物后的目标，这样不就更好的提高模型的召回率吗，还有就是训练时模拟遮挡，但是在测试时是没有遮挡的，这样的话，测试时有更多的数据信息形成感受野，得到的精度也会更高。于是我就把它用在网络里。
- 得到的结果有些意外，因为他并没有提升模型的性能，各项指标都降低了，我觉得他不work的原因是，我的网络本来就是轻量级网络，训练也只是收敛，并没有出现过拟合现象，而且在我的改动之后，模型的参数量几乎减半，此时再加入随机擦除就相当于引入很大的噪声，更加不利于网络的性能提升。

模型检测如下：

| 模型 \ 指标 | map    | F1     | recall | precision |
| ----------- | ------ | ------ | ------ | --------- |
| baseline    | 83.10% | 79.14% | 74.47% | 84.88%    |
| change_net  | 90.51% | 85.13% | 78.33% | 93.96%    |
| 调整        | 87.89% | 81.43% | 72.82% | 93.35%    |
| cutout      | 86.58% | 79.15% | 69.60% | 93.17%    |





<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220901112921543.png" alt="image-20220901112921543" style="zoom:67%;" />



<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220901113007865.png" alt="image-20220901113007865" style="zoom:50%;" />



### 总结

- 随机擦除更适用于网络结构复杂的模型，或者参数量较大的模型，对于相对简单的模型，引入可能会导致巨大的噪声影响，不能达到是网络性能提升的效果。





## 改进点六、通道注意力

- 因为之前加的注意力都是在yolo_neck后面添加的，所以这一次我在backbone里面添加，我的目的很明确，我就是想通过SE获得特征层的通道信息，让卷积提取信息是更加注重特征比较明显的区域，提取更有用的特征，更有利于感受野的形成，增强模型的泛化能力。

模型指标如下：

| 模型 \ 指标 | map    | F1     | recall | precision |
| ----------- | ------ | ------ | ------ | --------- |
| baseline    | 83.10% | 79.14% | 74.47% | 84.88%    |
| change_net  | 90.51% | 85.13% | 78.33% | 93.96%    |
| 调整        | 87.89% | 81.43% | 72.82% | 93.35%    |
| cutout      | 86.58% | 79.15% | 69.60% | 93.17%    |
| backbone    | 91.04% | 85.61% | 79.26% | 93.59%    |



<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220901145010185.png" alt="image-20220901145010185" style="zoom:80%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220904134808158.png" alt="image-20220904134808158" style="zoom:67%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220904134854143.png" alt="image-20220904134854143" style="zoom:50%;" />

### 总结

- 合理的利用注意力模块，可以更好的提升模型的性能，添加时，注意全面考虑，知道添加的模块主要影响的区域。
