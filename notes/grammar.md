### %matplotlib inline

**%matplotlib inline** 用在Jupyter notebook中（代替plt.show()）； PyCharm中不支持，在PyCharm中去掉这个即可（用plt.show()代替图像的显示）。

- 注：%matplotlib inline 可以在Ipython编译器里直接使用，功能是可以内嵌绘图，并且可以省略掉plt.show()这一步。
- 注：而%matplotlib具体作用是当你调用matplotlib.pyplot的绘图函数plot()进行绘图的时候，或者生成一个figure画布的时候，可以直接在你的python console里面生成图像。







if classname.find('Conv') != -1:  **# find ()函数，实现查找classname中是否含有conv字符，没有返回-1；有返回0**.

 nn.init.normal_(m.weight.data, 0.0, 0.02)   #  m.weight.data 表示需要初始化的权重。



### tensor.detach()

返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。

即使之后重新将它的requires_grad置为true,它也不会具有梯度grad

这样我们就会继续使用这个新的tensor进行计算，后面当我们进行反向传播时，到该调用detach()的tensor就会停止，不能再继续向前进行传播

注意：

使用detach返回的tensor和原始的tensor共同一个内存，即一个修改另一个也会跟着改变。


```python
import torch
 
a = torch.tensor([1, 2, 3.], requires_grad=True)
print(a.grad)
out = a.sigmoid()
 
out.sum().backward()
print(a.grad)
'''返回：
None
tensor([0.1966, 0.1050, 0.0452])
'''
```



**当使用detach()分离tensor但是没有更改这个tensor时，并不会影响backward():**

```python
import torch
 
a = torch.tensor([1, 2, 3.], requires_grad=True)
print(a.grad)
out = a.sigmoid()
print(out)
 
#添加detach(),c的requires_grad为False
c = out.detach()
print(c)
 
#这时候没有对c进行更改，所以并不会影响backward()
out.sum().backward()
print(a.grad)
 
'''返回：
None
tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)
tensor([0.7311, 0.8808, 0.9526])
tensor([0.1966, 0.1050, 0.0452])
'''
```

tensor c是由out分离得到的，但是我也没有去改变这个c，这个时候依然对原来的out求导是不会有错误的，即

c,out之间的区别是c是没有梯度的，out是有梯度的,但是需要注意的是下面两种情况是汇报错的

- **当使用detach()分离tensor，然后用这个分离出来的tensor去求导数，会影响backward()，会出现错误**

- **当使用detach()分离tensor并且更改这个tensor时，即使再对原来的out求导数，会影响backward()，会出现错误如果此时对c进行了更改，这个更改会被autograd追踪，在对out.sum()进行backward()时也会报错，因为此时的值进行backward()得到的梯度是错误的**



### pytorch.clamp()

将小于0的元素修改为0，规范元素的取值

---



## 卷积

### **权重 **

**即filter的所有元素的总量** 

###  **偏置**

**每个filter 都有一个bias，它是一个实数，都等于卷积核的个数**

例如，input特征图的channel=3， input特征图 + 5个3*3的filter + output特征图 ，5即为output特征图的channel

其中  weight = Fh * Fw * Fc *Fn = 3 * 3 * 3 * 5 = 135 个参数  （姑且把filter的elements成为像素点）

其中    bias  = 1 * 1 * 1 * channel_output 的一个张量（4大维）

注：大维即张量有多少个大括号，小维即张量内部有多少个元素



### groups

[笔记出处](https://www.cnblogs.com/wanghui-garcia/p/10775851.html)   

- **groups** ( [*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional* ) -- 从输入通道到输出通道的阻塞连接数。默认值：1  
- groups的值必须能整除in_channels



#### 例子

- 假设这里没有降维的操作

#### 1.当设置group=1时：

```
conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, groups=1)
conv.weight.data.size()
```

返回： 

```
torch.Size([6, 6, 1, 1])
```

此时卷积层需要 6 * 6 * 1 * 1 个参数，即需要6个的卷积核 ， 权值所要参数为 6 * 6 * 1 * 1 = 36 个

计算 ： h * w * 6 的input乘以一个 1 * 1 *6卷积核，得到一个 h * w * 1 的output，需要6个卷积核才能得到6个 h * w * 1，最后将这6个一维特征图cat起来即为最后的输出。



#### 2.当设置group=3时

```
conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, groups=3)
conv.weight.data.size()
```

返回：

```
torch.Size([6, 2, 1, 1])
```

当groups = 3时， 权重参数shape[6， 2， 1， 1] 即6个1 * 1 * 2的卷积核，权值所要参数为 6 * 2 * 1 * 1 = 12

计算 ： 此时将输入iutput按channels分成3分，每一份为 h *  w * 2 的特征图，6个卷积核也分为3分，每一份为2个1 * 1 * 2的卷积核，分别处理input的3份特征图，可以看作将一张特征图按channels分成groups份输入处理

一份h *  w * 2 的特征图，一份为2个1 * 1 * 2的卷积核，得到输出的 一份 h * w * 2 的output的特征图

最后将3份output特征图cat，即为最终的输出out。

**借鉴处的** ： **up的实际实验中，同样的网络结构，分组卷积的效果好于未分组的卷积的效果**



#### **总结**

可以看出 groups 控制的输入的特征图将采取什么方式进行卷积操作，以及权重参数的多少！

---





## upsample

- **最邻近插值法**

举例：

```
import torch
from torch import nn
input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
input
```

返回：

```
tensor([[[[1., 2.],
          [3., 4.]]]])
```

 

```
m = nn.Upsample(scale_factor=2, mode='nearest')
m(input)
```

返回：

```
tensor([[[[1., 1., 2., 2.],
          [1., 1., 2., 2.],
          [3., 3., 4., 4.],
          [3., 3., 4., 4.]]]])
```
