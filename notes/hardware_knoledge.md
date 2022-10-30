# 关于电脑硬件相关概念

-------------

[TOC]

SDK

全名： software development kit 软件开发工具包

tensorTR

TensorRT 专注于在 GPU 上快速高效地运行已经训练好的网络，以生成结果；也称为推理。

## **什么是TensorRT**

TensorRT是可以在**NVIDIA**各种**GPU硬件平台**下运行的一个**C++推理框架**。我们利用Pytorch、TF或者其他框架训练好的模型，可以转化为TensorRT的格式，然后利用TensorRT推理引擎去运行我们这个模型，从而提升这个模型在英伟达GPU上运行的速度。速度提升的比例是**比较可观**的。



##  **什么是ONNX？**

简单描述一下官方介绍，开放神经网络交换（Open Neural Network Exchange）简称ONNX是微软和Facebook提出用来表示深度学习模型的**开放**格式。所谓开放就是ONNX定义了一组和环境，平台均无关的标准格式，来增强各种AI模型的可交互性。

换句话说，无论你使用何种训练框架训练模型（比如TensorFlow/Pytorch/OneFlow/Paddle），在训练完毕后你都可以将这些框架的模型统一转换为ONNX这种统一的格式进行存储。注意ONNX文件不仅仅存储了神经网络模型的权重，同时也存储了模型的结构信息以及网络中每一层的输入输出和一些其它的辅助信息。

----------------



7月19日

**借鉴出处**：https://blog.csdn.net/u012605037/article/details/115294898

## 进程

python函数os.getpid可以获取当前进程的PID，
python函数os.getppid 可以获取当前进程的主进程的PPID

### 分布式

分布式指多个系统协同合作完成一个特定任务的系统，它由不同的系统部署在不同的服务器上，服务器之间相互调用，类似于任务减轻化，大问题分成多个小问题解决。

### node

物理节点，一台机器也可以是一个容器，节点内部可以有多GPU，一个pc可以有多个显卡

### rank

表示进程的编号序号，每一个进程对应一个rank的进程，整个分布式由许多rank完成。在一些结构图中rank指的是软件点，可看作一个计算单位，

### local_rank

指的是在某node一台机器上的进程序号，（个人理解，进程对应GPU），如机器一有4个进程，GPU0到GPU4，local_rank也是0-4。即，每个node的local_rank是独立的，而rank是node之间有关联的，rank是整个分布式进程的序号，local_rank是node上进程相对的序号。

nnodes：                  物理节点数量

node_rank：             物理节点序号

nproc_per_node：    每个物理节点上面进程的数量

groud：进程组。       默认只有一个组

world size：			   全局的并行数，一个分布式任务相当于一个全局，全局并行数就是rank的数量

----------



### 一个运算题：

每个node包含16个GPU，且nproc_per_node=8，nnodes=3，机器的node_rank=5，请问word_size是多少？ 
答案：word_size = 3*8 = 24 

### 下图是一个例子

比如分布式有三台机器，每台机器4个进程，每个进程占用1个GPU（进程客栈用多个GPU）

![img](https://img-blog.csdnimg.cn/20210329102344470.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI2MDUwMzc=,size_16,color_FFFFFF,t_70#pic_center)

图中：12 个rank，nproc_per_node=4, nnodes=3,每一个节点都对应一个node_rankGroup：进程组，一个分布式任务对应了一个进程组。只有用户需要创立多个进程组时才会用到group来管理，默认情况下只有一个group。
注意：

1、rank与GPU之间没有必然的对应关系，一个rank可以包含多个GPU；一个GPU也可以为多个rank服务（多进程共享GPU）。

这一点在理解分布式通信原理的时候比较重要。因为很多资料里面对RingAllReduce、PS-WorK 等模式解释时，习惯默认一个rank对应着一个GPU，导致了很多人认为rank就是对GPU的编号。

2、“为什么程序里面的进程用rank表示而不用proc表示?”

这是因为pytorch是在不断迭代中开发出来的，有些名词或者概念并不是一开始就设计好的。所以，会发现node_rank 跟软节点的rank没有直接关系。
