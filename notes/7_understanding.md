# 网络中一些相关概念理解

1. [原理实现过程](##原理实现过程)
2. [图像的上采样与下采样](###图像的上采样余下采样)
3. [插值法](###插值法)
4. [letterbox_image_mrhzbsjs](###letterbox_image_mrhzbsjs)
5. [非极大值抑制](###非极大值抑制)
6. [预训练模型](###预训练模型)
7. [网络预处理后处理](##网络预处理后处理)
8. [预处理](###预处理)
9. [前处理](###前处理)
10. [Adam优化器](#Adam优化器)

---



## 原理实现过程

- 很重要！！！

https://v.youku.com/v_show/id_XNTEyNjU4MTcyNA==.html?spm=a1z3jc.11711052.0.0&isextonly=1

https://v.youku.com/v_show/id_XNTEyNjU4Mjc2NA==.html?spm=a2hbt.13141534.1_2.d_3&scm=20140719.manual.114461.video_XNTEyNjU4Mjc2NA==

-------



（7月3日）

### 图像的上采样与下采样

https://blog.csdn.net/stf1065716904/article/details/78450997



### 插值法

[插值法](https://so.csdn.net/so/search?q=插值法&spm=1001.2101.3001.7020)(最邻近，双线性，双三次）的原理及实现

https://blog.csdn.net/weixin_40163266/article/details/113776345



### letterbox_image_mrhzbsjs

**https://blog.csdn.net/mrhzbsjs/article/details/88828307**

一、预测过程 letterbox _ image 为了防止失帧，不进行简单的resize，先放大图片，进行三次样条插值，创建一个300*300的灰色图片，把放大后的图片粘贴到灰色图片上，相当于在边缘加上灰条。

---



（7月6日）

### IOU  和 G_IOU

#### 第一步，计算重合面积

1. 先取锚框左上角最大的坐标，再取锚框右下角最小的坐标
2. 计算w  将取得的坐标  右下角的横坐标 - 左上角的横坐标 = w
3. 计算h  将取得的坐标  右下角的纵坐标 - 左上角的纵坐标 = h
4. S1 = w * h

#### 第二步，计算合并总面积

1. 先取锚框左上角最小的坐标，再取锚框右下角最大的坐标
2. 计算w  将取得的坐标  右下角的横坐标 - 左上角的横坐标 = w
3. 计算h  将取得的坐标  右下角的纵坐标 - 左上角的纵坐标 = h
4. S2 = w * h
5. S3 =  S2 - S1
6. IOU = S1 / S3  越大越好
7. G_IOU = (S2 - S1) / S3  越小越好



### 非极大值抑制

1. 选取大于一定阈值的锚框，将所有锚框按得分值由高到低排序

2. 按照索引将得分符合要求的锚框取出来

3. 取出得分最高的锚框

4. 判断如果只有一个锚框，跳出循环，说明只有一个预测框

5. 如果还有其他框，则进行IOU比较，相近的则去掉（很可能这个锚框与得分最高的锚框预测的是同一个物体，所以去掉重复的）

   小于比较阈值的话则保留

6. 重复，再去剩下得分最高的锚框，重复上述操作，直到所有锚框都挑选出来

7. 将所得锚框进行cat

8. 返回预测框

---



### 预训练模型

预训练模型是深度学习的架构，已经过训练已执行大量数据上的特定任务，通常指的是在[Imagenet上](https://links.jianshu.com/go?to=http%3A%2F%2Fimage-net.org%2F)训练的CNN（用于视觉相关任务的架构）。ImageNet数据集包含超过1400万个图像，其中120万个图像分为1000个类别（大约100万个图像含边界框和注释）。

那么什么是预训练模型？这是在训练结束时结果比较好的一组权重值，研究人员分享出来供其他人使用。我们可以在github上找到许多具有权重的库，但是获取预训练模型的最简单方法可能是直接来自您选择的深度学习库。



---



# 网络预处理后处理

### 预处理

- 数据集的导入、划分、打包

### 前处理

- 从划分好的数据集里，导入图片，转换图片数据类型
- 对图片进行resize
- 对图片进行数据增强
- 将处理好的图片送进网络

### 后处理

- 在模型完成预测之后的处理
- 如yolov5，他根据与之检测出的框可能会有重叠的，进行非极大抑制等



## Adam优化器

- 下方就是此模型的优化器内部参数等

Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.937, 0.999)
    eps: 1e-08
    lr: 2.9999999999999997e-05
    maximize: False
    weight_decay: 0

Parameter Group 1
    amsgrad: False
    betas: (0.937, 0.999)
    eps: 1e-08
    lr: 2.9999999999999997e-05
    maximize: False
    weight_decay: 0

Parameter Group 2
    amsgrad: False
    betas: (0.937, 0.999)
    eps: 1e-08
    lr: 2.9999999999999997e-05
    maximize: False
    weight_decay: 0
)
