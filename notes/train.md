### 训练过程

以下就是baseline训练过程口述。



参数初始化（—人话就是给参数赋值）

判断是否适用多卡训练

根据路径导入类和类的数量

根据路径导入锚框和锚框的数量

判断pretrained是否使用backbone的预训练权重

创建 yolobody 模型

判断pretrained，是否加载与训练模型，没有的话直接初始化

损失函数初始化

local_rank=0初始化损失函数的记录

判断是否使用混合精度

train()模型

 判断是否使用多卡训练

进行权值平滑（参数设置为零即为无效，默认为零）

读取train-data 和val-data 编码方式utf-8

打印出config

优化器的选择，以及判断数据集的参数是否合理，数据量够不够

根据是否冻结训练来给batch-size赋值，根据batch-size和公式自适应调整学习率范围

取出权重，用优化器优化

获得学习率下降的方式

获取每个epoch的长度，若epoch的总步长(一个步长为一个batch-size)为0，补充数据集

判断权重平滑更新

自定义dataset打包数据，此时的数据已经符合网络要求

判断多卡训练，dataset的分配

判断local-rank==0 记录val的map

---

至此初始化完成

---



---

模型开始训练

（如若是冻结训练，这里判断冻结训练是否完了，解冻训练，调整优化器学习率的参数

判断多卡，（多卡，则将batch-size平分给每一张卡训练

得到数据集迭代器

获得当前epoch，根据epoch调整学习率



调用 fit_one_epoch 完成前向传播、反向传播



打印进度显示 loss lr

批量取出train迭代器和数据  数据为 ： batch-size图片   targets(框框)    y_true(get-targets)

加载cuda数据

梯度清零

for循环

### 		前向传播

​		利用model类得到网络三个输出

​		循环迭代，根据输出设定锚框。利用yolo-loss的类计算损失

### 		反向传播

​		混合精度，scaler 反向 优化器 更新

​		标签平滑

​		pbar将损失存入字典中

提示一个train 的 epoch完成，开始测试 val 并显示val进度条

权值平滑



for循环，批量取出val迭代器和数据  数据为 ： batch-size图片   targets(框框)    y_true(get-targets)

​		没有梯度

​		使用cuda

​		清零优化器梯度

​		前向传播

​		计算损失

​		进度显示，pbar更新

画曲线图

- losshistory文件后面添加loss
- eval_callback后添加map

保存最好的权重

保存最后的权重

