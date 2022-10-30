# yolov5-v5.0目标检测，pytorch框架实现

---

**注意**：此模型的baseline是[bubbliiiing](https://github.com/bubbliiiing)这个up主的yolov5模型。

# 目录

1. [information](#information)
2. [所需环境](#所需环境)
3. [数据集及权值](#数据集及权值)
4. [模型使用](#模型使用)
5. [评估步骤](#评估步骤)

# information

此模型是yolov5模型，按流程配置好环境后可以直接训练测试。此项目主要是网络优化，提高网络性能，以及一些简单的前后端部署（后面添上）。

此项目支持多卡训练，图片检测，视频检测（可实时监测），heatmap，get_map.py可以得到平均精度，平均F1 recall precision以及但各类的对应指标等。

b导原来设置的功能基本保留。查看源地址 [点击这里](https://github.com/bubbliiiing/yolov5-pytorch)

# 所需环境

其中cuda可以为 9.2 10.1 10.2 11.0， pytorch建议1.7.1及以上。

```
# CUDA 11.0
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```



# 数据集及权值

详情，请前往B导的GitHub：https://github.com/bubbliiiing/efficientnet-yolo3-pytorch

# 模型使用



1. 将库fork到本地，下载配置文件配置环境

   

2. 此项目默认VOC格式的数据集，在yolov5_pytorch_main（根目录）下创建VOCdevket，请将下载的数据集解压到到VOCdeviket目录下，修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   

   

3. 开始网络训练   train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。

   

4. 训练epoch根据优化器 sgd 300epoch，adam 100 epoch，没有上限要求。   

   

5. 训练结果预测   训练结果预测需要用到两个文件，分别是yolo.py和predict.py。我们首先需要去yolo.py里面修改model_path以及classes_path，这两个参数必须要修改（如果是相对路径就不改也行）。**model_path指向训练好的权值文件，在logs文件夹里。   classes_path指向检测类别所对应的txt（也在logs文件里）。**   完成修改后就可以运行predict.py进行检测了。运行后在pycharm的terminal（pycharm的终端）输入图片路径即可检测。   



## 评估步骤 

### 一、评估VOC07+12的测试集

1. 本文使用VOC格式进行评估。VOC07+12已经划分好了测试集，无需利用voc_annotation.py生成ImageSets文件夹下的txt。
2. 在yolo.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
3. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。注意map_out文件有的话会被覆盖，没有则会新建。



## Reference

https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  
https://github.com/ultralytics/yolov5   
