# import torch
# import numpy as np
# import torch.nn as nn
# import os
# from multiprocessing import Process
# from PIL import Image
# import random
# import math

# from torch.nn import functional as F
# # %matplotlib inline
# import pandas as pd
# from scipy import stats
# import matplotlib.pyplot as plt
# from utils.activations import FReLU, Hardswish
# from torchsummary import summary
# import glob
# # from __future__ import absolute_import


# import cv2
# #  捕捉视频画面NBNB
# videoCaputer = cv2.VideoCapture(0)
# cap=videoCaputer
# while(cap.isOpened()):

#     ret, frame = cap.read()
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# print("well done !")

import torch

a = torch.tensor([1, 2, 3])
print(a)


'''
from torchvision.transforms import *

import cv2
import numpy as np
import torch


# 随机擦除
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.  小黑块的数量
        length (int): The length (in pixels) of each square patch.  小黑快的长度
    """

    def __init__(self, n_holes, length, fill_value):
        self.n_holes = n_holes
        self.length = length
        self.fill_value = fill_value

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
    img = cv2.imread("img/1_2.jpg")
    print(img.shape)
    save_path = "1_21.jpg"
    transform = Compose([
        transforms.ToTensor(),
        Cutout(n_holes=3, length=70, fill_value=20.)  # n_holes控制数量 length 控制擦除范围，fill_value控制用什么值进行擦除
    ])
    # transform暂且把他看作一个已经实例化的函数，将多步骤给合在一起的函数
    img2 = transform(img=img)
    print(img2.shape)
    img2 = img2.numpy().transpose([1, 2, 0])
    print(img2.shape)
    cv2.imwrite(save_path, img2)
    cv2.imshow("test", img2)
    cv2.waitKey(0)
'''


# model = RandomErasing()
#
# if __name__ == "__main__":
#     file = 'img/1_2.jpg'
#     save_path = "1_21.jpg"
#
#     img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
#     img = model(img)
#     cv2.imwrite(save_path, img)
#     cv2.imshow("image", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

'''
class RandomErasing(object):
    
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
        #     return img

        for attempt in range(100):

            print(img.size)
            print(img.shape)
            area = img[0] * img[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = np.round(np.sqrt(target_area * aspect_ratio))
            w = np.round(np.sqrt(target_area / aspect_ratio))
            print(h, w)

            if w < img[1] and h < img[0]:
                x1 = random.randint(0, img[0] - h)
                y1 = random.randint(0, img[1] - w)
                if img.size()[2] == 3:
                    img[3, x1:x1 + h, y1:y1 + w] = self.mean[3]
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                else:
                    img[3, x1:x1 + h, y1:y1 + w] = self.mean[3]
                return img

        return img
'''


'''
打开某目录下 所有的某文件
ground_truth_files_list = glob.glob("D:\shuqikaohe\yolov5-pytorch-main" + '/*.txt')  # 获取该文件夹下的.txt file

for txt_file in ground_truth_files_list:
    file_id = txt_file.split(".txt", 1)[0]
    file = os.path.normpath(file_id)
    file_id = os.path.basename(os.path.normpath(file_id))
    print(file)
    print(file_id)

'''


'''
VOC格式数据集的访问

import xml.etree.ElementTree as ET

with open('sample.xml', 'r') as f:
    tree = ET.parse(f)

root = tree.getroot()
print(root)

root
# <Element 'annotation' at 0x107577a98>

list(root)
# <Element 'folder' at 0x107577908>
# <Element 'filename' at 0x107577ef8>
# <Element 'size' at 0x107577868>
# <Element 'segmented' at 0x107577d18>
# <Element 'object' at 0x1075777c8>
# <Element 'object' at 0x1075775e8>
# <Element 'object' at 0x10741dcc8>

size = root.find('size')  # 获取子元素
print(size)
w = int(size.find('width').text)  # 读取值
print(w)
h = int(size.find('height').text)
print(h)

for obj in root.iter('object'):  # 多个元素
    difficult = obj.find('difficult').text
    cls = obj.find('name').text
    print(obj)
    print(difficult)
    print(cls)

'''

'''
file = 'img/1_2.jpg'
save_path = "1_21.jpg"

img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
sigma = 0
kernel_size = (5, 5)
img = cv2.GaussianBlur(img, kernel_size, sigma)
cv2.imwrite(save_path, img)

print("The picture is currently being processed")
'''

'''
// 掩码的操作

arr1 = torch.randn(10, 15)
arr2 = torch.randn(2, 3, 4)

a, b =torch.max(arr1[:, 5:15], 1, keepdim=True)
print(arr1[:, 5:15])
print(a)
print(b)
print(arr1[:, 4])
print(a[:, 0])
print(arr1[:, 4]*a[:, 0])
mask = (arr1[:, 4]*a[:, 0]>=0.5).squeeze()
print(mask)

arr1_m = arr1[mask]
a_m = a[mask]
b_m = b[mask]
print(arr1_m)
print(a_m)
print(b_m)

'''


'''//  深度可分离卷积的添加

# 自定义自动padding模块
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        # auto-pad #如果k是整数，p为k与2整除后向下取整；如果k是列表等，p对应的是列表中每个元素整除2。
    return p



class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # p 通过 autopad函数 自适应，保持shape对应
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        # self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())  # 加入非线性因素
        self.act = FReLU(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act    = Hardswish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # 加入非线性因素  frelu目前除了召回率其他都比baseline好的模型

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))  # 完整的经过一层卷积操作 即，conv + bn + act

class depth_conv(nn.Module):
    def __init__(self, ch_in, ch_out,k=1, s=1, p=None, act=True):
        super(depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(ch_in, ch_in, k, s, autopad(k, p), groups=ch_in, bias=False)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.act = FReLU(ch_out) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.act(x)
        return x




class mymodel(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.Dark = depth_conv(base_channels, base_channels * 2, 3, 2)

    def forward(self, x):
        x = self.Dark(x)

        return x


device = torch.device("cuda")
model = mymodel(4).to(device)
print(model)
summary(model, (4, 20, 20))  # 模型总结函数会自己打印

a = torch.randn(2, 4, 20, 20).to(device)

if __name__ == "__main__":
    b = model(a)
    print("---"*30)
    print(a.shape)
    print(b.shape)

'''
'''

class depth_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.relu(x)
        return x

class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = depth_conv(4, 4)

    def forward(self, x):
        x = self.conv2(x)

        return x


device = torch.device("cuda")  # 设置机器
model = mymodel().to(device)  # 将模型搬到GPU上，返回GPU上的模型
# summary(model, (4, 3, 3))  # 模型总结函数会自己打印

a = torch.randn(2, 4, 20, 20).to(device)  # 转化为cuda的数据类型

if __name__ == "__main__":
    b = model(a)
    print("---"*30)
    print(a.shape)
    print(b.shape)
'''


# content = torch.load('weights/best_epoch_weights.pth')
# print(content.keys())

'''


class Hardswish(nn.Module):
    # Hard-SiLU activation
    @staticmethod
    def forward(x):
        return x * F.hardsigmoid(x)  # for TorchScript and CoreML
        # return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for TorchScript, CoreML and ONNX

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.linspace(-10,10)
x = torch.tensor(x)
##### 绘制sigmoid图像
fig = plt.figure()
y_sigmoid = x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0
ax = fig.add_subplot(321)
ax.plot(x,y_sigmoid,color='blue')
ax.grid()
# ax.set_title('F.hardtanh(x + 3, 0.0, 6.0) / 6.0')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
#
##### 绘制Tanh图像
ax = fig.add_subplot(322)
y_tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
ax.plot(x,y_tanh,color='blue')
ax.grid()
ax.set_title('(b) Tanh')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

##### 绘制Relu图像
ax = fig.add_subplot(323)
y_relu = np.array([0*item  if item<0 else item for item in x ])
ax.plot(x,y_relu,color='darkviolet')
ax.grid()
ax.set_title('(c) ReLu')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

##### 绘制Leaky Relu图像
ax = fig.add_subplot(324)
y_relu = np.array([0.2*item  if item<0 else item for item in x ])
ax.plot(x,y_relu,color='darkviolet')
ax.grid()
ax.set_title('(d) Leaky Relu')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

##### 绘制ELU图像
ax = fig.add_subplot(325)
y_elu = np.array([2.0*(np.exp(item)-1)  if item<0 else item for item in x ])
ax.plot(x,y_elu,color='darkviolet')
ax.grid()
ax.set_title('(d) ELU alpha=2.0')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


ax = fig.add_subplot(326)
y_sigmoid_dev = x * sigmoid(x)
ax.plot(x,y_sigmoid_dev,color='green')
ax.grid()
ax.set_title('(e) SiLU')
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

plt.tight_layout()
plt.savefig('Activation.png')
plt.show()


'''



'''        if n != 0:
            #---------------------------------------------------------------#
            #   计算预测结果和真实结果的giou，计算对应有真实框的先验框的giou损失
            #   y_true[..., 4] == 1 取出有物体的框， loss_cls计算对应有真实框的先验框的分类损失
            #----------------------------------------------------------------#
            giou        = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
            print("===" * 30)  #device='cuda:0', dtype=torch.float16, grad_fn=<ToCopyBackward0>
            print(giou)
            loss_loc    = torch.mean((1 - giou)[y_true[..., 4] == 1])
            loss_cls    = torch.mean(self.BCELoss(pred_cls[y_true[..., 4] == 1], self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1], self.label_smoothing, self.num_classes)))
            aaa = y_true.cpu().numpy()
            loss        += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
            #-----------------------------------------------------------#
            #   计算置信度的loss
            #   也就意味着先验框对应的预测框预测的更准确
            #   它才是用来预测这个物体的。
            #-----------------------------------------------------------#
            tobj        = torch.where(y_true[..., 4] == 1, giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))  # (12, 3, 20, 20)
            print("==="*30)  
            print(tobj)   # device='cuda:0', dtype=torch.float16
        else:
            tobj        = torch.zeros_like(y_true[..., 4])
        print("==="*30)
        print(conf)    # device='cuda:0', dtype=torch.float16, grad_fn=<SigmoidBackward0>
        loss_conf   = torch.mean(self.BCELoss(conf, tobj))  # (12, 3, 20, 20)
        print("===" * 30)
        print(loss_conf)   # device='cuda:0', grad_fn=<MeanBackward0>
        print("===" * 30)



        if self.focal_loss:
            pos_neg_ratio = torch.where(y_true[..., 4] == 1, torch.ones_like(conf) * self.alpha, torch.ones_like(conf) * (1 - self.alpha))
            hard_easy_ratio = torch.where(y_true[..., 4] == 1, torch.ones_like(conf) - conf, conf) ** self.gamma
            loss_conf = torch.mean((self.BCELoss(conf, tobj) * pos_neg_ratio * hard_easy_ratio)) * self.focal_loss_ratio
        else:
            loss_conf = torch.mean(self.BCELoss(conf, tobj))  # (12, 3, 20, 20)

            





'''







# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 4))
# plt.scatter(2, 4)
# plt.show()
#


#
# # 元素树
# import xml.etree.ElementTree as ET
#
# tree = ET.parse('sample.xml')
# print(tree)
# root = tree.getroot()
# print(root)
# for obj in root.iter('object'):
#     print(obj)


# <xml.etree.ElementTree.ElementTree object at 0x0000022D01F3C730>
# <Element 'data' at 0x0000022D0B3119A0>
#

# # anchor的结果预测   去留判定
# a = torch.tensor([[0., 2., 1., 2.],
#                  [1., 2., 0., 1.],
#                  [0., 2., 1., 0.]])
# print(a)
# mark = [True, False, True]
# mark1 = [1., 0., 2.]
#
# print(a[:, -1])
# print(a[a[:, -1] == 1.])



# mark和unique用法
# a = torch.tensor([[0., 2., 1., 2.],
#                  [1., 2., 0., 1.],
#                  [0., 2., 1., 0.]])
# print(a)
# mark = [True, False, True]
# print(a.unique())  # 不重复取出所有的值，按升序排列
# print(a[mark])  # 去除对应维度的值

# tensor([[0., 2., 1., 2.],
#         [1., 2., 0., 1.],
#         [0., 2., 1., 0.]])
# tensor([0., 1., 2.])
# tensor([[0., 2., 1., 2.],
#         [0., 2., 1., 0.]])





# # yolov5先验框的形成
# anchors = [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]
# a = torch.randn(1, 3, 20, 20, 80)
# # print(a)
# print(a.shape)
# print(a[..., 0].shape)
# a1 = torch.sigmoid(a[..., 0])
# # print(a1)
# # print(a1.shape)
# FloatTensor = torch.cuda.FloatTensor if a.is_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if a.is_cuda else torch.LongTensor
#
# anchor_w = FloatTensor(anchors).index_select(1, LongTensor([0]))
# anchor_h = FloatTensor(anchors).index_select(1, LongTensor([1]))
# print(anchor_h)
# anchor_w1 = anchor_w.repeat(1, 1)
# print(anchor_w1.shape)
# anchor_w2 = anchor_w.repeat(1, 1).repeat(1, 1, 20 * 20)
# print(anchor_w2.shape)
# anchor_w3 = anchor_w.repeat(1, 1).repeat(1, 1, 20 * 20).view(a1.shape)
# print(anchor_w3.shape)
# anchor_h = anchor_h.repeat(1, 1).repeat(1, 1, 20 * 20).view(a1.shape)

# torch.Size([1, 3, 20, 20, 80])
# torch.Size([1, 3, 20, 20])
# tensor([[ 2.8125],
#         [ 6.1875],
#         [10.1875]])
# torch.Size([3, 1])
# torch.Size([1, 3, 400])
# torch.Size([1, 3, 20, 20])




# 、、、、、focus特征提取原理实现，保留了特征![](C:/Users/Happy/AppData/Roaming/Tencent/QQ/Temp/)M14SEALAJW$TP)X%O~E)~P.png)
# a = torch.randn(1, 1, 4, 4)  # [b, c, h, w]不要看成[c, h, w],因为一般会批量处理

# print(a)
# print(a[..., ::2, ::2])  # 由第0行，第0列开始，横着隔一个，竖着隔一个取
# print(a[..., 1::2, ::2])  # 第1行，第0列开始，横着隔一个，竖着隔一个取
# print(a[..., ::2, 1::2])  # 第0行，第1列开始，横着隔一个，竖着隔一个取
# print(a[..., 1::2, 1::2])  # 第1行，第1列开始，横着隔一个，竖着隔一个取
# b = torch.cat([a[..., ::2, ::2],
#                a[..., 1::2, ::2],
#                a[..., ::2, 1::2],
#                a[..., 1::2, 1::2],
#                ], 1)
#
# print(a.shape)
# print(b.shape)
# print(b)
# image_shape = np.array(np.shape(a)[0:2])
# print(np.shape(a))
# print(np.shape(a)[0:2])
# print(np.array(np.shape(a)[0:2]))
# print(a)
# print(image_shape)
#


# 、、、、、试验，控制可视图片的大小的调整
# from PIL import Image
#
# im = Image.open(r"E:\DL_environments\torch\bubbliiiing\yolov5-pytorch-main\img\1-1.png")
# width, height = im.size
# print(np.shape(im))
#
# left = 1
# top = height / 110  # 分母越大，图片高度越高
# right = 1700
# bottom = 3 * height / 3  # 分母越小，图片低处越低
# im1 = im.crop((left, top, right, bottom))
# newsize = (2000, 1000)
# im1 = im1.resize(newsize)
# im1.show()


# 、、、、、不失真的resize 灰图pading
# from PIL import Image
#
#
# def letterbox_image(image, size):
#     # 对图片进行resize，使图片不失真。在空缺的地方进行padding
#     iw, ih = image.size  # 宽大于高
#     w, h = size
#     scale = min(w / iw, h / ih)
#     print(scale)
#     print(iw*scale)
#     print(ih*scale)
#     nw = int(iw * scale)
#     nh = int(ih * scale)
#
#     image = image.resize((nw, nh), Image.BICUBIC)  # Image.BICUBIC：图像插值-双三次插值  可认为在原图像后加灰色图，根据指定维度phi
#     new_image = Image.new('RGB', size, (128, 128, 128))
#     new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
#     return new_image
#
#、、、、、
# img = Image.open(r"img\1-1.jpg")
# new_image = letterbox_image(img, [640, 640])
# new_image.show()
#

# from PIL import Image
# import matplotlib.pyplot as plt
#
# img = Image.new("RGB",(500,500),(128,128,128)) # 128为灰度图
# img.show()


# 、、、、matplotlib画图在pycharm中要show才看得见
# import numpy as np
# from matplotlib import pyplot as plt
#
# x = np.arange(0, 35)
# y = 2 * x + 5
# plt.title("Matplotlib demo")
# plt.xlabel("x axis caption")
# plt.ylabel("y axis caption")
# plt.plot(x, y)
# plt.show()















