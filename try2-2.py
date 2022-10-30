import torch
from tqdm import tqdm
import time
import os
import numpy as np
import random

import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 300)
cap.set(4, 300)  # 设置摄像头分辨率，3为高，4为宽

while True:
    ret, frame = cap.read()

    # 设置显示的窗口大小为500,500，建议大于等于摄像头分辨率
    cv2.resizeWindow("camera", 500, 500)

    cv2.imshow("camera", frame)

    # 移动当前显示窗口至（0,0）
    cv2.moveWindow("camera", 0, 0)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

# 整型随机数
# bbb = [10, 11, 12, 13, 14, 15, 16]
# c = range(len(bbb))
# indexs = random.sample(c, 1)
#
#
# a = np.asarray(bbb)[indexs][0]
# a = str(a)
# b = os.path.join(a + '.jpg')
# print(b)

# a = torch.tensor([[1., 2., 1.],[1., 1., 2.]])
# b = torch.tensor([[[1., 2., 1.], [2., 1., 1.]], [[2., 1., 2.], [2., 1., 1.]]])
#
# x1, x2 = a.size()
#
# a1 = a.view(x1, 1, x2)
#
# d = b*a1
#
# print('a: ', a)
# print('a1: ', a1)
# print('b: ', b)
# print('d: ', d)
# print(a.shape)
# print(a1.shape)
# print(b.shape)
# print(d.shape)



# txt = input()
# a, b = txt.split(',')
# a = int(a)
# b = int(b)
# c = a + b
# print(f'{c}')
#
'''
run:
3,4
7
'''
# a = torch.tensor([1, 2, 2, 3.], requires_grad=True)
# b = torch.tensor([4, 5, 6.])
# c = torch.ones_like(b)*2
# print(c)


# b = out.detach()
# print(b)
#
# c = torch.where(a[1]==2, b, torch.zeros_like(a))
# print(c)





# x = torch.randint(10, (3, 3, 3, 3)).numpy()

# print("==="*30)
# print(x, x.shape)
# print("==="*30)
# a = x[..., 4]
# print("ss", a, a.shape)
# print("==="*30)
# b = x[a == 7]
# print("ssss", b, b.shape)
# print("==="*30)
#
# y = torch.randint(10, (3, 3, 3, 3)).numpy()
#
# print(y)
# print("==="*30)
# print(x[y == 4])
#

# mask = x.ge(0.5)  # 大于0.5为true
#
# print(mask)
# print(torch.masked_select(x, mask))
#
# print(x[x[..., 0]==1])
