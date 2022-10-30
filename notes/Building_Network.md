# Mobilenet

https://arxiv.org/pdf/1704.04861v1.pdf



### 标准卷积

![image-20220814213415360](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220814213415360.png)

标准卷积输入与输出公式
$$
G{_k,_l,_n} = \sum_{i,j,m}K_{i,j,m,n} \cdot F_{k+i-1},_{l+j-1,m} \qquad(1)
$$
F为输入featuremap，[h, w, c] 即 ： 高 h = k + i - 1，宽 w = l + j -1，channel = m

K为 filter ， [ i, l, m ]

G 为输出map [ k, l, n ]

点 ：即为滑动窗口对应featuremap各像素点乘积

求和 ：即滑动窗口对应featuremap各像素点乘积 求和 为输出map的一个像素点



### 深度可分离卷积

![image-20220814213536583](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220814213536583.png)



![image-20220814213508855](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220814213508855.png)



其公式为
$$
\hat{G}_{k,l,m} = \sum_{i,j}\hat{K}_{i,j,m}\cdot F_{k+i-1},_{l+j-1,m}\qquad(2)
$$



尝试步骤

1. 全部可分离卷积
2. 融合自己的想法，
3. 融合自己的想法，相加
4. 融合自己的想法，cat
5. 在较好的基础上，改进着重关注突出特征



标准卷积

![image-20220817130325059](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220817130325059.png)



![image-20220817130330697](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220817130330697.png)



深度可分离卷积

![image-20220817130334994](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220817130334994.png)



```python
YoloBody(
  (backbone): CSPDarknet(
    (stem): Sequential(
      (0): CBA(
        (conv1): Conv(
          (conv): Conv2d(3, 6, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(6, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(6, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
            (bn): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (conv2): Conv(
          (conv): Conv2d(6, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (1): depth_conv(
        (dep_conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (poi_conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (dark2): Sequential(
      (0): depth_conv(
        (dep_conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
        (poi_conv): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): C3(
        (cv1): depth_conv(
          (dep_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64, bias=False)
          (poi_conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): depth_conv(
          (dep_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64, bias=False)
          (poi_conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): depth_conv(
              (dep_conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), groups=32, bias=False)
              (poi_conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): depth_conv(
              (dep_conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
              (poi_conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
        )
      )
    )
    (dark3): Sequential(
      (0): depth_conv(
        (dep_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
        (poi_conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): C3(
        (cv1): depth_conv(
          (dep_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), groups=128, bias=False)
          (poi_conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): depth_conv(
          (dep_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), groups=128, bias=False)
          (poi_conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): depth_conv(
              (dep_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64, bias=False)
              (poi_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): depth_conv(
              (dep_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              (poi_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (1): Bottleneck(
            (cv1): depth_conv(
              (dep_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64, bias=False)
              (poi_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): depth_conv(
              (dep_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              (poi_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
        )
      )
    )
    (dark4): Sequential(
      (0): depth_conv(
        (dep_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
        (poi_conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): C3(
        (cv1): depth_conv(
          (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
          (poi_conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): depth_conv(
          (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
          (poi_conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): depth_conv(
              (dep_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), groups=128, bias=False)
              (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): depth_conv(
              (dep_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
              (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (1): Bottleneck(
            (cv1): depth_conv(
              (dep_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), groups=128, bias=False)
              (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): depth_conv(
              (dep_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
              (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
        )
      )
    )
    (dark5): Sequential(
      (0): depth_conv(
        (dep_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False)
        (poi_conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): SPP(
        (cv1): depth_conv(
          (dep_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=512, bias=False)
          (poi_conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): ModuleList(
          (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
          (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
          (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
        )
      )
      (2): C3(
        (cv1): depth_conv(
          (dep_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=512, bias=False)
          (poi_conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): depth_conv(
          (dep_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=512, bias=False)
          (poi_conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): depth_conv(
              (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
              (poi_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): depth_conv(
              (dep_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
              (poi_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
              (act): FReLU(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
        )
      )
    )
  )
  (upsample): Upsample(scale_factor=2.0, mode=nearest)
  (conv_for_feat3): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): FReLU(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (conv3_for_upsample1): C3(
    (cv1): depth_conv(
      (dep_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=512, bias=False)
      (poi_conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): depth_conv(
      (dep_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=512, bias=False)
      (poi_conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): depth_conv(
          (dep_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), groups=128, bias=False)
          (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): depth_conv(
          (dep_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
    )
  )
  (conv_for_feat2): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): FReLU(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (conv3_for_upsample2): C3(
    (cv1): depth_conv(
      (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
      (poi_conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): depth_conv(
      (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
      (poi_conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): depth_conv(
          (dep_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64, bias=False)
          (poi_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): depth_conv(
          (dep_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (poi_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
    )
  )
  (down_sample1): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): FReLU(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (conv3_for_downsample1): C3(
    (cv1): depth_conv(
      (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
      (poi_conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): depth_conv(
      (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
      (poi_conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): depth_conv(
          (dep_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), groups=128, bias=False)
          (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): depth_conv(
          (dep_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
    )
  )
  (down_sample2): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): FReLU(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (conv3_for_downsample2): C3(
    (cv1): depth_conv(
      (dep_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=512, bias=False)
      (poi_conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): depth_conv(
      (dep_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=512, bias=False)
      (poi_conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): depth_conv(
          (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
          (poi_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): depth_conv(
          (dep_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (poi_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
    )
  )
  (yolo_head_P3): Conv2d(128, 75, kernel_size=(1, 1), stride=(1, 1))
  (yolo_head_P4): Conv2d(256, 75, kernel_size=(1, 1), stride=(1, 1))
  (yolo_head_P5): Conv2d(512, 75, kernel_size=(1, 1), stride=(1, 1))
)
torch.Size([2, 32, 320, 320])
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 6, 320, 320]             162
       BatchNorm2d-2          [-1, 6, 320, 320]              12
            Conv2d-3          [-1, 6, 320, 320]              54
       BatchNorm2d-4          [-1, 6, 320, 320]              12
             FReLU-5          [-1, 6, 320, 320]               0
              Conv-6          [-1, 6, 320, 320]               0
            Conv2d-7         [-1, 32, 320, 320]             192
       BatchNorm2d-8         [-1, 32, 320, 320]              64
            Conv2d-9         [-1, 32, 320, 320]             288
      BatchNorm2d-10         [-1, 32, 320, 320]              64
            FReLU-11         [-1, 32, 320, 320]               0
             Conv-12         [-1, 32, 320, 320]               0
              CBA-13         [-1, 32, 320, 320]               0
           Conv2d-14         [-1, 32, 320, 320]             288
           Conv2d-15         [-1, 32, 320, 320]           1,056
           Conv2d-16         [-1, 32, 320, 320]             288
      BatchNorm2d-17         [-1, 32, 320, 320]              64
            FReLU-18         [-1, 32, 320, 320]               0
       depth_conv-19         [-1, 32, 320, 320]               0
           Conv2d-20         [-1, 32, 160, 160]             288
           Conv2d-21         [-1, 64, 160, 160]           2,112
           Conv2d-22         [-1, 64, 160, 160]             576
      BatchNorm2d-23         [-1, 64, 160, 160]             128
            FReLU-24         [-1, 64, 160, 160]               0
       depth_conv-25         [-1, 64, 160, 160]               0
           Conv2d-26         [-1, 64, 160, 160]              64
           Conv2d-27         [-1, 32, 160, 160]           2,080
           Conv2d-28         [-1, 32, 160, 160]             288
      BatchNorm2d-29         [-1, 32, 160, 160]              64
            FReLU-30         [-1, 32, 160, 160]               0
       depth_conv-31         [-1, 32, 160, 160]               0
           Conv2d-32         [-1, 32, 160, 160]              32
           Conv2d-33         [-1, 32, 160, 160]           1,056
           Conv2d-34         [-1, 32, 160, 160]             288
      BatchNorm2d-35         [-1, 32, 160, 160]              64
            FReLU-36         [-1, 32, 160, 160]               0
       depth_conv-37         [-1, 32, 160, 160]               0
           Conv2d-38         [-1, 32, 160, 160]             288
           Conv2d-39         [-1, 32, 160, 160]           1,056
           Conv2d-40         [-1, 32, 160, 160]             288
      BatchNorm2d-41         [-1, 32, 160, 160]              64
            FReLU-42         [-1, 32, 160, 160]               0
       depth_conv-43         [-1, 32, 160, 160]               0
       Bottleneck-44         [-1, 32, 160, 160]               0
           Conv2d-45         [-1, 64, 160, 160]              64
           Conv2d-46         [-1, 32, 160, 160]           2,080
           Conv2d-47         [-1, 32, 160, 160]             288
      BatchNorm2d-48         [-1, 32, 160, 160]              64
            FReLU-49         [-1, 32, 160, 160]               0
       depth_conv-50         [-1, 32, 160, 160]               0
           Conv2d-51         [-1, 64, 160, 160]           2,048
      BatchNorm2d-52         [-1, 64, 160, 160]             128
           Conv2d-53         [-1, 64, 160, 160]             576
      BatchNorm2d-54         [-1, 64, 160, 160]             128
            FReLU-55         [-1, 64, 160, 160]               0
             Conv-56         [-1, 64, 160, 160]               0
               C3-57         [-1, 64, 160, 160]               0
           Conv2d-58           [-1, 64, 80, 80]             576
           Conv2d-59          [-1, 128, 80, 80]           8,320
           Conv2d-60          [-1, 128, 80, 80]           1,152
      BatchNorm2d-61          [-1, 128, 80, 80]             256
            FReLU-62          [-1, 128, 80, 80]               0
       depth_conv-63          [-1, 128, 80, 80]               0
           Conv2d-64          [-1, 128, 80, 80]             128
           Conv2d-65           [-1, 64, 80, 80]           8,256
           Conv2d-66           [-1, 64, 80, 80]             576
      BatchNorm2d-67           [-1, 64, 80, 80]             128
            FReLU-68           [-1, 64, 80, 80]               0
       depth_conv-69           [-1, 64, 80, 80]               0
           Conv2d-70           [-1, 64, 80, 80]              64
           Conv2d-71           [-1, 64, 80, 80]           4,160
           Conv2d-72           [-1, 64, 80, 80]             576
      BatchNorm2d-73           [-1, 64, 80, 80]             128
            FReLU-74           [-1, 64, 80, 80]               0
       depth_conv-75           [-1, 64, 80, 80]               0
           Conv2d-76           [-1, 64, 80, 80]             576
           Conv2d-77           [-1, 64, 80, 80]           4,160
           Conv2d-78           [-1, 64, 80, 80]             576
      BatchNorm2d-79           [-1, 64, 80, 80]             128
            FReLU-80           [-1, 64, 80, 80]               0
       depth_conv-81           [-1, 64, 80, 80]               0
       Bottleneck-82           [-1, 64, 80, 80]               0
           Conv2d-83           [-1, 64, 80, 80]              64
           Conv2d-84           [-1, 64, 80, 80]           4,160
           Conv2d-85           [-1, 64, 80, 80]             576
      BatchNorm2d-86           [-1, 64, 80, 80]             128
            FReLU-87           [-1, 64, 80, 80]               0
       depth_conv-88           [-1, 64, 80, 80]               0
           Conv2d-89           [-1, 64, 80, 80]             576
           Conv2d-90           [-1, 64, 80, 80]           4,160
           Conv2d-91           [-1, 64, 80, 80]             576
      BatchNorm2d-92           [-1, 64, 80, 80]             128
            FReLU-93           [-1, 64, 80, 80]               0
       depth_conv-94           [-1, 64, 80, 80]               0
       Bottleneck-95           [-1, 64, 80, 80]               0
           Conv2d-96          [-1, 128, 80, 80]             128
           Conv2d-97           [-1, 64, 80, 80]           8,256
           Conv2d-98           [-1, 64, 80, 80]             576
      BatchNorm2d-99           [-1, 64, 80, 80]             128
           FReLU-100           [-1, 64, 80, 80]               0
      depth_conv-101           [-1, 64, 80, 80]               0
          Conv2d-102          [-1, 128, 80, 80]           8,192
     BatchNorm2d-103          [-1, 128, 80, 80]             256
          Conv2d-104          [-1, 128, 80, 80]           1,152
     BatchNorm2d-105          [-1, 128, 80, 80]             256
           FReLU-106          [-1, 128, 80, 80]               0
            Conv-107          [-1, 128, 80, 80]               0
              C3-108          [-1, 128, 80, 80]               0
          Conv2d-109          [-1, 128, 40, 40]           1,152
          Conv2d-110          [-1, 256, 40, 40]          33,024
          Conv2d-111          [-1, 256, 40, 40]           2,304
     BatchNorm2d-112          [-1, 256, 40, 40]             512
           FReLU-113          [-1, 256, 40, 40]               0
      depth_conv-114          [-1, 256, 40, 40]               0
          Conv2d-115          [-1, 256, 40, 40]             256
          Conv2d-116          [-1, 128, 40, 40]          32,896
          Conv2d-117          [-1, 128, 40, 40]           1,152
     BatchNorm2d-118          [-1, 128, 40, 40]             256
           FReLU-119          [-1, 128, 40, 40]               0
      depth_conv-120          [-1, 128, 40, 40]               0
          Conv2d-121          [-1, 128, 40, 40]             128
          Conv2d-122          [-1, 128, 40, 40]          16,512
          Conv2d-123          [-1, 128, 40, 40]           1,152
     BatchNorm2d-124          [-1, 128, 40, 40]             256
           FReLU-125          [-1, 128, 40, 40]               0
      depth_conv-126          [-1, 128, 40, 40]               0
          Conv2d-127          [-1, 128, 40, 40]           1,152
          Conv2d-128          [-1, 128, 40, 40]          16,512
          Conv2d-129          [-1, 128, 40, 40]           1,152
     BatchNorm2d-130          [-1, 128, 40, 40]             256
           FReLU-131          [-1, 128, 40, 40]               0
      depth_conv-132          [-1, 128, 40, 40]               0
      Bottleneck-133          [-1, 128, 40, 40]               0
          Conv2d-134          [-1, 128, 40, 40]             128
          Conv2d-135          [-1, 128, 40, 40]          16,512
          Conv2d-136          [-1, 128, 40, 40]           1,152
     BatchNorm2d-137          [-1, 128, 40, 40]             256
           FReLU-138          [-1, 128, 40, 40]               0
      depth_conv-139          [-1, 128, 40, 40]               0
          Conv2d-140          [-1, 128, 40, 40]           1,152
          Conv2d-141          [-1, 128, 40, 40]          16,512
          Conv2d-142          [-1, 128, 40, 40]           1,152
     BatchNorm2d-143          [-1, 128, 40, 40]             256
           FReLU-144          [-1, 128, 40, 40]               0
      depth_conv-145          [-1, 128, 40, 40]               0
      Bottleneck-146          [-1, 128, 40, 40]               0
          Conv2d-147          [-1, 256, 40, 40]             256
          Conv2d-148          [-1, 128, 40, 40]          32,896
          Conv2d-149          [-1, 128, 40, 40]           1,152
     BatchNorm2d-150          [-1, 128, 40, 40]             256
           FReLU-151          [-1, 128, 40, 40]               0
      depth_conv-152          [-1, 128, 40, 40]               0
          Conv2d-153          [-1, 256, 40, 40]          32,768
     BatchNorm2d-154          [-1, 256, 40, 40]             512
          Conv2d-155          [-1, 256, 40, 40]           2,304
     BatchNorm2d-156          [-1, 256, 40, 40]             512
           FReLU-157          [-1, 256, 40, 40]               0
            Conv-158          [-1, 256, 40, 40]               0
              C3-159          [-1, 256, 40, 40]               0
          Conv2d-160          [-1, 256, 20, 20]           2,304
          Conv2d-161          [-1, 512, 20, 20]         131,584
          Conv2d-162          [-1, 512, 20, 20]           4,608
     BatchNorm2d-163          [-1, 512, 20, 20]           1,024
           FReLU-164          [-1, 512, 20, 20]               0
      depth_conv-165          [-1, 512, 20, 20]               0
          Conv2d-166          [-1, 512, 20, 20]             512
          Conv2d-167          [-1, 256, 20, 20]         131,328
          Conv2d-168          [-1, 256, 20, 20]           2,304
     BatchNorm2d-169          [-1, 256, 20, 20]             512
           FReLU-170          [-1, 256, 20, 20]               0
      depth_conv-171          [-1, 256, 20, 20]               0
       MaxPool2d-172          [-1, 256, 20, 20]               0
       MaxPool2d-173          [-1, 256, 20, 20]               0
       MaxPool2d-174          [-1, 256, 20, 20]               0
          Conv2d-175          [-1, 512, 20, 20]         524,288
     BatchNorm2d-176          [-1, 512, 20, 20]           1,024
          Conv2d-177          [-1, 512, 20, 20]           4,608
     BatchNorm2d-178          [-1, 512, 20, 20]           1,024
           FReLU-179          [-1, 512, 20, 20]               0
            Conv-180          [-1, 512, 20, 20]               0
             SPP-181          [-1, 512, 20, 20]               0
          Conv2d-182          [-1, 512, 20, 20]             512
          Conv2d-183          [-1, 256, 20, 20]         131,328
          Conv2d-184          [-1, 256, 20, 20]           2,304
     BatchNorm2d-185          [-1, 256, 20, 20]             512
           FReLU-186          [-1, 256, 20, 20]               0
      depth_conv-187          [-1, 256, 20, 20]               0
          Conv2d-188          [-1, 256, 20, 20]             256
          Conv2d-189          [-1, 256, 20, 20]          65,792
          Conv2d-190          [-1, 256, 20, 20]           2,304
     BatchNorm2d-191          [-1, 256, 20, 20]             512
           FReLU-192          [-1, 256, 20, 20]               0
      depth_conv-193          [-1, 256, 20, 20]               0
          Conv2d-194          [-1, 256, 20, 20]           2,304
          Conv2d-195          [-1, 256, 20, 20]          65,792
          Conv2d-196          [-1, 256, 20, 20]           2,304
     BatchNorm2d-197          [-1, 256, 20, 20]             512
           FReLU-198          [-1, 256, 20, 20]               0
      depth_conv-199          [-1, 256, 20, 20]               0
      Bottleneck-200          [-1, 256, 20, 20]               0
          Conv2d-201          [-1, 512, 20, 20]             512
          Conv2d-202          [-1, 256, 20, 20]         131,328
          Conv2d-203          [-1, 256, 20, 20]           2,304
     BatchNorm2d-204          [-1, 256, 20, 20]             512
           FReLU-205          [-1, 256, 20, 20]               0
      depth_conv-206          [-1, 256, 20, 20]               0
          Conv2d-207          [-1, 512, 20, 20]         131,072
     BatchNorm2d-208          [-1, 512, 20, 20]           1,024
          Conv2d-209          [-1, 512, 20, 20]           4,608
     BatchNorm2d-210          [-1, 512, 20, 20]           1,024
           FReLU-211          [-1, 512, 20, 20]               0
            Conv-212          [-1, 512, 20, 20]               0
              C3-213          [-1, 512, 20, 20]               0
      CSPDarknet-214  [[-1, 128, 80, 80], [-1, 256, 40, 40], [-1, 512, 20, 20]]               0
          Conv2d-215          [-1, 256, 20, 20]         131,072
     BatchNorm2d-216          [-1, 256, 20, 20]             512
          Conv2d-217          [-1, 256, 20, 20]           2,304
     BatchNorm2d-218          [-1, 256, 20, 20]             512
           FReLU-219          [-1, 256, 20, 20]               0
            Conv-220          [-1, 256, 20, 20]               0
        Upsample-221          [-1, 256, 40, 40]               0
          Conv2d-222          [-1, 512, 40, 40]             512
          Conv2d-223          [-1, 128, 40, 40]          65,664
          Conv2d-224          [-1, 128, 40, 40]           1,152
     BatchNorm2d-225          [-1, 128, 40, 40]             256
           FReLU-226          [-1, 128, 40, 40]               0
      depth_conv-227          [-1, 128, 40, 40]               0
          Conv2d-228          [-1, 128, 40, 40]             128
          Conv2d-229          [-1, 128, 40, 40]          16,512
          Conv2d-230          [-1, 128, 40, 40]           1,152
     BatchNorm2d-231          [-1, 128, 40, 40]             256
           FReLU-232          [-1, 128, 40, 40]               0
      depth_conv-233          [-1, 128, 40, 40]               0
          Conv2d-234          [-1, 128, 40, 40]           1,152
          Conv2d-235          [-1, 128, 40, 40]          16,512
          Conv2d-236          [-1, 128, 40, 40]           1,152
     BatchNorm2d-237          [-1, 128, 40, 40]             256
           FReLU-238          [-1, 128, 40, 40]               0
      depth_conv-239          [-1, 128, 40, 40]               0
      Bottleneck-240          [-1, 128, 40, 40]               0
          Conv2d-241          [-1, 512, 40, 40]             512
          Conv2d-242          [-1, 128, 40, 40]          65,664
          Conv2d-243          [-1, 128, 40, 40]           1,152
     BatchNorm2d-244          [-1, 128, 40, 40]             256
           FReLU-245          [-1, 128, 40, 40]               0
      depth_conv-246          [-1, 128, 40, 40]               0
          Conv2d-247          [-1, 256, 40, 40]          32,768
     BatchNorm2d-248          [-1, 256, 40, 40]             512
          Conv2d-249          [-1, 256, 40, 40]           2,304
     BatchNorm2d-250          [-1, 256, 40, 40]             512
           FReLU-251          [-1, 256, 40, 40]               0
            Conv-252          [-1, 256, 40, 40]               0
              C3-253          [-1, 256, 40, 40]               0
          Conv2d-254          [-1, 128, 40, 40]          32,768
     BatchNorm2d-255          [-1, 128, 40, 40]             256
          Conv2d-256          [-1, 128, 40, 40]           1,152
     BatchNorm2d-257          [-1, 128, 40, 40]             256
           FReLU-258          [-1, 128, 40, 40]               0
            Conv-259          [-1, 128, 40, 40]               0
        Upsample-260          [-1, 128, 80, 80]               0
          Conv2d-261          [-1, 256, 80, 80]             256
          Conv2d-262           [-1, 64, 80, 80]          16,448
          Conv2d-263           [-1, 64, 80, 80]             576
     BatchNorm2d-264           [-1, 64, 80, 80]             128
           FReLU-265           [-1, 64, 80, 80]               0
      depth_conv-266           [-1, 64, 80, 80]               0
          Conv2d-267           [-1, 64, 80, 80]              64
          Conv2d-268           [-1, 64, 80, 80]           4,160
          Conv2d-269           [-1, 64, 80, 80]             576
     BatchNorm2d-270           [-1, 64, 80, 80]             128
           FReLU-271           [-1, 64, 80, 80]               0
      depth_conv-272           [-1, 64, 80, 80]               0
          Conv2d-273           [-1, 64, 80, 80]             576
          Conv2d-274           [-1, 64, 80, 80]           4,160
          Conv2d-275           [-1, 64, 80, 80]             576
     BatchNorm2d-276           [-1, 64, 80, 80]             128
           FReLU-277           [-1, 64, 80, 80]               0
      depth_conv-278           [-1, 64, 80, 80]               0
      Bottleneck-279           [-1, 64, 80, 80]               0
          Conv2d-280          [-1, 256, 80, 80]             256
          Conv2d-281           [-1, 64, 80, 80]          16,448
          Conv2d-282           [-1, 64, 80, 80]             576
     BatchNorm2d-283           [-1, 64, 80, 80]             128
           FReLU-284           [-1, 64, 80, 80]               0
      depth_conv-285           [-1, 64, 80, 80]               0
          Conv2d-286          [-1, 128, 80, 80]           8,192
     BatchNorm2d-287          [-1, 128, 80, 80]             256
          Conv2d-288          [-1, 128, 80, 80]           1,152
     BatchNorm2d-289          [-1, 128, 80, 80]             256
           FReLU-290          [-1, 128, 80, 80]               0
            Conv-291          [-1, 128, 80, 80]               0
              C3-292          [-1, 128, 80, 80]               0
          Conv2d-293          [-1, 128, 40, 40]         147,456
     BatchNorm2d-294          [-1, 128, 40, 40]             256
          Conv2d-295          [-1, 128, 40, 40]           1,152
     BatchNorm2d-296          [-1, 128, 40, 40]             256
           FReLU-297          [-1, 128, 40, 40]               0
            Conv-298          [-1, 128, 40, 40]               0
          Conv2d-299          [-1, 256, 40, 40]             256
          Conv2d-300          [-1, 128, 40, 40]          32,896
          Conv2d-301          [-1, 128, 40, 40]           1,152
     BatchNorm2d-302          [-1, 128, 40, 40]             256
           FReLU-303          [-1, 128, 40, 40]               0
      depth_conv-304          [-1, 128, 40, 40]               0
          Conv2d-305          [-1, 128, 40, 40]             128
          Conv2d-306          [-1, 128, 40, 40]          16,512
          Conv2d-307          [-1, 128, 40, 40]           1,152
     BatchNorm2d-308          [-1, 128, 40, 40]             256
           FReLU-309          [-1, 128, 40, 40]               0
      depth_conv-310          [-1, 128, 40, 40]               0
          Conv2d-311          [-1, 128, 40, 40]           1,152
          Conv2d-312          [-1, 128, 40, 40]          16,512
          Conv2d-313          [-1, 128, 40, 40]           1,152
     BatchNorm2d-314          [-1, 128, 40, 40]             256
           FReLU-315          [-1, 128, 40, 40]               0
      depth_conv-316          [-1, 128, 40, 40]               0
      Bottleneck-317          [-1, 128, 40, 40]               0
          Conv2d-318          [-1, 256, 40, 40]             256
          Conv2d-319          [-1, 128, 40, 40]          32,896
          Conv2d-320          [-1, 128, 40, 40]           1,152
     BatchNorm2d-321          [-1, 128, 40, 40]             256
           FReLU-322          [-1, 128, 40, 40]               0
      depth_conv-323          [-1, 128, 40, 40]               0
          Conv2d-324          [-1, 256, 40, 40]          32,768
     BatchNorm2d-325          [-1, 256, 40, 40]             512
          Conv2d-326          [-1, 256, 40, 40]           2,304
     BatchNorm2d-327          [-1, 256, 40, 40]             512
           FReLU-328          [-1, 256, 40, 40]               0
            Conv-329          [-1, 256, 40, 40]               0
              C3-330          [-1, 256, 40, 40]               0
          Conv2d-331          [-1, 256, 20, 20]         589,824
     BatchNorm2d-332          [-1, 256, 20, 20]             512
          Conv2d-333          [-1, 256, 20, 20]           2,304
     BatchNorm2d-334          [-1, 256, 20, 20]             512
           FReLU-335          [-1, 256, 20, 20]               0
            Conv-336          [-1, 256, 20, 20]               0
          Conv2d-337          [-1, 512, 20, 20]             512
          Conv2d-338          [-1, 256, 20, 20]         131,328
          Conv2d-339          [-1, 256, 20, 20]           2,304
     BatchNorm2d-340          [-1, 256, 20, 20]             512
           FReLU-341          [-1, 256, 20, 20]               0
      depth_conv-342          [-1, 256, 20, 20]               0
          Conv2d-343          [-1, 256, 20, 20]             256
          Conv2d-344          [-1, 256, 20, 20]          65,792
          Conv2d-345          [-1, 256, 20, 20]           2,304
     BatchNorm2d-346          [-1, 256, 20, 20]             512
           FReLU-347          [-1, 256, 20, 20]               0
      depth_conv-348          [-1, 256, 20, 20]               0
          Conv2d-349          [-1, 256, 20, 20]           2,304
          Conv2d-350          [-1, 256, 20, 20]          65,792
          Conv2d-351          [-1, 256, 20, 20]           2,304
     BatchNorm2d-352          [-1, 256, 20, 20]             512
           FReLU-353          [-1, 256, 20, 20]               0
      depth_conv-354          [-1, 256, 20, 20]               0
      Bottleneck-355          [-1, 256, 20, 20]               0
          Conv2d-356          [-1, 512, 20, 20]             512
          Conv2d-357          [-1, 256, 20, 20]         131,328
          Conv2d-358          [-1, 256, 20, 20]           2,304
     BatchNorm2d-359          [-1, 256, 20, 20]             512
           FReLU-360          [-1, 256, 20, 20]               0
      depth_conv-361          [-1, 256, 20, 20]               0
          Conv2d-362          [-1, 512, 20, 20]         131,072
     BatchNorm2d-363          [-1, 512, 20, 20]           1,024
          Conv2d-364          [-1, 512, 20, 20]           4,608
     BatchNorm2d-365          [-1, 512, 20, 20]           1,024
           FReLU-366          [-1, 512, 20, 20]               0
            Conv-367          [-1, 512, 20, 20]               0
              C3-368          [-1, 512, 20, 20]               0
          Conv2d-369           [-1, 75, 80, 80]           9,675
          Conv2d-370           [-1, 75, 40, 40]          19,275
          Conv2d-371           [-1, 75, 20, 20]          38,475
================================================================
Total params: 3,575,217
Trainable params: 3,575,217
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.69
Forward/backward pass size (MB): 1391.53
Params size (MB): 13.64
Estimated Total Size (MB): 1409.85
----------------------------------------------------------------
torch.Size([1, 32, 320, 320])
torch.Size([1, 75, 20, 20])
torch.Size([1, 75, 40, 40])
torch.Size([1, 75, 80, 80])

```





![image-20220818094016271](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220818094016271.png)



![image-20220818094038187](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220818094038187.png)



![image-20220818094101234](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220818094101234.png)

![image-20220818094153679](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220818094153679.png)







```python
CSPDarknet(
  (stem): Sequential(
    (0): CBA(
      (conv1): Conv(
        (conv): Conv2d(3, 6, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(6, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(6, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
          (bn): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (conv2): Conv(
        (conv): Conv2d(6, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (1): depth_conv(
      (dep_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
      (poi_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (dark2): Sequential(
    (0): depth_conv(
      (dep_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
      (poi_conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): C3(
      (cv1): depth_conv(
        (dep_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), groups=128, bias=False)
        (poi_conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (cv2): depth_conv(
        (dep_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), groups=128, bias=False)
        (poi_conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (cv3): Conv(
        (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): depth_conv(
            (dep_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), groups=64, bias=False)
            (poi_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (cv2): depth_conv(
            (dep_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (poi_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
      )
    )
  )
  (dark3): Sequential(
    (0): depth_conv(
      (dep_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
      (poi_conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): C3(
      (cv1): depth_conv(
        (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
        (poi_conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (cv2): depth_conv(
        (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
        (poi_conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (cv3): Conv(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): depth_conv(
            (dep_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), groups=128, bias=False)
            (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (cv2): depth_conv(
            (dep_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (1): Bottleneck(
          (cv1): depth_conv(
            (dep_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), groups=128, bias=False)
            (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (cv2): depth_conv(
            (dep_conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (poi_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
      )
    )
  )
  (dark4): Sequential(
    (0): depth_conv(
      (dep_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False)
      (poi_conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): C3(
      (cv1): depth_conv(
        (dep_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=512, bias=False)
        (poi_conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (cv2): depth_conv(
        (dep_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=512, bias=False)
        (poi_conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (cv3): Conv(
        (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): depth_conv(
            (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
            (poi_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (cv2): depth_conv(
            (dep_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (poi_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (1): Bottleneck(
          (cv1): depth_conv(
            (dep_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
            (poi_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (cv2): depth_conv(
            (dep_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (poi_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
      )
    )
  )
  (dark5): Sequential(
    (0): depth_conv(
      (dep_conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=512, bias=False)
      (poi_conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
      (act): FReLU(
        (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024, bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): SPP(
      (cv1): depth_conv(
        (dep_conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), groups=1024, bias=False)
        (poi_conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (cv2): Conv(
        (conv): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024, bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (m): ModuleList(
        (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
      )
    )
    (2): C3(
      (cv1): depth_conv(
        (dep_conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), groups=1024, bias=False)
        (poi_conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (cv2): depth_conv(
        (dep_conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), groups=1024, bias=False)
        (poi_conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): FReLU(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (cv3): Conv(
        (conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024, bias=False)
          (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): depth_conv(
            (dep_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=512, bias=False)
            (poi_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (cv2): depth_conv(
            (dep_conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
            (poi_conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (act): FReLU(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
      )
    )
  )
)
torch.Size([2, 64, 320, 320])
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 6, 320, 320]             162
       BatchNorm2d-2          [-1, 6, 320, 320]              12
            Conv2d-3          [-1, 6, 320, 320]              54
       BatchNorm2d-4          [-1, 6, 320, 320]              12
             FReLU-5          [-1, 6, 320, 320]               0
              Conv-6          [-1, 6, 320, 320]               0
            Conv2d-7         [-1, 64, 320, 320]             384
       BatchNorm2d-8         [-1, 64, 320, 320]             128
            Conv2d-9         [-1, 64, 320, 320]             576
      BatchNorm2d-10         [-1, 64, 320, 320]             128
            FReLU-11         [-1, 64, 320, 320]               0
             Conv-12         [-1, 64, 320, 320]               0
              CBA-13         [-1, 64, 320, 320]               0
           Conv2d-14         [-1, 64, 320, 320]             576
           Conv2d-15         [-1, 64, 320, 320]           4,160
           Conv2d-16         [-1, 64, 320, 320]             576
      BatchNorm2d-17         [-1, 64, 320, 320]             128
            FReLU-18         [-1, 64, 320, 320]               0
       depth_conv-19         [-1, 64, 320, 320]               0
           Conv2d-20         [-1, 64, 160, 160]             576
           Conv2d-21        [-1, 128, 160, 160]           8,320
           Conv2d-22        [-1, 128, 160, 160]           1,152
      BatchNorm2d-23        [-1, 128, 160, 160]             256
            FReLU-24        [-1, 128, 160, 160]               0
       depth_conv-25        [-1, 128, 160, 160]               0
           Conv2d-26        [-1, 128, 160, 160]             128
           Conv2d-27         [-1, 64, 160, 160]           8,256
           Conv2d-28         [-1, 64, 160, 160]             576
      BatchNorm2d-29         [-1, 64, 160, 160]             128
            FReLU-30         [-1, 64, 160, 160]               0
       depth_conv-31         [-1, 64, 160, 160]               0
           Conv2d-32         [-1, 64, 160, 160]              64
           Conv2d-33         [-1, 64, 160, 160]           4,160
           Conv2d-34         [-1, 64, 160, 160]             576
      BatchNorm2d-35         [-1, 64, 160, 160]             128
            FReLU-36         [-1, 64, 160, 160]               0
       depth_conv-37         [-1, 64, 160, 160]               0
           Conv2d-38         [-1, 64, 160, 160]             576
           Conv2d-39         [-1, 64, 160, 160]           4,160
           Conv2d-40         [-1, 64, 160, 160]             576
      BatchNorm2d-41         [-1, 64, 160, 160]             128
            FReLU-42         [-1, 64, 160, 160]               0
       depth_conv-43         [-1, 64, 160, 160]               0
       Bottleneck-44         [-1, 64, 160, 160]               0
           Conv2d-45        [-1, 128, 160, 160]             128
           Conv2d-46         [-1, 64, 160, 160]           8,256
           Conv2d-47         [-1, 64, 160, 160]             576
      BatchNorm2d-48         [-1, 64, 160, 160]             128
            FReLU-49         [-1, 64, 160, 160]               0
       depth_conv-50         [-1, 64, 160, 160]               0
           Conv2d-51        [-1, 128, 160, 160]           8,192
      BatchNorm2d-52        [-1, 128, 160, 160]             256
           Conv2d-53        [-1, 128, 160, 160]           1,152
      BatchNorm2d-54        [-1, 128, 160, 160]             256
            FReLU-55        [-1, 128, 160, 160]               0
             Conv-56        [-1, 128, 160, 160]               0
               C3-57        [-1, 128, 160, 160]               0
           Conv2d-58          [-1, 128, 80, 80]           1,152
           Conv2d-59          [-1, 256, 80, 80]          33,024
           Conv2d-60          [-1, 256, 80, 80]           2,304
      BatchNorm2d-61          [-1, 256, 80, 80]             512
            FReLU-62          [-1, 256, 80, 80]               0
       depth_conv-63          [-1, 256, 80, 80]               0
           Conv2d-64          [-1, 256, 80, 80]             256
           Conv2d-65          [-1, 128, 80, 80]          32,896
           Conv2d-66          [-1, 128, 80, 80]           1,152
      BatchNorm2d-67          [-1, 128, 80, 80]             256
            FReLU-68          [-1, 128, 80, 80]               0
       depth_conv-69          [-1, 128, 80, 80]               0
           Conv2d-70          [-1, 128, 80, 80]             128
           Conv2d-71          [-1, 128, 80, 80]          16,512
           Conv2d-72          [-1, 128, 80, 80]           1,152
      BatchNorm2d-73          [-1, 128, 80, 80]             256
            FReLU-74          [-1, 128, 80, 80]               0
       depth_conv-75          [-1, 128, 80, 80]               0
           Conv2d-76          [-1, 128, 80, 80]           1,152
           Conv2d-77          [-1, 128, 80, 80]          16,512
           Conv2d-78          [-1, 128, 80, 80]           1,152
      BatchNorm2d-79          [-1, 128, 80, 80]             256
            FReLU-80          [-1, 128, 80, 80]               0
       depth_conv-81          [-1, 128, 80, 80]               0
       Bottleneck-82          [-1, 128, 80, 80]               0
           Conv2d-83          [-1, 128, 80, 80]             128
           Conv2d-84          [-1, 128, 80, 80]          16,512
           Conv2d-85          [-1, 128, 80, 80]           1,152
      BatchNorm2d-86          [-1, 128, 80, 80]             256
            FReLU-87          [-1, 128, 80, 80]               0
       depth_conv-88          [-1, 128, 80, 80]               0
           Conv2d-89          [-1, 128, 80, 80]           1,152
           Conv2d-90          [-1, 128, 80, 80]          16,512
           Conv2d-91          [-1, 128, 80, 80]           1,152
      BatchNorm2d-92          [-1, 128, 80, 80]             256
            FReLU-93          [-1, 128, 80, 80]               0
       depth_conv-94          [-1, 128, 80, 80]               0
       Bottleneck-95          [-1, 128, 80, 80]               0
           Conv2d-96          [-1, 256, 80, 80]             256
           Conv2d-97          [-1, 128, 80, 80]          32,896
           Conv2d-98          [-1, 128, 80, 80]           1,152
      BatchNorm2d-99          [-1, 128, 80, 80]             256
           FReLU-100          [-1, 128, 80, 80]               0
      depth_conv-101          [-1, 128, 80, 80]               0
          Conv2d-102          [-1, 256, 80, 80]          32,768
     BatchNorm2d-103          [-1, 256, 80, 80]             512
          Conv2d-104          [-1, 256, 80, 80]           2,304
     BatchNorm2d-105          [-1, 256, 80, 80]             512
           FReLU-106          [-1, 256, 80, 80]               0
            Conv-107          [-1, 256, 80, 80]               0
              C3-108          [-1, 256, 80, 80]               0
          Conv2d-109          [-1, 256, 40, 40]           2,304
          Conv2d-110          [-1, 512, 40, 40]         131,584
          Conv2d-111          [-1, 512, 40, 40]           4,608
     BatchNorm2d-112          [-1, 512, 40, 40]           1,024
           FReLU-113          [-1, 512, 40, 40]               0
      depth_conv-114          [-1, 512, 40, 40]               0
          Conv2d-115          [-1, 512, 40, 40]             512
          Conv2d-116          [-1, 256, 40, 40]         131,328
          Conv2d-117          [-1, 256, 40, 40]           2,304
     BatchNorm2d-118          [-1, 256, 40, 40]             512
           FReLU-119          [-1, 256, 40, 40]               0
      depth_conv-120          [-1, 256, 40, 40]               0
          Conv2d-121          [-1, 256, 40, 40]             256
          Conv2d-122          [-1, 256, 40, 40]          65,792
          Conv2d-123          [-1, 256, 40, 40]           2,304
     BatchNorm2d-124          [-1, 256, 40, 40]             512
           FReLU-125          [-1, 256, 40, 40]               0
      depth_conv-126          [-1, 256, 40, 40]               0
          Conv2d-127          [-1, 256, 40, 40]           2,304
          Conv2d-128          [-1, 256, 40, 40]          65,792
          Conv2d-129          [-1, 256, 40, 40]           2,304
     BatchNorm2d-130          [-1, 256, 40, 40]             512
           FReLU-131          [-1, 256, 40, 40]               0
      depth_conv-132          [-1, 256, 40, 40]               0
      Bottleneck-133          [-1, 256, 40, 40]               0
          Conv2d-134          [-1, 256, 40, 40]             256
          Conv2d-135          [-1, 256, 40, 40]          65,792
          Conv2d-136          [-1, 256, 40, 40]           2,304
     BatchNorm2d-137          [-1, 256, 40, 40]             512
           FReLU-138          [-1, 256, 40, 40]               0
      depth_conv-139          [-1, 256, 40, 40]               0
          Conv2d-140          [-1, 256, 40, 40]           2,304
          Conv2d-141          [-1, 256, 40, 40]          65,792
          Conv2d-142          [-1, 256, 40, 40]           2,304
     BatchNorm2d-143          [-1, 256, 40, 40]             512
           FReLU-144          [-1, 256, 40, 40]               0
      depth_conv-145          [-1, 256, 40, 40]               0
      Bottleneck-146          [-1, 256, 40, 40]               0
          Conv2d-147          [-1, 512, 40, 40]             512
          Conv2d-148          [-1, 256, 40, 40]         131,328
          Conv2d-149          [-1, 256, 40, 40]           2,304
     BatchNorm2d-150          [-1, 256, 40, 40]             512
           FReLU-151          [-1, 256, 40, 40]               0
      depth_conv-152          [-1, 256, 40, 40]               0
          Conv2d-153          [-1, 512, 40, 40]         131,072
     BatchNorm2d-154          [-1, 512, 40, 40]           1,024
          Conv2d-155          [-1, 512, 40, 40]           4,608
     BatchNorm2d-156          [-1, 512, 40, 40]           1,024
           FReLU-157          [-1, 512, 40, 40]               0
            Conv-158          [-1, 512, 40, 40]               0
              C3-159          [-1, 512, 40, 40]               0
          Conv2d-160          [-1, 512, 20, 20]           4,608
          Conv2d-161         [-1, 1024, 20, 20]         525,312
          Conv2d-162         [-1, 1024, 20, 20]           9,216
     BatchNorm2d-163         [-1, 1024, 20, 20]           2,048
           FReLU-164         [-1, 1024, 20, 20]               0
      depth_conv-165         [-1, 1024, 20, 20]               0
          Conv2d-166         [-1, 1024, 20, 20]           1,024
          Conv2d-167          [-1, 512, 20, 20]         524,800
          Conv2d-168          [-1, 512, 20, 20]           4,608
     BatchNorm2d-169          [-1, 512, 20, 20]           1,024
           FReLU-170          [-1, 512, 20, 20]               0
      depth_conv-171          [-1, 512, 20, 20]               0
       MaxPool2d-172          [-1, 512, 20, 20]               0
       MaxPool2d-173          [-1, 512, 20, 20]               0
       MaxPool2d-174          [-1, 512, 20, 20]               0
          Conv2d-175         [-1, 1024, 20, 20]       2,097,152
     BatchNorm2d-176         [-1, 1024, 20, 20]           2,048
          Conv2d-177         [-1, 1024, 20, 20]           9,216
     BatchNorm2d-178         [-1, 1024, 20, 20]           2,048
           FReLU-179         [-1, 1024, 20, 20]               0
            Conv-180         [-1, 1024, 20, 20]               0
             SPP-181         [-1, 1024, 20, 20]               0
          Conv2d-182         [-1, 1024, 20, 20]           1,024
          Conv2d-183          [-1, 512, 20, 20]         524,800
          Conv2d-184          [-1, 512, 20, 20]           4,608
     BatchNorm2d-185          [-1, 512, 20, 20]           1,024
           FReLU-186          [-1, 512, 20, 20]               0
      depth_conv-187          [-1, 512, 20, 20]               0
          Conv2d-188          [-1, 512, 20, 20]             512
          Conv2d-189          [-1, 512, 20, 20]         262,656
          Conv2d-190          [-1, 512, 20, 20]           4,608
     BatchNorm2d-191          [-1, 512, 20, 20]           1,024
           FReLU-192          [-1, 512, 20, 20]               0
      depth_conv-193          [-1, 512, 20, 20]               0
          Conv2d-194          [-1, 512, 20, 20]           4,608
          Conv2d-195          [-1, 512, 20, 20]         262,656
          Conv2d-196          [-1, 512, 20, 20]           4,608
     BatchNorm2d-197          [-1, 512, 20, 20]           1,024
           FReLU-198          [-1, 512, 20, 20]               0
      depth_conv-199          [-1, 512, 20, 20]               0
      Bottleneck-200          [-1, 512, 20, 20]               0
          Conv2d-201         [-1, 1024, 20, 20]           1,024
          Conv2d-202          [-1, 512, 20, 20]         524,800
          Conv2d-203          [-1, 512, 20, 20]           4,608
     BatchNorm2d-204          [-1, 512, 20, 20]           1,024
           FReLU-205          [-1, 512, 20, 20]               0
      depth_conv-206          [-1, 512, 20, 20]               0
          Conv2d-207         [-1, 1024, 20, 20]         524,288
     BatchNorm2d-208         [-1, 1024, 20, 20]           2,048
          Conv2d-209         [-1, 1024, 20, 20]           9,216
     BatchNorm2d-210         [-1, 1024, 20, 20]           2,048
           FReLU-211         [-1, 1024, 20, 20]               0
            Conv-212         [-1, 1024, 20, 20]               0
              C3-213         [-1, 1024, 20, 20]               0
================================================================
Total params: 6,423,472
Trainable params: 6,423,472
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.69
Forward/backward pass size (MB): 2056.25
Params size (MB): 24.50
Estimated Total Size (MB): 2085.44
----------------------------------------------------------------
torch.Size([1, 64, 320, 320])
torch.Size([1, 256, 80, 80])
torch.Size([1, 512, 40, 40])
torch.Size([1, 1024, 20, 20])

Process finished with exit code 0
```

