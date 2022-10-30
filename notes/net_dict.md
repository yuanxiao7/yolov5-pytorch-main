## 网络结构



### print(YloBody)  

- 可以在yolobody初始化以后，直接打印网络结构，一目了然！

原来baseline的网络结构：

```python
YoloBody(
  (backbone): CSPDarknet(
    (stem): Focus(
      (conv): Conv(
        (conv): Conv2d(12, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (dark2): Sequential(
      (0): Conv(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): C3(
        (cv1): Conv(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
      (0): Conv(
        (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): C3(
        (cv1): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (1): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (2): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
      (0): Conv(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): C3(
        (cv1): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (1): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (2): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
      (0): Conv(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): SPP(
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
    (cv1): Conv(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): Conv(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
    (cv1): Conv(
      (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): Conv(
      (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
    (cv1): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
    (cv1): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
```





### summary.py  

- 打印网络结构脚本

表格如下

这里得pai指定的是 ‘s’ 最小的那个模型，可以看到，我在yolo neck得进口处添加了了注意力机制

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 320, 320]           3,456
       BatchNorm2d-2         [-1, 32, 320, 320]              64
              SiLU-3         [-1, 32, 320, 320]               0
              Conv-4         [-1, 32, 320, 320]               0
             Focus-5         [-1, 32, 320, 320]               0
            Conv2d-6         [-1, 64, 160, 160]          18,432
       BatchNorm2d-7         [-1, 64, 160, 160]             128
              SiLU-8         [-1, 64, 160, 160]               0
              Conv-9         [-1, 64, 160, 160]               0
           Conv2d-10         [-1, 32, 160, 160]           2,048
      BatchNorm2d-11         [-1, 32, 160, 160]              64
             SiLU-12         [-1, 32, 160, 160]               0
             Conv-13         [-1, 32, 160, 160]               0
           Conv2d-14         [-1, 32, 160, 160]           1,024
      BatchNorm2d-15         [-1, 32, 160, 160]              64
             SiLU-16         [-1, 32, 160, 160]               0
             Conv-17         [-1, 32, 160, 160]               0
           Conv2d-18         [-1, 32, 160, 160]           9,216
      BatchNorm2d-19         [-1, 32, 160, 160]              64
             SiLU-20         [-1, 32, 160, 160]               0
             Conv-21         [-1, 32, 160, 160]               0
       Bottleneck-22         [-1, 32, 160, 160]               0
           Conv2d-23         [-1, 32, 160, 160]           2,048
      BatchNorm2d-24         [-1, 32, 160, 160]              64
             SiLU-25         [-1, 32, 160, 160]               0
             Conv-26         [-1, 32, 160, 160]               0
           Conv2d-27         [-1, 64, 160, 160]           4,096
      BatchNorm2d-28         [-1, 64, 160, 160]             128
             SiLU-29         [-1, 64, 160, 160]               0
             Conv-30         [-1, 64, 160, 160]               0
               C3-31         [-1, 64, 160, 160]               0
           Conv2d-32          [-1, 128, 80, 80]          73,728
      BatchNorm2d-33          [-1, 128, 80, 80]             256
             SiLU-34          [-1, 128, 80, 80]               0
             Conv-35          [-1, 128, 80, 80]               0
           Conv2d-36           [-1, 64, 80, 80]           8,192
      BatchNorm2d-37           [-1, 64, 80, 80]             128
             SiLU-38           [-1, 64, 80, 80]               0
             Conv-39           [-1, 64, 80, 80]               0
           Conv2d-40           [-1, 64, 80, 80]           4,096
      BatchNorm2d-41           [-1, 64, 80, 80]             128
             SiLU-42           [-1, 64, 80, 80]               0
             Conv-43           [-1, 64, 80, 80]               0
           Conv2d-44           [-1, 64, 80, 80]          36,864
      BatchNorm2d-45           [-1, 64, 80, 80]             128
             SiLU-46           [-1, 64, 80, 80]               0
             Conv-47           [-1, 64, 80, 80]               0
       Bottleneck-48           [-1, 64, 80, 80]               0
           Conv2d-49           [-1, 64, 80, 80]           4,096
      BatchNorm2d-50           [-1, 64, 80, 80]             128
             SiLU-51           [-1, 64, 80, 80]               0
             Conv-52           [-1, 64, 80, 80]               0
           Conv2d-53           [-1, 64, 80, 80]          36,864
      BatchNorm2d-54           [-1, 64, 80, 80]             128
             SiLU-55           [-1, 64, 80, 80]               0
             Conv-56           [-1, 64, 80, 80]               0
       Bottleneck-57           [-1, 64, 80, 80]               0
           Conv2d-58           [-1, 64, 80, 80]           4,096
      BatchNorm2d-59           [-1, 64, 80, 80]             128
             SiLU-60           [-1, 64, 80, 80]               0
             Conv-61           [-1, 64, 80, 80]               0
           Conv2d-62           [-1, 64, 80, 80]          36,864
      BatchNorm2d-63           [-1, 64, 80, 80]             128
             SiLU-64           [-1, 64, 80, 80]               0
             Conv-65           [-1, 64, 80, 80]               0
       Bottleneck-66           [-1, 64, 80, 80]               0
           Conv2d-67           [-1, 64, 80, 80]           8,192
      BatchNorm2d-68           [-1, 64, 80, 80]             128
             SiLU-69           [-1, 64, 80, 80]               0
             Conv-70           [-1, 64, 80, 80]               0
           Conv2d-71          [-1, 128, 80, 80]          16,384
      BatchNorm2d-72          [-1, 128, 80, 80]             256
             SiLU-73          [-1, 128, 80, 80]               0
             Conv-74          [-1, 128, 80, 80]               0
               C3-75          [-1, 128, 80, 80]               0
           Conv2d-76          [-1, 256, 40, 40]         294,912
      BatchNorm2d-77          [-1, 256, 40, 40]             512
             SiLU-78          [-1, 256, 40, 40]               0
             Conv-79          [-1, 256, 40, 40]               0
           Conv2d-80          [-1, 128, 40, 40]          32,768
      BatchNorm2d-81          [-1, 128, 40, 40]             256
             SiLU-82          [-1, 128, 40, 40]               0
             Conv-83          [-1, 128, 40, 40]               0
           Conv2d-84          [-1, 128, 40, 40]          16,384
      BatchNorm2d-85          [-1, 128, 40, 40]             256
             SiLU-86          [-1, 128, 40, 40]               0
             Conv-87          [-1, 128, 40, 40]               0
           Conv2d-88          [-1, 128, 40, 40]         147,456
      BatchNorm2d-89          [-1, 128, 40, 40]             256
             SiLU-90          [-1, 128, 40, 40]               0
             Conv-91          [-1, 128, 40, 40]               0
       Bottleneck-92          [-1, 128, 40, 40]               0
           Conv2d-93          [-1, 128, 40, 40]          16,384
      BatchNorm2d-94          [-1, 128, 40, 40]             256
             SiLU-95          [-1, 128, 40, 40]               0
             Conv-96          [-1, 128, 40, 40]               0
           Conv2d-97          [-1, 128, 40, 40]         147,456
      BatchNorm2d-98          [-1, 128, 40, 40]             256
             SiLU-99          [-1, 128, 40, 40]               0
            Conv-100          [-1, 128, 40, 40]               0
      Bottleneck-101          [-1, 128, 40, 40]               0
          Conv2d-102          [-1, 128, 40, 40]          16,384
     BatchNorm2d-103          [-1, 128, 40, 40]             256
            SiLU-104          [-1, 128, 40, 40]               0
            Conv-105          [-1, 128, 40, 40]               0
          Conv2d-106          [-1, 128, 40, 40]         147,456
     BatchNorm2d-107          [-1, 128, 40, 40]             256
            SiLU-108          [-1, 128, 40, 40]               0
            Conv-109          [-1, 128, 40, 40]               0
      Bottleneck-110          [-1, 128, 40, 40]               0
          Conv2d-111          [-1, 128, 40, 40]          32,768
     BatchNorm2d-112          [-1, 128, 40, 40]             256
            SiLU-113          [-1, 128, 40, 40]               0
            Conv-114          [-1, 128, 40, 40]               0
          Conv2d-115          [-1, 256, 40, 40]          65,536
     BatchNorm2d-116          [-1, 256, 40, 40]             512
            SiLU-117          [-1, 256, 40, 40]               0
            Conv-118          [-1, 256, 40, 40]               0
              C3-119          [-1, 256, 40, 40]               0
          Conv2d-120          [-1, 512, 20, 20]       1,179,648
     BatchNorm2d-121          [-1, 512, 20, 20]           1,024
            SiLU-122          [-1, 512, 20, 20]               0
            Conv-123          [-1, 512, 20, 20]               0
          Conv2d-124          [-1, 256, 20, 20]         131,072
     BatchNorm2d-125          [-1, 256, 20, 20]             512
            SiLU-126          [-1, 256, 20, 20]               0
            Conv-127          [-1, 256, 20, 20]               0
       MaxPool2d-128          [-1, 256, 20, 20]               0
       MaxPool2d-129          [-1, 256, 20, 20]               0
       MaxPool2d-130          [-1, 256, 20, 20]               0
          Conv2d-131          [-1, 512, 20, 20]         524,288
     BatchNorm2d-132          [-1, 512, 20, 20]           1,024
            SiLU-133          [-1, 512, 20, 20]               0
            Conv-134          [-1, 512, 20, 20]               0
             SPP-135          [-1, 512, 20, 20]               0
          Conv2d-136          [-1, 256, 20, 20]         131,072
     BatchNorm2d-137          [-1, 256, 20, 20]             512
            SiLU-138          [-1, 256, 20, 20]               0
            Conv-139          [-1, 256, 20, 20]               0
          Conv2d-140          [-1, 256, 20, 20]          65,536
     BatchNorm2d-141          [-1, 256, 20, 20]             512
            SiLU-142          [-1, 256, 20, 20]               0
            Conv-143          [-1, 256, 20, 20]               0
          Conv2d-144          [-1, 256, 20, 20]         589,824
     BatchNorm2d-145          [-1, 256, 20, 20]             512
            SiLU-146          [-1, 256, 20, 20]               0
            Conv-147          [-1, 256, 20, 20]               0
      Bottleneck-148          [-1, 256, 20, 20]               0
          Conv2d-149          [-1, 256, 20, 20]         131,072
     BatchNorm2d-150          [-1, 256, 20, 20]             512
            SiLU-151          [-1, 256, 20, 20]               0
            Conv-152          [-1, 256, 20, 20]               0
          Conv2d-153          [-1, 512, 20, 20]         262,144
     BatchNorm2d-154          [-1, 512, 20, 20]           1,024
            SiLU-155          [-1, 512, 20, 20]               0
            Conv-156          [-1, 512, 20, 20]               0
              C3-157          [-1, 512, 20, 20]               0
      CSPDarknet-158  [[-1, 128, 80, 80], [-1, 256, 40, 40], [-1, 512, 20, 20]]               0
AdaptiveAvgPool2d-159            [-1, 128, 1, 1]               0
          Linear-160                    [-1, 8]           1,024
            ReLU-161                    [-1, 8]               0
          Linear-162                  [-1, 128]           1,024
         Sigmoid-163                  [-1, 128]               0
        se_block-164          [-1, 128, 80, 80]               0
AdaptiveAvgPool2d-165            [-1, 256, 1, 1]               0
          Linear-166                   [-1, 16]           4,096
            ReLU-167                   [-1, 16]               0
          Linear-168                  [-1, 256]           4,096
         Sigmoid-169                  [-1, 256]               0
        se_block-170          [-1, 256, 40, 40]               0
AdaptiveAvgPool2d-171            [-1, 512, 1, 1]               0
          Linear-172                   [-1, 32]          16,384
            ReLU-173                   [-1, 32]               0
          Linear-174                  [-1, 512]          16,384
         Sigmoid-175                  [-1, 512]               0
        se_block-176          [-1, 512, 20, 20]               0
          Conv2d-177          [-1, 256, 20, 20]         131,072
     BatchNorm2d-178          [-1, 256, 20, 20]             512
            SiLU-179          [-1, 256, 20, 20]               0
            Conv-180          [-1, 256, 20, 20]               0
        Upsample-181          [-1, 256, 40, 40]               0
          Conv2d-182          [-1, 128, 40, 40]          65,536
     BatchNorm2d-183          [-1, 128, 40, 40]             256
            SiLU-184          [-1, 128, 40, 40]               0
            Conv-185          [-1, 128, 40, 40]               0
          Conv2d-186          [-1, 128, 40, 40]          16,384
     BatchNorm2d-187          [-1, 128, 40, 40]             256
            SiLU-188          [-1, 128, 40, 40]               0
            Conv-189          [-1, 128, 40, 40]               0
          Conv2d-190          [-1, 128, 40, 40]         147,456
     BatchNorm2d-191          [-1, 128, 40, 40]             256
            SiLU-192          [-1, 128, 40, 40]               0
            Conv-193          [-1, 128, 40, 40]               0
      Bottleneck-194          [-1, 128, 40, 40]               0
          Conv2d-195          [-1, 128, 40, 40]          65,536
     BatchNorm2d-196          [-1, 128, 40, 40]             256
            SiLU-197          [-1, 128, 40, 40]               0
            Conv-198          [-1, 128, 40, 40]               0
          Conv2d-199          [-1, 256, 40, 40]          65,536
     BatchNorm2d-200          [-1, 256, 40, 40]             512
            SiLU-201          [-1, 256, 40, 40]               0
            Conv-202          [-1, 256, 40, 40]               0
              C3-203          [-1, 256, 40, 40]               0
          Conv2d-204          [-1, 128, 40, 40]          32,768
     BatchNorm2d-205          [-1, 128, 40, 40]             256
            SiLU-206          [-1, 128, 40, 40]               0
            Conv-207          [-1, 128, 40, 40]               0
        Upsample-208          [-1, 128, 80, 80]               0
          Conv2d-209           [-1, 64, 80, 80]          16,384
     BatchNorm2d-210           [-1, 64, 80, 80]             128
            SiLU-211           [-1, 64, 80, 80]               0
            Conv-212           [-1, 64, 80, 80]               0
          Conv2d-213           [-1, 64, 80, 80]           4,096
     BatchNorm2d-214           [-1, 64, 80, 80]             128
            SiLU-215           [-1, 64, 80, 80]               0
            Conv-216           [-1, 64, 80, 80]               0
          Conv2d-217           [-1, 64, 80, 80]          36,864
     BatchNorm2d-218           [-1, 64, 80, 80]             128
            SiLU-219           [-1, 64, 80, 80]               0
            Conv-220           [-1, 64, 80, 80]               0
      Bottleneck-221           [-1, 64, 80, 80]               0
          Conv2d-222           [-1, 64, 80, 80]          16,384
     BatchNorm2d-223           [-1, 64, 80, 80]             128
            SiLU-224           [-1, 64, 80, 80]               0
            Conv-225           [-1, 64, 80, 80]               0
          Conv2d-226          [-1, 128, 80, 80]          16,384
     BatchNorm2d-227          [-1, 128, 80, 80]             256
            SiLU-228          [-1, 128, 80, 80]               0
            Conv-229          [-1, 128, 80, 80]               0
              C3-230          [-1, 128, 80, 80]               0
          Conv2d-231          [-1, 128, 40, 40]         147,456
     BatchNorm2d-232          [-1, 128, 40, 40]             256
            SiLU-233          [-1, 128, 40, 40]               0
            Conv-234          [-1, 128, 40, 40]               0
          Conv2d-235          [-1, 128, 40, 40]          32,768
     BatchNorm2d-236          [-1, 128, 40, 40]             256
            SiLU-237          [-1, 128, 40, 40]               0
            Conv-238          [-1, 128, 40, 40]               0
          Conv2d-239          [-1, 128, 40, 40]          16,384
     BatchNorm2d-240          [-1, 128, 40, 40]             256
            SiLU-241          [-1, 128, 40, 40]               0
            Conv-242          [-1, 128, 40, 40]               0
          Conv2d-243          [-1, 128, 40, 40]         147,456
     BatchNorm2d-244          [-1, 128, 40, 40]             256
            SiLU-245          [-1, 128, 40, 40]               0
            Conv-246          [-1, 128, 40, 40]               0
      Bottleneck-247          [-1, 128, 40, 40]               0
          Conv2d-248          [-1, 128, 40, 40]          32,768
     BatchNorm2d-249          [-1, 128, 40, 40]             256
            SiLU-250          [-1, 128, 40, 40]               0
            Conv-251          [-1, 128, 40, 40]               0
          Conv2d-252          [-1, 256, 40, 40]          65,536
     BatchNorm2d-253          [-1, 256, 40, 40]             512
            SiLU-254          [-1, 256, 40, 40]               0
            Conv-255          [-1, 256, 40, 40]               0
              C3-256          [-1, 256, 40, 40]               0
          Conv2d-257          [-1, 256, 20, 20]         589,824
     BatchNorm2d-258          [-1, 256, 20, 20]             512
            SiLU-259          [-1, 256, 20, 20]               0
            Conv-260          [-1, 256, 20, 20]               0
          Conv2d-261          [-1, 256, 20, 20]         131,072
     BatchNorm2d-262          [-1, 256, 20, 20]             512
            SiLU-263          [-1, 256, 20, 20]               0
            Conv-264          [-1, 256, 20, 20]               0
          Conv2d-265          [-1, 256, 20, 20]          65,536
     BatchNorm2d-266          [-1, 256, 20, 20]             512
            SiLU-267          [-1, 256, 20, 20]               0
            Conv-268          [-1, 256, 20, 20]               0
          Conv2d-269          [-1, 256, 20, 20]         589,824
     BatchNorm2d-270          [-1, 256, 20, 20]             512
            SiLU-271          [-1, 256, 20, 20]               0
            Conv-272          [-1, 256, 20, 20]               0
      Bottleneck-273          [-1, 256, 20, 20]               0
          Conv2d-274          [-1, 256, 20, 20]         131,072
     BatchNorm2d-275          [-1, 256, 20, 20]             512
            SiLU-276          [-1, 256, 20, 20]               0
            Conv-277          [-1, 256, 20, 20]               0
          Conv2d-278          [-1, 512, 20, 20]         262,144
     BatchNorm2d-279          [-1, 512, 20, 20]           1,024
            SiLU-280          [-1, 512, 20, 20]               0
            Conv-281          [-1, 512, 20, 20]               0
              C3-282          [-1, 512, 20, 20]               0
          Conv2d-283           [-1, 75, 80, 80]           9,675
          Conv2d-284           [-1, 75, 40, 40]          19,275
          Conv2d-285           [-1, 75, 20, 20]          38,475
================================================================
Total params: 7,157,793
Trainable params: 7,157,793
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.69
Forward/backward pass size (MB): 866.55
Params size (MB): 27.30
Estimated Total Size (MB): 898.54
----------------------------------------------------------------
Total GFLOPS: 16.643G
Total params: 7.158M

```





### The summary of FReLU_CBAM

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 320, 320]           3,456
       BatchNorm2d-2         [-1, 32, 320, 320]              64
            Conv2d-3         [-1, 32, 320, 320]             288
       BatchNorm2d-4         [-1, 32, 320, 320]              64
             FReLU-5         [-1, 32, 320, 320]               0
              Conv-6         [-1, 32, 320, 320]               0
             Focus-7         [-1, 32, 320, 320]               0
            Conv2d-8         [-1, 64, 160, 160]          18,432
       BatchNorm2d-9         [-1, 64, 160, 160]             128
           Conv2d-10         [-1, 64, 160, 160]             576
      BatchNorm2d-11         [-1, 64, 160, 160]             128
            FReLU-12         [-1, 64, 160, 160]               0
             Conv-13         [-1, 64, 160, 160]               0
           Conv2d-14         [-1, 32, 160, 160]           2,048
      BatchNorm2d-15         [-1, 32, 160, 160]              64
           Conv2d-16         [-1, 32, 160, 160]             288
      BatchNorm2d-17         [-1, 32, 160, 160]              64
            FReLU-18         [-1, 32, 160, 160]               0
             Conv-19         [-1, 32, 160, 160]               0
           Conv2d-20         [-1, 32, 160, 160]           1,024
      BatchNorm2d-21         [-1, 32, 160, 160]              64
           Conv2d-22         [-1, 32, 160, 160]             288
      BatchNorm2d-23         [-1, 32, 160, 160]              64
            FReLU-24         [-1, 32, 160, 160]               0
             Conv-25         [-1, 32, 160, 160]               0
           Conv2d-26         [-1, 32, 160, 160]           9,216
      BatchNorm2d-27         [-1, 32, 160, 160]              64
           Conv2d-28         [-1, 32, 160, 160]             288
      BatchNorm2d-29         [-1, 32, 160, 160]              64
            FReLU-30         [-1, 32, 160, 160]               0
             Conv-31         [-1, 32, 160, 160]               0
       Bottleneck-32         [-1, 32, 160, 160]               0
           Conv2d-33         [-1, 32, 160, 160]           2,048
      BatchNorm2d-34         [-1, 32, 160, 160]              64
           Conv2d-35         [-1, 32, 160, 160]             288
      BatchNorm2d-36         [-1, 32, 160, 160]              64
            FReLU-37         [-1, 32, 160, 160]               0
             Conv-38         [-1, 32, 160, 160]               0
           Conv2d-39         [-1, 64, 160, 160]           4,096
      BatchNorm2d-40         [-1, 64, 160, 160]             128
           Conv2d-41         [-1, 64, 160, 160]             576
      BatchNorm2d-42         [-1, 64, 160, 160]             128
            FReLU-43         [-1, 64, 160, 160]               0
             Conv-44         [-1, 64, 160, 160]               0
               C3-45         [-1, 64, 160, 160]               0
           Conv2d-46          [-1, 128, 80, 80]          73,728
      BatchNorm2d-47          [-1, 128, 80, 80]             256
           Conv2d-48          [-1, 128, 80, 80]           1,152
      BatchNorm2d-49          [-1, 128, 80, 80]             256
            FReLU-50          [-1, 128, 80, 80]               0
             Conv-51          [-1, 128, 80, 80]               0
           Conv2d-52           [-1, 64, 80, 80]           8,192
      BatchNorm2d-53           [-1, 64, 80, 80]             128
           Conv2d-54           [-1, 64, 80, 80]             576
      BatchNorm2d-55           [-1, 64, 80, 80]             128
            FReLU-56           [-1, 64, 80, 80]               0
             Conv-57           [-1, 64, 80, 80]               0
           Conv2d-58           [-1, 64, 80, 80]           4,096
      BatchNorm2d-59           [-1, 64, 80, 80]             128
           Conv2d-60           [-1, 64, 80, 80]             576
      BatchNorm2d-61           [-1, 64, 80, 80]             128
            FReLU-62           [-1, 64, 80, 80]               0
             Conv-63           [-1, 64, 80, 80]               0
           Conv2d-64           [-1, 64, 80, 80]          36,864
      BatchNorm2d-65           [-1, 64, 80, 80]             128
           Conv2d-66           [-1, 64, 80, 80]             576
      BatchNorm2d-67           [-1, 64, 80, 80]             128
            FReLU-68           [-1, 64, 80, 80]               0
             Conv-69           [-1, 64, 80, 80]               0
       Bottleneck-70           [-1, 64, 80, 80]               0
           Conv2d-71           [-1, 64, 80, 80]           4,096
      BatchNorm2d-72           [-1, 64, 80, 80]             128
           Conv2d-73           [-1, 64, 80, 80]             576
      BatchNorm2d-74           [-1, 64, 80, 80]             128
            FReLU-75           [-1, 64, 80, 80]               0
             Conv-76           [-1, 64, 80, 80]               0
           Conv2d-77           [-1, 64, 80, 80]          36,864
      BatchNorm2d-78           [-1, 64, 80, 80]             128
           Conv2d-79           [-1, 64, 80, 80]             576
      BatchNorm2d-80           [-1, 64, 80, 80]             128
            FReLU-81           [-1, 64, 80, 80]               0
             Conv-82           [-1, 64, 80, 80]               0
       Bottleneck-83           [-1, 64, 80, 80]               0
           Conv2d-84           [-1, 64, 80, 80]           4,096
      BatchNorm2d-85           [-1, 64, 80, 80]             128
           Conv2d-86           [-1, 64, 80, 80]             576
      BatchNorm2d-87           [-1, 64, 80, 80]             128
            FReLU-88           [-1, 64, 80, 80]               0
             Conv-89           [-1, 64, 80, 80]               0
           Conv2d-90           [-1, 64, 80, 80]          36,864
      BatchNorm2d-91           [-1, 64, 80, 80]             128
           Conv2d-92           [-1, 64, 80, 80]             576
      BatchNorm2d-93           [-1, 64, 80, 80]             128
            FReLU-94           [-1, 64, 80, 80]               0
             Conv-95           [-1, 64, 80, 80]               0
       Bottleneck-96           [-1, 64, 80, 80]               0
           Conv2d-97           [-1, 64, 80, 80]           8,192
      BatchNorm2d-98           [-1, 64, 80, 80]             128
           Conv2d-99           [-1, 64, 80, 80]             576
     BatchNorm2d-100           [-1, 64, 80, 80]             128
           FReLU-101           [-1, 64, 80, 80]               0
            Conv-102           [-1, 64, 80, 80]               0
          Conv2d-103          [-1, 128, 80, 80]          16,384
     BatchNorm2d-104          [-1, 128, 80, 80]             256
          Conv2d-105          [-1, 128, 80, 80]           1,152
     BatchNorm2d-106          [-1, 128, 80, 80]             256
           FReLU-107          [-1, 128, 80, 80]               0
            Conv-108          [-1, 128, 80, 80]               0
              C3-109          [-1, 128, 80, 80]               0
          Conv2d-110          [-1, 256, 40, 40]         294,912
     BatchNorm2d-111          [-1, 256, 40, 40]             512
          Conv2d-112          [-1, 256, 40, 40]           2,304
     BatchNorm2d-113          [-1, 256, 40, 40]             512
           FReLU-114          [-1, 256, 40, 40]               0
            Conv-115          [-1, 256, 40, 40]               0
          Conv2d-116          [-1, 128, 40, 40]          32,768
     BatchNorm2d-117          [-1, 128, 40, 40]             256
          Conv2d-118          [-1, 128, 40, 40]           1,152
     BatchNorm2d-119          [-1, 128, 40, 40]             256
           FReLU-120          [-1, 128, 40, 40]               0
            Conv-121          [-1, 128, 40, 40]               0
          Conv2d-122          [-1, 128, 40, 40]          16,384
     BatchNorm2d-123          [-1, 128, 40, 40]             256
          Conv2d-124          [-1, 128, 40, 40]           1,152
     BatchNorm2d-125          [-1, 128, 40, 40]             256
           FReLU-126          [-1, 128, 40, 40]               0
            Conv-127          [-1, 128, 40, 40]               0
          Conv2d-128          [-1, 128, 40, 40]         147,456
     BatchNorm2d-129          [-1, 128, 40, 40]             256
          Conv2d-130          [-1, 128, 40, 40]           1,152
     BatchNorm2d-131          [-1, 128, 40, 40]             256
           FReLU-132          [-1, 128, 40, 40]               0
            Conv-133          [-1, 128, 40, 40]               0
      Bottleneck-134          [-1, 128, 40, 40]               0
          Conv2d-135          [-1, 128, 40, 40]          16,384
     BatchNorm2d-136          [-1, 128, 40, 40]             256
          Conv2d-137          [-1, 128, 40, 40]           1,152
     BatchNorm2d-138          [-1, 128, 40, 40]             256
           FReLU-139          [-1, 128, 40, 40]               0
            Conv-140          [-1, 128, 40, 40]               0
          Conv2d-141          [-1, 128, 40, 40]         147,456
     BatchNorm2d-142          [-1, 128, 40, 40]             256
          Conv2d-143          [-1, 128, 40, 40]           1,152
     BatchNorm2d-144          [-1, 128, 40, 40]             256
           FReLU-145          [-1, 128, 40, 40]               0
            Conv-146          [-1, 128, 40, 40]               0
      Bottleneck-147          [-1, 128, 40, 40]               0
          Conv2d-148          [-1, 128, 40, 40]          16,384
     BatchNorm2d-149          [-1, 128, 40, 40]             256
          Conv2d-150          [-1, 128, 40, 40]           1,152
     BatchNorm2d-151          [-1, 128, 40, 40]             256
           FReLU-152          [-1, 128, 40, 40]               0
            Conv-153          [-1, 128, 40, 40]               0
          Conv2d-154          [-1, 128, 40, 40]         147,456
     BatchNorm2d-155          [-1, 128, 40, 40]             256
          Conv2d-156          [-1, 128, 40, 40]           1,152
     BatchNorm2d-157          [-1, 128, 40, 40]             256
           FReLU-158          [-1, 128, 40, 40]               0
            Conv-159          [-1, 128, 40, 40]               0
      Bottleneck-160          [-1, 128, 40, 40]               0
          Conv2d-161          [-1, 128, 40, 40]          32,768
     BatchNorm2d-162          [-1, 128, 40, 40]             256
          Conv2d-163          [-1, 128, 40, 40]           1,152
     BatchNorm2d-164          [-1, 128, 40, 40]             256
           FReLU-165          [-1, 128, 40, 40]               0
            Conv-166          [-1, 128, 40, 40]               0
          Conv2d-167          [-1, 256, 40, 40]          65,536
     BatchNorm2d-168          [-1, 256, 40, 40]             512
          Conv2d-169          [-1, 256, 40, 40]           2,304
     BatchNorm2d-170          [-1, 256, 40, 40]             512
           FReLU-171          [-1, 256, 40, 40]               0
            Conv-172          [-1, 256, 40, 40]               0
              C3-173          [-1, 256, 40, 40]               0
          Conv2d-174          [-1, 512, 20, 20]       1,179,648
     BatchNorm2d-175          [-1, 512, 20, 20]           1,024
          Conv2d-176          [-1, 512, 20, 20]           4,608
     BatchNorm2d-177          [-1, 512, 20, 20]           1,024
           FReLU-178          [-1, 512, 20, 20]               0
            Conv-179          [-1, 512, 20, 20]               0
          Conv2d-180          [-1, 256, 20, 20]         131,072
     BatchNorm2d-181          [-1, 256, 20, 20]             512
          Conv2d-182          [-1, 256, 20, 20]           2,304
     BatchNorm2d-183          [-1, 256, 20, 20]             512
           FReLU-184          [-1, 256, 20, 20]               0
            Conv-185          [-1, 256, 20, 20]               0
       MaxPool2d-186          [-1, 256, 20, 20]               0
       MaxPool2d-187          [-1, 256, 20, 20]               0
       MaxPool2d-188          [-1, 256, 20, 20]               0
          Conv2d-189          [-1, 512, 20, 20]         524,288
     BatchNorm2d-190          [-1, 512, 20, 20]           1,024
          Conv2d-191          [-1, 512, 20, 20]           4,608
     BatchNorm2d-192          [-1, 512, 20, 20]           1,024
           FReLU-193          [-1, 512, 20, 20]               0
            Conv-194          [-1, 512, 20, 20]               0
             SPP-195          [-1, 512, 20, 20]               0
          Conv2d-196          [-1, 256, 20, 20]         131,072
     BatchNorm2d-197          [-1, 256, 20, 20]             512
          Conv2d-198          [-1, 256, 20, 20]           2,304
     BatchNorm2d-199          [-1, 256, 20, 20]             512
           FReLU-200          [-1, 256, 20, 20]               0
            Conv-201          [-1, 256, 20, 20]               0
          Conv2d-202          [-1, 256, 20, 20]          65,536
     BatchNorm2d-203          [-1, 256, 20, 20]             512
          Conv2d-204          [-1, 256, 20, 20]           2,304
     BatchNorm2d-205          [-1, 256, 20, 20]             512
           FReLU-206          [-1, 256, 20, 20]               0
            Conv-207          [-1, 256, 20, 20]               0
          Conv2d-208          [-1, 256, 20, 20]         589,824
     BatchNorm2d-209          [-1, 256, 20, 20]             512
          Conv2d-210          [-1, 256, 20, 20]           2,304
     BatchNorm2d-211          [-1, 256, 20, 20]             512
           FReLU-212          [-1, 256, 20, 20]               0
            Conv-213          [-1, 256, 20, 20]               0
      Bottleneck-214          [-1, 256, 20, 20]               0
          Conv2d-215          [-1, 256, 20, 20]         131,072
     BatchNorm2d-216          [-1, 256, 20, 20]             512
          Conv2d-217          [-1, 256, 20, 20]           2,304
     BatchNorm2d-218          [-1, 256, 20, 20]             512
           FReLU-219          [-1, 256, 20, 20]               0
            Conv-220          [-1, 256, 20, 20]               0
          Conv2d-221          [-1, 512, 20, 20]         262,144
     BatchNorm2d-222          [-1, 512, 20, 20]           1,024
          Conv2d-223          [-1, 512, 20, 20]           4,608
     BatchNorm2d-224          [-1, 512, 20, 20]           1,024
           FReLU-225          [-1, 512, 20, 20]               0
            Conv-226          [-1, 512, 20, 20]               0
              C3-227          [-1, 512, 20, 20]               0
      CSPDarknet-228  [[-1, 128, 80, 80], [-1, 256, 40, 40], [-1, 512, 20, 20]]               0
AdaptiveAvgPool2d-229            [-1, 128, 1, 1]               0
          Linear-230                    [-1, 8]           1,024
            ReLU-231                    [-1, 8]               0
          Linear-232                  [-1, 128]           1,024
         Sigmoid-233                  [-1, 128]               0
        se_block-234          [-1, 128, 80, 80]               0
AdaptiveAvgPool2d-235            [-1, 256, 1, 1]               0
          Linear-236                   [-1, 16]           4,096
            ReLU-237                   [-1, 16]               0
          Linear-238                  [-1, 256]           4,096
         Sigmoid-239                  [-1, 256]               0
        se_block-240          [-1, 256, 40, 40]               0
AdaptiveAvgPool2d-241            [-1, 512, 1, 1]               0
          Linear-242                   [-1, 32]          16,384
            ReLU-243                   [-1, 32]               0
          Linear-244                  [-1, 512]          16,384
         Sigmoid-245                  [-1, 512]               0
        se_block-246          [-1, 512, 20, 20]               0
          Conv2d-247          [-1, 256, 20, 20]         131,072
     BatchNorm2d-248          [-1, 256, 20, 20]             512
          Conv2d-249          [-1, 256, 20, 20]           2,304
     BatchNorm2d-250          [-1, 256, 20, 20]             512
           FReLU-251          [-1, 256, 20, 20]               0
            Conv-252          [-1, 256, 20, 20]               0
        Upsample-253          [-1, 256, 40, 40]               0
          Conv2d-254          [-1, 128, 40, 40]          65,536
     BatchNorm2d-255          [-1, 128, 40, 40]             256
          Conv2d-256          [-1, 128, 40, 40]           1,152
     BatchNorm2d-257          [-1, 128, 40, 40]             256
           FReLU-258          [-1, 128, 40, 40]               0
            Conv-259          [-1, 128, 40, 40]               0
          Conv2d-260          [-1, 128, 40, 40]          16,384
     BatchNorm2d-261          [-1, 128, 40, 40]             256
          Conv2d-262          [-1, 128, 40, 40]           1,152
     BatchNorm2d-263          [-1, 128, 40, 40]             256
           FReLU-264          [-1, 128, 40, 40]               0
            Conv-265          [-1, 128, 40, 40]               0
          Conv2d-266          [-1, 128, 40, 40]         147,456
     BatchNorm2d-267          [-1, 128, 40, 40]             256
          Conv2d-268          [-1, 128, 40, 40]           1,152
     BatchNorm2d-269          [-1, 128, 40, 40]             256
           FReLU-270          [-1, 128, 40, 40]               0
            Conv-271          [-1, 128, 40, 40]               0
      Bottleneck-272          [-1, 128, 40, 40]               0
          Conv2d-273          [-1, 128, 40, 40]          65,536
     BatchNorm2d-274          [-1, 128, 40, 40]             256
          Conv2d-275          [-1, 128, 40, 40]           1,152
     BatchNorm2d-276          [-1, 128, 40, 40]             256
           FReLU-277          [-1, 128, 40, 40]               0
            Conv-278          [-1, 128, 40, 40]               0
          Conv2d-279          [-1, 256, 40, 40]          65,536
     BatchNorm2d-280          [-1, 256, 40, 40]             512
          Conv2d-281          [-1, 256, 40, 40]           2,304
     BatchNorm2d-282          [-1, 256, 40, 40]             512
           FReLU-283          [-1, 256, 40, 40]               0
            Conv-284          [-1, 256, 40, 40]               0
              C3-285          [-1, 256, 40, 40]               0
          Conv2d-286          [-1, 128, 40, 40]          32,768
     BatchNorm2d-287          [-1, 128, 40, 40]             256
          Conv2d-288          [-1, 128, 40, 40]           1,152
     BatchNorm2d-289          [-1, 128, 40, 40]             256
           FReLU-290          [-1, 128, 40, 40]               0
            Conv-291          [-1, 128, 40, 40]               0
        Upsample-292          [-1, 128, 80, 80]               0
          Conv2d-293           [-1, 64, 80, 80]          16,384
     BatchNorm2d-294           [-1, 64, 80, 80]             128
          Conv2d-295           [-1, 64, 80, 80]             576
     BatchNorm2d-296           [-1, 64, 80, 80]             128
           FReLU-297           [-1, 64, 80, 80]               0
            Conv-298           [-1, 64, 80, 80]               0
          Conv2d-299           [-1, 64, 80, 80]           4,096
     BatchNorm2d-300           [-1, 64, 80, 80]             128
          Conv2d-301           [-1, 64, 80, 80]             576
     BatchNorm2d-302           [-1, 64, 80, 80]             128
           FReLU-303           [-1, 64, 80, 80]               0
            Conv-304           [-1, 64, 80, 80]               0
          Conv2d-305           [-1, 64, 80, 80]          36,864
     BatchNorm2d-306           [-1, 64, 80, 80]             128
          Conv2d-307           [-1, 64, 80, 80]             576
     BatchNorm2d-308           [-1, 64, 80, 80]             128
           FReLU-309           [-1, 64, 80, 80]               0
            Conv-310           [-1, 64, 80, 80]               0
      Bottleneck-311           [-1, 64, 80, 80]               0
          Conv2d-312           [-1, 64, 80, 80]          16,384
     BatchNorm2d-313           [-1, 64, 80, 80]             128
          Conv2d-314           [-1, 64, 80, 80]             576
     BatchNorm2d-315           [-1, 64, 80, 80]             128
           FReLU-316           [-1, 64, 80, 80]               0
            Conv-317           [-1, 64, 80, 80]               0
          Conv2d-318          [-1, 128, 80, 80]          16,384
     BatchNorm2d-319          [-1, 128, 80, 80]             256
          Conv2d-320          [-1, 128, 80, 80]           1,152
     BatchNorm2d-321          [-1, 128, 80, 80]             256
           FReLU-322          [-1, 128, 80, 80]               0
            Conv-323          [-1, 128, 80, 80]               0
              C3-324          [-1, 128, 80, 80]               0
          Conv2d-325          [-1, 128, 40, 40]         147,456
     BatchNorm2d-326          [-1, 128, 40, 40]             256
          Conv2d-327          [-1, 128, 40, 40]           1,152
     BatchNorm2d-328          [-1, 128, 40, 40]             256
           FReLU-329          [-1, 128, 40, 40]               0
            Conv-330          [-1, 128, 40, 40]               0
          Conv2d-331          [-1, 128, 40, 40]          32,768
     BatchNorm2d-332          [-1, 128, 40, 40]             256
          Conv2d-333          [-1, 128, 40, 40]           1,152
     BatchNorm2d-334          [-1, 128, 40, 40]             256
           FReLU-335          [-1, 128, 40, 40]               0
            Conv-336          [-1, 128, 40, 40]               0
          Conv2d-337          [-1, 128, 40, 40]          16,384
     BatchNorm2d-338          [-1, 128, 40, 40]             256
          Conv2d-339          [-1, 128, 40, 40]           1,152
     BatchNorm2d-340          [-1, 128, 40, 40]             256
           FReLU-341          [-1, 128, 40, 40]               0
            Conv-342          [-1, 128, 40, 40]               0
          Conv2d-343          [-1, 128, 40, 40]         147,456
     BatchNorm2d-344          [-1, 128, 40, 40]             256
          Conv2d-345          [-1, 128, 40, 40]           1,152
     BatchNorm2d-346          [-1, 128, 40, 40]             256
           FReLU-347          [-1, 128, 40, 40]               0
            Conv-348          [-1, 128, 40, 40]               0
      Bottleneck-349          [-1, 128, 40, 40]               0
          Conv2d-350          [-1, 128, 40, 40]          32,768
     BatchNorm2d-351          [-1, 128, 40, 40]             256
          Conv2d-352          [-1, 128, 40, 40]           1,152
     BatchNorm2d-353          [-1, 128, 40, 40]             256
           FReLU-354          [-1, 128, 40, 40]               0
            Conv-355          [-1, 128, 40, 40]               0
          Conv2d-356          [-1, 256, 40, 40]          65,536
     BatchNorm2d-357          [-1, 256, 40, 40]             512
          Conv2d-358          [-1, 256, 40, 40]           2,304
     BatchNorm2d-359          [-1, 256, 40, 40]             512
           FReLU-360          [-1, 256, 40, 40]               0
            Conv-361          [-1, 256, 40, 40]               0
              C3-362          [-1, 256, 40, 40]               0
          Conv2d-363          [-1, 256, 20, 20]         589,824
     BatchNorm2d-364          [-1, 256, 20, 20]             512
          Conv2d-365          [-1, 256, 20, 20]           2,304
     BatchNorm2d-366          [-1, 256, 20, 20]             512
           FReLU-367          [-1, 256, 20, 20]               0
            Conv-368          [-1, 256, 20, 20]               0
          Conv2d-369          [-1, 256, 20, 20]         131,072
     BatchNorm2d-370          [-1, 256, 20, 20]             512
          Conv2d-371          [-1, 256, 20, 20]           2,304
     BatchNorm2d-372          [-1, 256, 20, 20]             512
           FReLU-373          [-1, 256, 20, 20]               0
            Conv-374          [-1, 256, 20, 20]               0
          Conv2d-375          [-1, 256, 20, 20]          65,536
     BatchNorm2d-376          [-1, 256, 20, 20]             512
          Conv2d-377          [-1, 256, 20, 20]           2,304
     BatchNorm2d-378          [-1, 256, 20, 20]             512
           FReLU-379          [-1, 256, 20, 20]               0
            Conv-380          [-1, 256, 20, 20]               0
          Conv2d-381          [-1, 256, 20, 20]         589,824
     BatchNorm2d-382          [-1, 256, 20, 20]             512
          Conv2d-383          [-1, 256, 20, 20]           2,304
     BatchNorm2d-384          [-1, 256, 20, 20]             512
           FReLU-385          [-1, 256, 20, 20]               0
            Conv-386          [-1, 256, 20, 20]               0
      Bottleneck-387          [-1, 256, 20, 20]               0
          Conv2d-388          [-1, 256, 20, 20]         131,072
     BatchNorm2d-389          [-1, 256, 20, 20]             512
          Conv2d-390          [-1, 256, 20, 20]           2,304
     BatchNorm2d-391          [-1, 256, 20, 20]             512
           FReLU-392          [-1, 256, 20, 20]               0
            Conv-393          [-1, 256, 20, 20]               0
          Conv2d-394          [-1, 512, 20, 20]         262,144
     BatchNorm2d-395          [-1, 512, 20, 20]           1,024
          Conv2d-396          [-1, 512, 20, 20]           4,608
     BatchNorm2d-397          [-1, 512, 20, 20]           1,024
           FReLU-398          [-1, 512, 20, 20]               0
            Conv-399          [-1, 512, 20, 20]               0
              C3-400          [-1, 512, 20, 20]               0
          Conv2d-401           [-1, 75, 80, 80]           9,675
          Conv2d-402           [-1, 75, 40, 40]          19,275
          Conv2d-403           [-1, 75, 20, 20]          38,475
================================================================
Total params: 7,263,745
Trainable params: 7,263,745
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.69
Forward/backward pass size (MB): 1239.98
Params size (MB): 27.71
Estimated Total Size (MB): 1272.38
----------------------------------------------------------------
Total GFLOPS: 17.279G
Total params: 7.264M

```



### 目前的网络结构

```python
YoloBody(
  (backbone): CSPDarknet(
    (stem): Focus(
      (conv): Conv(
        (conv): Conv2d(12, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (dark2): Sequential(
      (0): Conv(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): C3(
        (cv1): Conv(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
      (0): Conv(
        (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): C3(
        (cv1): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (1): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (2): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
      (0): Conv(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): C3(
        (cv1): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (1): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (2): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
      (0): Conv(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): FReLU(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): SPP(
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv3): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
            (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (m): Sequential(
          (0): Bottleneck(
            (cv1): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): FReLU(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (cv2): Conv(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
  (feat1_att): cbam_block(
    (channelattention): ChannelAttention(
      (avg_pool): AdaptiveAvgPool2d(output_size=1)
      (max_pool): AdaptiveMaxPool2d(output_size=1)
      (fc1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (relu1): ReLU()
      (fc2): Conv2d(16, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (sigmoid): Sigmoid()
    )
    (spatialattention): SpatialAttention(
      (conv1): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
      (sigmoid): Sigmoid()
    )
  )
  (feat2_att): cbam_block(
    (channelattention): ChannelAttention(
      (avg_pool): AdaptiveAvgPool2d(output_size=1)
      (max_pool): AdaptiveMaxPool2d(output_size=1)
      (fc1): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (relu1): ReLU()
      (fc2): Conv2d(32, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (sigmoid): Sigmoid()
    )
    (spatialattention): SpatialAttention(
      (conv1): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
      (sigmoid): Sigmoid()
    )
  )
  (feat3_att): cbam_block(
    (channelattention): ChannelAttention(
      (avg_pool): AdaptiveAvgPool2d(output_size=1)
      (max_pool): AdaptiveMaxPool2d(output_size=1)
      (fc1): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (relu1): ReLU()
      (fc2): Conv2d(64, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (sigmoid): Sigmoid()
    )
    (spatialattention): SpatialAttention(
      (conv1): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
      (sigmoid): Sigmoid()
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
    (cv1): Conv(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): Conv(
      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
    (cv1): Conv(
      (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): Conv(
      (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
    (cv1): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
    (cv1): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv2): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (cv3): Conv(
      (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): FReLU(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (m): Sequential(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): FReLU(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (cv2): Conv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
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
```



### 更换过的网络结构

```python
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
           Conv2d-14         [-1, 32, 320, 320]           9,216
      BatchNorm2d-15         [-1, 32, 320, 320]              64
           Conv2d-16         [-1, 32, 320, 320]             288
      BatchNorm2d-17         [-1, 32, 320, 320]              64
            FReLU-18         [-1, 32, 320, 320]               0
             Conv-19         [-1, 32, 320, 320]               0
           Conv2d-20         [-1, 32, 320, 320]             288
           Conv2d-21         [-1, 32, 320, 320]           1,056
           Conv2d-22         [-1, 32, 320, 320]             288
      BatchNorm2d-23         [-1, 32, 320, 320]              64
            FReLU-24         [-1, 32, 320, 320]               0
       depth_conv-25         [-1, 32, 320, 320]               0
           Conv2d-26         [-1, 32, 160, 160]             288
           Conv2d-27         [-1, 64, 160, 160]           2,112
           Conv2d-28         [-1, 64, 160, 160]             576
      BatchNorm2d-29         [-1, 64, 160, 160]             128
            FReLU-30         [-1, 64, 160, 160]               0
       depth_conv-31         [-1, 64, 160, 160]               0
           Conv2d-32         [-1, 64, 160, 160]              64
           Conv2d-33         [-1, 32, 160, 160]           2,080
           Conv2d-34         [-1, 32, 160, 160]             288
      BatchNorm2d-35         [-1, 32, 160, 160]              64
            FReLU-36         [-1, 32, 160, 160]               0
       depth_conv-37         [-1, 32, 160, 160]               0
           Conv2d-38         [-1, 32, 160, 160]              32
           Conv2d-39         [-1, 32, 160, 160]           1,056
           Conv2d-40         [-1, 32, 160, 160]             288
      BatchNorm2d-41         [-1, 32, 160, 160]              64
            FReLU-42         [-1, 32, 160, 160]               0
       depth_conv-43         [-1, 32, 160, 160]               0
           Conv2d-44         [-1, 32, 160, 160]             288
           Conv2d-45         [-1, 32, 160, 160]           1,056
           Conv2d-46         [-1, 32, 160, 160]             288
      BatchNorm2d-47         [-1, 32, 160, 160]              64
            FReLU-48         [-1, 32, 160, 160]               0
       depth_conv-49         [-1, 32, 160, 160]               0
       Bottleneck-50         [-1, 32, 160, 160]               0
           Conv2d-51         [-1, 64, 160, 160]              64
           Conv2d-52         [-1, 32, 160, 160]           2,080
           Conv2d-53         [-1, 32, 160, 160]             288
      BatchNorm2d-54         [-1, 32, 160, 160]              64
            FReLU-55         [-1, 32, 160, 160]               0
       depth_conv-56         [-1, 32, 160, 160]               0
           Conv2d-57         [-1, 64, 160, 160]           2,048
      BatchNorm2d-58         [-1, 64, 160, 160]             128
           Conv2d-59         [-1, 64, 160, 160]             576
      BatchNorm2d-60         [-1, 64, 160, 160]             128
            FReLU-61         [-1, 64, 160, 160]               0
             Conv-62         [-1, 64, 160, 160]               0
               C3-63         [-1, 64, 160, 160]               0
           Conv2d-64           [-1, 64, 80, 80]             576
           Conv2d-65          [-1, 128, 80, 80]           8,320
           Conv2d-66          [-1, 128, 80, 80]           1,152
      BatchNorm2d-67          [-1, 128, 80, 80]             256
            FReLU-68          [-1, 128, 80, 80]               0
       depth_conv-69          [-1, 128, 80, 80]               0
           Conv2d-70          [-1, 128, 80, 80]             128
           Conv2d-71           [-1, 64, 80, 80]           8,256
           Conv2d-72           [-1, 64, 80, 80]             576
      BatchNorm2d-73           [-1, 64, 80, 80]             128
            FReLU-74           [-1, 64, 80, 80]               0
       depth_conv-75           [-1, 64, 80, 80]               0
           Conv2d-76           [-1, 64, 80, 80]              64
           Conv2d-77           [-1, 64, 80, 80]           4,160
           Conv2d-78           [-1, 64, 80, 80]             576
      BatchNorm2d-79           [-1, 64, 80, 80]             128
            FReLU-80           [-1, 64, 80, 80]               0
       depth_conv-81           [-1, 64, 80, 80]               0
           Conv2d-82           [-1, 64, 80, 80]             576
           Conv2d-83           [-1, 64, 80, 80]           4,160
           Conv2d-84           [-1, 64, 80, 80]             576
      BatchNorm2d-85           [-1, 64, 80, 80]             128
            FReLU-86           [-1, 64, 80, 80]               0
       depth_conv-87           [-1, 64, 80, 80]               0
       Bottleneck-88           [-1, 64, 80, 80]               0
           Conv2d-89           [-1, 64, 80, 80]              64
           Conv2d-90           [-1, 64, 80, 80]           4,160
           Conv2d-91           [-1, 64, 80, 80]             576
      BatchNorm2d-92           [-1, 64, 80, 80]             128
            FReLU-93           [-1, 64, 80, 80]               0
       depth_conv-94           [-1, 64, 80, 80]               0
           Conv2d-95           [-1, 64, 80, 80]             576
           Conv2d-96           [-1, 64, 80, 80]           4,160
           Conv2d-97           [-1, 64, 80, 80]             576
      BatchNorm2d-98           [-1, 64, 80, 80]             128
            FReLU-99           [-1, 64, 80, 80]               0
      depth_conv-100           [-1, 64, 80, 80]               0
      Bottleneck-101           [-1, 64, 80, 80]               0
          Conv2d-102          [-1, 128, 80, 80]             128
          Conv2d-103           [-1, 64, 80, 80]           8,256
          Conv2d-104           [-1, 64, 80, 80]             576
     BatchNorm2d-105           [-1, 64, 80, 80]             128
           FReLU-106           [-1, 64, 80, 80]               0
      depth_conv-107           [-1, 64, 80, 80]               0
          Conv2d-108          [-1, 128, 80, 80]           8,192
     BatchNorm2d-109          [-1, 128, 80, 80]             256
          Conv2d-110          [-1, 128, 80, 80]           1,152
     BatchNorm2d-111          [-1, 128, 80, 80]             256
           FReLU-112          [-1, 128, 80, 80]               0
            Conv-113          [-1, 128, 80, 80]               0
              C3-114          [-1, 128, 80, 80]               0
          Conv2d-115          [-1, 128, 40, 40]           1,152
          Conv2d-116          [-1, 256, 40, 40]          33,024
          Conv2d-117          [-1, 256, 40, 40]           2,304
     BatchNorm2d-118          [-1, 256, 40, 40]             512
           FReLU-119          [-1, 256, 40, 40]               0
      depth_conv-120          [-1, 256, 40, 40]               0
          Conv2d-121          [-1, 256, 40, 40]             256
          Conv2d-122          [-1, 128, 40, 40]          32,896
          Conv2d-123          [-1, 128, 40, 40]           1,152
     BatchNorm2d-124          [-1, 128, 40, 40]             256
           FReLU-125          [-1, 128, 40, 40]               0
      depth_conv-126          [-1, 128, 40, 40]               0
          Conv2d-127          [-1, 128, 40, 40]             128
          Conv2d-128          [-1, 128, 40, 40]          16,512
          Conv2d-129          [-1, 128, 40, 40]           1,152
     BatchNorm2d-130          [-1, 128, 40, 40]             256
           FReLU-131          [-1, 128, 40, 40]               0
      depth_conv-132          [-1, 128, 40, 40]               0
          Conv2d-133          [-1, 128, 40, 40]           1,152
          Conv2d-134          [-1, 128, 40, 40]          16,512
          Conv2d-135          [-1, 128, 40, 40]           1,152
     BatchNorm2d-136          [-1, 128, 40, 40]             256
           FReLU-137          [-1, 128, 40, 40]               0
      depth_conv-138          [-1, 128, 40, 40]               0
      Bottleneck-139          [-1, 128, 40, 40]               0
          Conv2d-140          [-1, 128, 40, 40]             128
          Conv2d-141          [-1, 128, 40, 40]          16,512
          Conv2d-142          [-1, 128, 40, 40]           1,152
     BatchNorm2d-143          [-1, 128, 40, 40]             256
           FReLU-144          [-1, 128, 40, 40]               0
      depth_conv-145          [-1, 128, 40, 40]               0
          Conv2d-146          [-1, 128, 40, 40]           1,152
          Conv2d-147          [-1, 128, 40, 40]          16,512
          Conv2d-148          [-1, 128, 40, 40]           1,152
     BatchNorm2d-149          [-1, 128, 40, 40]             256
           FReLU-150          [-1, 128, 40, 40]               0
      depth_conv-151          [-1, 128, 40, 40]               0
      Bottleneck-152          [-1, 128, 40, 40]               0
          Conv2d-153          [-1, 256, 40, 40]             256
          Conv2d-154          [-1, 128, 40, 40]          32,896
          Conv2d-155          [-1, 128, 40, 40]           1,152
     BatchNorm2d-156          [-1, 128, 40, 40]             256
           FReLU-157          [-1, 128, 40, 40]               0
      depth_conv-158          [-1, 128, 40, 40]               0
          Conv2d-159          [-1, 256, 40, 40]          32,768
     BatchNorm2d-160          [-1, 256, 40, 40]             512
          Conv2d-161          [-1, 256, 40, 40]           2,304
     BatchNorm2d-162          [-1, 256, 40, 40]             512
           FReLU-163          [-1, 256, 40, 40]               0
            Conv-164          [-1, 256, 40, 40]               0
              C3-165          [-1, 256, 40, 40]               0
          Conv2d-166          [-1, 256, 20, 20]           2,304
          Conv2d-167          [-1, 512, 20, 20]         131,584
          Conv2d-168          [-1, 512, 20, 20]           4,608
     BatchNorm2d-169          [-1, 512, 20, 20]           1,024
           FReLU-170          [-1, 512, 20, 20]               0
      depth_conv-171          [-1, 512, 20, 20]               0
          Conv2d-172          [-1, 512, 20, 20]             512
          Conv2d-173          [-1, 256, 20, 20]         131,328
          Conv2d-174          [-1, 256, 20, 20]           2,304
     BatchNorm2d-175          [-1, 256, 20, 20]             512
           FReLU-176          [-1, 256, 20, 20]               0
      depth_conv-177          [-1, 256, 20, 20]               0
       MaxPool2d-178          [-1, 256, 20, 20]               0
       MaxPool2d-179          [-1, 256, 20, 20]               0
       MaxPool2d-180          [-1, 256, 20, 20]               0
          Conv2d-181          [-1, 512, 20, 20]         524,288
     BatchNorm2d-182          [-1, 512, 20, 20]           1,024
          Conv2d-183          [-1, 512, 20, 20]           4,608
     BatchNorm2d-184          [-1, 512, 20, 20]           1,024
           FReLU-185          [-1, 512, 20, 20]               0
            Conv-186          [-1, 512, 20, 20]               0
             SPP-187          [-1, 512, 20, 20]               0
          Conv2d-188          [-1, 512, 20, 20]             512
          Conv2d-189          [-1, 256, 20, 20]         131,328
          Conv2d-190          [-1, 256, 20, 20]           2,304
     BatchNorm2d-191          [-1, 256, 20, 20]             512
           FReLU-192          [-1, 256, 20, 20]               0
      depth_conv-193          [-1, 256, 20, 20]               0
          Conv2d-194          [-1, 256, 20, 20]             256
          Conv2d-195          [-1, 256, 20, 20]          65,792
          Conv2d-196          [-1, 256, 20, 20]           2,304
     BatchNorm2d-197          [-1, 256, 20, 20]             512
           FReLU-198          [-1, 256, 20, 20]               0
      depth_conv-199          [-1, 256, 20, 20]               0
          Conv2d-200          [-1, 256, 20, 20]           2,304
          Conv2d-201          [-1, 256, 20, 20]          65,792
          Conv2d-202          [-1, 256, 20, 20]           2,304
     BatchNorm2d-203          [-1, 256, 20, 20]             512
           FReLU-204          [-1, 256, 20, 20]               0
      depth_conv-205          [-1, 256, 20, 20]               0
      Bottleneck-206          [-1, 256, 20, 20]               0
          Conv2d-207          [-1, 512, 20, 20]             512
          Conv2d-208          [-1, 256, 20, 20]         131,328
          Conv2d-209          [-1, 256, 20, 20]           2,304
     BatchNorm2d-210          [-1, 256, 20, 20]             512
           FReLU-211          [-1, 256, 20, 20]               0
      depth_conv-212          [-1, 256, 20, 20]               0
          Conv2d-213          [-1, 512, 20, 20]         131,072
     BatchNorm2d-214          [-1, 512, 20, 20]           1,024
          Conv2d-215          [-1, 512, 20, 20]           4,608
     BatchNorm2d-216          [-1, 512, 20, 20]           1,024
           FReLU-217          [-1, 512, 20, 20]               0
            Conv-218          [-1, 512, 20, 20]               0
              C3-219          [-1, 512, 20, 20]               0
      CSPDarknet-220  [[-1, 128, 80, 80], [-1, 256, 40, 40], [-1, 512, 20, 20]]               0
AdaptiveAvgPool2d-221            [-1, 128, 1, 1]               0
          Linear-222                    [-1, 8]           1,024
            ReLU-223                    [-1, 8]               0
          Linear-224                  [-1, 128]           1,024
         Sigmoid-225                  [-1, 128]               0
        se_block-226          [-1, 128, 80, 80]               0
AdaptiveAvgPool2d-227            [-1, 256, 1, 1]               0
          Linear-228                   [-1, 16]           4,096
            ReLU-229                   [-1, 16]               0
          Linear-230                  [-1, 256]           4,096
         Sigmoid-231                  [-1, 256]               0
        se_block-232          [-1, 256, 40, 40]               0
AdaptiveAvgPool2d-233            [-1, 512, 1, 1]               0
          Linear-234                   [-1, 32]          16,384
            ReLU-235                   [-1, 32]               0
          Linear-236                  [-1, 512]          16,384
         Sigmoid-237                  [-1, 512]               0
        se_block-238          [-1, 512, 20, 20]               0
          Conv2d-239          [-1, 256, 20, 20]         131,072
     BatchNorm2d-240          [-1, 256, 20, 20]             512
          Conv2d-241          [-1, 256, 20, 20]           2,304
     BatchNorm2d-242          [-1, 256, 20, 20]             512
           FReLU-243          [-1, 256, 20, 20]               0
            Conv-244          [-1, 256, 20, 20]               0
        Upsample-245          [-1, 256, 40, 40]               0
          Conv2d-246          [-1, 512, 40, 40]             512
          Conv2d-247          [-1, 128, 40, 40]          65,664
          Conv2d-248          [-1, 128, 40, 40]           1,152
     BatchNorm2d-249          [-1, 128, 40, 40]             256
           FReLU-250          [-1, 128, 40, 40]               0
      depth_conv-251          [-1, 128, 40, 40]               0
          Conv2d-252          [-1, 128, 40, 40]             128
          Conv2d-253          [-1, 128, 40, 40]          16,512
          Conv2d-254          [-1, 128, 40, 40]           1,152
     BatchNorm2d-255          [-1, 128, 40, 40]             256
           FReLU-256          [-1, 128, 40, 40]               0
      depth_conv-257          [-1, 128, 40, 40]               0
          Conv2d-258          [-1, 128, 40, 40]           1,152
          Conv2d-259          [-1, 128, 40, 40]          16,512
          Conv2d-260          [-1, 128, 40, 40]           1,152
     BatchNorm2d-261          [-1, 128, 40, 40]             256
           FReLU-262          [-1, 128, 40, 40]               0
      depth_conv-263          [-1, 128, 40, 40]               0
      Bottleneck-264          [-1, 128, 40, 40]               0
          Conv2d-265          [-1, 512, 40, 40]             512
          Conv2d-266          [-1, 128, 40, 40]          65,664
          Conv2d-267          [-1, 128, 40, 40]           1,152
     BatchNorm2d-268          [-1, 128, 40, 40]             256
           FReLU-269          [-1, 128, 40, 40]               0
      depth_conv-270          [-1, 128, 40, 40]               0
          Conv2d-271          [-1, 256, 40, 40]          32,768
     BatchNorm2d-272          [-1, 256, 40, 40]             512
          Conv2d-273          [-1, 256, 40, 40]           2,304
     BatchNorm2d-274          [-1, 256, 40, 40]             512
           FReLU-275          [-1, 256, 40, 40]               0
            Conv-276          [-1, 256, 40, 40]               0
              C3-277          [-1, 256, 40, 40]               0
          Conv2d-278          [-1, 128, 40, 40]          32,768
     BatchNorm2d-279          [-1, 128, 40, 40]             256
          Conv2d-280          [-1, 128, 40, 40]           1,152
     BatchNorm2d-281          [-1, 128, 40, 40]             256
           FReLU-282          [-1, 128, 40, 40]               0
            Conv-283          [-1, 128, 40, 40]               0
        Upsample-284          [-1, 128, 80, 80]               0
          Conv2d-285          [-1, 256, 80, 80]             256
          Conv2d-286           [-1, 64, 80, 80]          16,448
          Conv2d-287           [-1, 64, 80, 80]             576
     BatchNorm2d-288           [-1, 64, 80, 80]             128
           FReLU-289           [-1, 64, 80, 80]               0
      depth_conv-290           [-1, 64, 80, 80]               0
          Conv2d-291           [-1, 64, 80, 80]              64
          Conv2d-292           [-1, 64, 80, 80]           4,160
          Conv2d-293           [-1, 64, 80, 80]             576
     BatchNorm2d-294           [-1, 64, 80, 80]             128
           FReLU-295           [-1, 64, 80, 80]               0
      depth_conv-296           [-1, 64, 80, 80]               0
          Conv2d-297           [-1, 64, 80, 80]             576
          Conv2d-298           [-1, 64, 80, 80]           4,160
          Conv2d-299           [-1, 64, 80, 80]             576
     BatchNorm2d-300           [-1, 64, 80, 80]             128
           FReLU-301           [-1, 64, 80, 80]               0
      depth_conv-302           [-1, 64, 80, 80]               0
      Bottleneck-303           [-1, 64, 80, 80]               0
          Conv2d-304          [-1, 256, 80, 80]             256
          Conv2d-305           [-1, 64, 80, 80]          16,448
          Conv2d-306           [-1, 64, 80, 80]             576
     BatchNorm2d-307           [-1, 64, 80, 80]             128
           FReLU-308           [-1, 64, 80, 80]               0
      depth_conv-309           [-1, 64, 80, 80]               0
          Conv2d-310          [-1, 128, 80, 80]           8,192
     BatchNorm2d-311          [-1, 128, 80, 80]             256
          Conv2d-312          [-1, 128, 80, 80]           1,152
     BatchNorm2d-313          [-1, 128, 80, 80]             256
           FReLU-314          [-1, 128, 80, 80]               0
            Conv-315          [-1, 128, 80, 80]               0
              C3-316          [-1, 128, 80, 80]               0
          Conv2d-317          [-1, 128, 40, 40]         147,456
     BatchNorm2d-318          [-1, 128, 40, 40]             256
          Conv2d-319          [-1, 128, 40, 40]           1,152
     BatchNorm2d-320          [-1, 128, 40, 40]             256
           FReLU-321          [-1, 128, 40, 40]               0
            Conv-322          [-1, 128, 40, 40]               0
          Conv2d-323          [-1, 256, 40, 40]             256
          Conv2d-324          [-1, 128, 40, 40]          32,896
          Conv2d-325          [-1, 128, 40, 40]           1,152
     BatchNorm2d-326          [-1, 128, 40, 40]             256
           FReLU-327          [-1, 128, 40, 40]               0
      depth_conv-328          [-1, 128, 40, 40]               0
          Conv2d-329          [-1, 128, 40, 40]             128
          Conv2d-330          [-1, 128, 40, 40]          16,512
          Conv2d-331          [-1, 128, 40, 40]           1,152
     BatchNorm2d-332          [-1, 128, 40, 40]             256
           FReLU-333          [-1, 128, 40, 40]               0
      depth_conv-334          [-1, 128, 40, 40]               0
          Conv2d-335          [-1, 128, 40, 40]           1,152
          Conv2d-336          [-1, 128, 40, 40]          16,512
          Conv2d-337          [-1, 128, 40, 40]           1,152
     BatchNorm2d-338          [-1, 128, 40, 40]             256
           FReLU-339          [-1, 128, 40, 40]               0
      depth_conv-340          [-1, 128, 40, 40]               0
      Bottleneck-341          [-1, 128, 40, 40]               0
          Conv2d-342          [-1, 256, 40, 40]             256
          Conv2d-343          [-1, 128, 40, 40]          32,896
          Conv2d-344          [-1, 128, 40, 40]           1,152
     BatchNorm2d-345          [-1, 128, 40, 40]             256
           FReLU-346          [-1, 128, 40, 40]               0
      depth_conv-347          [-1, 128, 40, 40]               0
          Conv2d-348          [-1, 256, 40, 40]          32,768
     BatchNorm2d-349          [-1, 256, 40, 40]             512
          Conv2d-350          [-1, 256, 40, 40]           2,304
     BatchNorm2d-351          [-1, 256, 40, 40]             512
           FReLU-352          [-1, 256, 40, 40]               0
            Conv-353          [-1, 256, 40, 40]               0
              C3-354          [-1, 256, 40, 40]               0
          Conv2d-355          [-1, 256, 20, 20]         589,824
     BatchNorm2d-356          [-1, 256, 20, 20]             512
          Conv2d-357          [-1, 256, 20, 20]           2,304
     BatchNorm2d-358          [-1, 256, 20, 20]             512
           FReLU-359          [-1, 256, 20, 20]               0
            Conv-360          [-1, 256, 20, 20]               0
          Conv2d-361          [-1, 512, 20, 20]             512
          Conv2d-362          [-1, 256, 20, 20]         131,328
          Conv2d-363          [-1, 256, 20, 20]           2,304
     BatchNorm2d-364          [-1, 256, 20, 20]             512
           FReLU-365          [-1, 256, 20, 20]               0
      depth_conv-366          [-1, 256, 20, 20]               0
          Conv2d-367          [-1, 256, 20, 20]             256
          Conv2d-368          [-1, 256, 20, 20]          65,792
          Conv2d-369          [-1, 256, 20, 20]           2,304
     BatchNorm2d-370          [-1, 256, 20, 20]             512
           FReLU-371          [-1, 256, 20, 20]               0
      depth_conv-372          [-1, 256, 20, 20]               0
          Conv2d-373          [-1, 256, 20, 20]           2,304
          Conv2d-374          [-1, 256, 20, 20]          65,792
          Conv2d-375          [-1, 256, 20, 20]           2,304
     BatchNorm2d-376          [-1, 256, 20, 20]             512
           FReLU-377          [-1, 256, 20, 20]               0
      depth_conv-378          [-1, 256, 20, 20]               0
      Bottleneck-379          [-1, 256, 20, 20]               0
          Conv2d-380          [-1, 512, 20, 20]             512
          Conv2d-381          [-1, 256, 20, 20]         131,328
          Conv2d-382          [-1, 256, 20, 20]           2,304
     BatchNorm2d-383          [-1, 256, 20, 20]             512
           FReLU-384          [-1, 256, 20, 20]               0
      depth_conv-385          [-1, 256, 20, 20]               0
          Conv2d-386          [-1, 512, 20, 20]         131,072
     BatchNorm2d-387          [-1, 512, 20, 20]           1,024
          Conv2d-388          [-1, 512, 20, 20]           4,608
     BatchNorm2d-389          [-1, 512, 20, 20]           1,024
           FReLU-390          [-1, 512, 20, 20]               0
            Conv-391          [-1, 512, 20, 20]               0
              C3-392          [-1, 512, 20, 20]               0
          Conv2d-393           [-1, 75, 80, 80]           9,675
          Conv2d-394           [-1, 75, 40, 40]          19,275
          Conv2d-395           [-1, 75, 20, 20]          38,475
================================================================
Total params: 3,627,857
Trainable params: 3,627,857
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.69
Forward/backward pass size (MB): 1552.48
Params size (MB): 13.84
Estimated Total Size (MB): 1571.01
----------------------------------------------------------------
torch.Size([1, 32, 320, 320])
Total GFLOPS: 9.539G
Total params: 3.628M

```



- 两个模型相比起来，baseline的总参数量7263745，更改后的总参数量为3627875，很明显，参数量几乎减少了一半。

- 来看看Forward/backward pass size (MB)，更改后的比baseline大了300多，Estimated Total Size (MB)也大了300多，显存占用变大了！

这里解释一下：Estimated Total Size (MB) 其实就是所有output的大小！先列出源代码，就是total_output_size， x2 for gradients，正向和反向传播。

- 故在更改网络以后，参数量减少了，但是速度也没有提升，反而下降了。
