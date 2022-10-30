## 模块构建

- 这个改动是在替换focusnet的组件上更改的，将深度可分离卷积换成添加通道信息的标准卷积+bn+act，是通道特征更加突出。



此改动再CSPdarknet1.py中的添加SE_Conv和attentions.py中的se_block的更改。

```python

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

# 自定义自动padding模块
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        # auto-pad #如果k是整数，p为k与2整除后向下取整；如果k是列表等，p对应的是列表中每个元素整除2。
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # 当指定p值时按照p值进行填充，当p值为默认时则通过autopad函数进行填充
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())  # 加入非线性因素
        # self.act    = FReLU(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act    = Hardswish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # 加入非线性因素  frelu目前除了召回率其他都比baseline好的模型

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))  # 完整的经过一层卷积操作 即，conv + bn + act

class SE_Conv(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(SE_Conv, self).__init__()
        self.cv1 = Conv(c1, c1, 1, 1)
        self.cv3 = Conv(2 * c1, c2, 1)  # act=FReLU(c2)
        self.se_block = se_block(c2)

    def forward(self, x):
        return self.cv3(torch.cat(  # 两次调整的cat
            (
                self.cv1(x),
                self.se_block(x)
            )
            , dim=1))



model = SE_Conv(512, 512)
map = torch.randn(2, 512, 26, 26)
print('map: ', map)
if __name__ == "__main__":
    outputs = model(map)
    print('output: ', outputs)

```



```python

class SE_Conv(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(SE_Conv, self).__init__()
        self.cv1 = Conv(c1, c1, 1, 1)
        self.cv3 = Conv(2 * c1, c2, 1)  # act=FReLU(c2)
        self.se_block = se_block(c2)

    def forward(self, x):
        return self.cv3(torch.cat(  # 两次调整的cat
            (
                self.cv1(x),
                self.se_block(x)
            )
            , dim=1))

```



```python

# SENet的代码实现通道注意力集中机制，经全局平均池化，提取出每个channel的信息

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        print(x.shape)
        print((x*y).shape)
        print(y.shape)
        print(y.expand_as(x).shape)
        return x + x * y     # 改动处

result：
torch.Size([1, 64, 320, 320])
torch.Size([1, 64, 320, 320])
torch.Size([1, 64, 1, 1])
torch.Size([1, 64, 320, 320])

```





下方随即擦除的展示

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220901112921543.png" alt="image-20220901112921543" style="zoom:67%;" />

<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220901113007865.png" alt="image-20220901113007865" style="zoom:50%;" />





速度

和图片对应： 0.02908216953277588 seconds, 34.38533012033337 FPS, @batch_size 1

0.02754760980606079 seconds, 36.30079005184648 FPS, @batch_size 1
