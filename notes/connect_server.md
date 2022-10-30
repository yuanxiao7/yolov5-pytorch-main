# 服务器的使用

目录如下

[TOC]

7月12日

## 连接服务器

今天连接服务器，我们使用服务器相当于在另一台电脑上建一个文件夹，利用他的算力来训练我们的模型，不过有一点对小编而言，有点恼，因为小编用的服务器环境是乌班图（Ubuntu）而小编只用过Windows。要明确一点，可视化功能在我们本地的环境里是可以展示的，但是在服务器上面就不行了，因为你是在服务器里运行的，不是本地，对应的摄像头之类的服务器可能不是，所以，你可以用它来训练模型，但是可视化展示还是等模型训好了再在本地展示就好了。

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------



首先，通过ssh链接，如 

- ssh 名字@域址 回车
- passward 密码  回车

进入远程系统，你可以认为来到主机的桌面，一个没有窗口显示的桌面。

进入你在主机创建的主文件夹

- cd 文件名  回车
- ll  回车 查看目录（包括日期等信息）
- ls 回车 之查看目录
- cd 文件名 回车，来到你的项目的根目录



---------------------------------------------------------



接下来开始创建的得虚拟环境

- 建议先配置anaconda，再用conda创建虚拟环境
- 配置好cuda和cdnn这里百度，小编是利用已有的
- - 首先敲命令
  
  1. conda create --环境名称 python版本 回车  （如：conda create --torch python3.7 回车  即torch环境，python为3.7版本
  2. conda activate 环境名称  回车 （激活环境
  3. 利用conda下载你想要的对应cuda得torch版本
  4. conda下载项目对应的requirement.txt配置环境
  5. 可以用conda或者pip install 第三方库
  6. 速度慢，可以换源，建议清华源
  7. 配置解释器后，通过专业版pycharm连接本地

关于如何获得专业版pycharm推荐  https://blog.csdn.net/weixin_45459911/article/details/104767525

关于在pycharm中远程连接推荐 https://blog.csdn.net/hehedadaq/article/details/118737855

当你的远程解释器可以应用于你的pycharm中时，说明连接成功

---------------



## 常见问题

一般情况会出现本地和远程更新不同步，这时我们就考虑直接在远程修改已上传但未能更新的代码，这时就用到了一个神器vim，(当时小编可想哭了，这是小伙伴建议用的，但是小编听都没听过，但我又要改，只好硬着头皮上)，这里有个教程，推荐  [vim](https://zhuanlan.zhihu.com/p/68111471) （小编为啥老是推荐呢，小编没有时间整理，但是又不想学了一遍后面又忘记，就把大致过程记录下来，下一次遇到就可以直接看）



## 使用vim

我们先来到服务器得文件根目录，开始更改文件内的代码

- vim 文件名（包括后缀）
- 进入文件内部，整个窗口显示文件的内容，当你开始进入insert状态时，你的前一个状态即为正常状态
- 更具相应的 i I a A o O s S 来修改你的文件内容
- 按ESC回到正常状态，按ESC 敲:wq 回车 即退出所要修改的文件
- 回到刚进入文件是的页面
- 至此，文件修改完毕

### vim鼠标模式打开与关闭

开启鼠标模式
:set mouse=x, x取值如下, 例如:set mouse=a, 开启所有模式的mouse支持

n 普通模式
v 可视模式
i 插入模式
c 命令行模式
h 在帮助文件里，以上所有的模式
a 以上所有的模式
r 跳过 |hit-enter| 提示
A 在可视模式下自动选择
关闭鼠标模式
:set mouse=, =后面不要跟任何值, 可以关闭鼠标模式

----------------



7月17日

## 文件创建与删除

这里的操作是直接使用相对路径的。

### 创建文件夹

在home里创建子文件夹或文件,先  cd + 文件夹名称  来到你想要创建文件夹的根目录，输入以下指令

- mkdir yolo_py  回车   yolo_py为文件夹名  
- touch train.py  回车  其中train为文件名，.py为文件后缀，几所要创建的文件类型
- ls  即可看到新建的文件夹  

### 删除文件夹

先  cd 文件夹名称  来到你想要删除的文件所在的目录，输入以下指令

- ls  查看当前目录，找到要删除的文件夹
- rm  -f  文件名（包括后缀）
- ls  查看文件已删除

------------



## 上传数据集

### 情况一

- 你准备用pycharm远程连接服务器
- 在你pycharm链接服务器之前，将数据集和标签处理好，再连接服务器，和代码一起上传即可。

### 情况二

- 如果你想要修改你的文件，可是你的pycharm不能和远程同时更新，可以试试
- 当你的数据集小于2G，你可以将其打包成zip压缩包上传，然后解压（具体请百度，没试过）
- 当你的数据集大于2G的话，你可以分别将标签和图片上传
  - 注意，上传文件的终端是在本地，不是在服务器上
  - 本地终端：在电脑左下角搜索框里输入cmd 点击以管理员的身份运行，进入本地终端
  - 服务器终端：在你本地终端通过ssh username@192.12.3.66 并输入密码进入的linux终端
  - 在本地终端输入以下命令
    - scp -r local_file  username@192.12.3.66:/home/username/file
    - passward:
  - scp  是命令， -r 是参数
  - local_file  是本地上传文件的绝对路径
  - username 是服务器账号
  - 192.12.3.66 是你的服务器ip地址
  - /home/username/file  是你想要在服务器存放的路径

---------



7月19日

## 服务器后台训练

### 一、连接服务器

- C:\Windows\system32>ssh  username@192.12.3.66
   username@192.12.3.66's password:
  Welcome to Ubuntu 17.04.6 LTS (GNU/Linux 5.4.0-42-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

 * Canonical Livepatch is available for installation.
   - Reduce system reboots and improve kernel security. Activate at:
     https://ubuntu.com/livepatch



### 二、进到服务器界面，查看是否已经装有tmux（和screen一样，比screen好用）

- (base) user@user:~$ tmux

- Command 'tmux' not found, but can be installed with:

- sudo snap install tmux  # version 2.3, or
  sudo apt  install tmux

- See 'snap info tmux' for additional versions.

  

### 三、 弹出提示，没有，并且说明如何下载，输入下载指令

- (base) user@user:~$ sudo apt install tmux
  [sudo] user 的密码：
  正在读取软件包列表... 完成
  正在分析软件包的依赖关系树
  ......
  解压缩后会消耗 689 kB 的额外空间。
  您希望继续执行吗？ [Y/n]
  获取:1 http://cn.archive.ubuntu.com/ubuntu bionic/main amd64 libutempter0 amd64 1.1.6-3 [7,898 B]
  ......
  正在处理用于 man-db (2.8.3-2ubuntu0.1) 的触发器 ...
  
  

### 四、tmux下载完毕，开始尝试使用

- (base) user@user:~$ tmux new -s train                                              打开个窗口，名称 train

  

  [detached (from session train)]                                                       保持窗口运行，并退出窗口，先按下 Ctrl + B 然后同时放开,
  [detached (from session train)]					                                   再按 D 即可退出界面 									

  

- (base) user@user:~$ tmux a -t train                                                    进入该窗口工作区
  [detached (from session train)]
  (base) user@user:~$ [detached (from session train)]

  

- (base) user@user:~$ tmux ls                                                               查看有多少个窗口
  train: 1 windows (created Sun Jul 17 21:31:41 2022) [237x62]



- (base) user@user:~$ tmux a -t train                                                     再次进入窗口，并且结束该窗口 exit 回车 或者使用快捷键Ctrl + D
  [exited]                                                                                             回到进入该窗口的目录
  (base) user@user:~$          

- 复制功能：按住shift 鼠标左击并滑动想要复制的内容                                                                  

- ctrl d 直接断开远程连接,并返回pc端链接的命令行页面

----------



### tmux窗口滑动

​                                                                                                     **法一** 历史浏览

![image-20220722111723013](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220722111723013.png)



​                                                                                               **法二**   窗口滑动

![image-20220722111947109](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220722111947109.png)

- Ubuntu Tmux 启用鼠标滚动 在Ubuntu上使用 Tmux 是一件非常舒服的事，但有时使用鼠标滚轮时，和平时使用终端的习惯不怎么一致，因此可以设置启用鼠标滚轮。 具体方式： 按完前缀ctrl+B后，再按分号：进入命令行模式， 输入以下命令： **set - g mouse on** 就启用了鼠标滚轮，可以通过鼠标直接选择不同的窗口，也可以上下直接翻页。



<img src="C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220718080834882.png" alt="image-20220718080834882" style="zoom:150%;" />



![image-20220718080846559](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220718080846559.png)

----





7月21日

## conda和pip list 不一致 

因为网络代码量几千行，一行一行看，比较慢，只能是看到哪里，遇到那个问题就解决那个问题了（多的话就一个一个来）。下面这个是前两天的报错，

RuntimeError: Couldn't load custom C++ ops. This can happen if your PyTorch and torchvision versions are incompatible, or if you had errors while compiling torchvision from source. For further information on the compatible versions, check https://github.com/pytorch/vision#installation for the compatibility matrix. Please check your PyTorch version with torch.__version__ and your torchvision version with torchvision.__version__ and verify if they are compatible, and if not please reinstall torchvision so that it matches your PyTorch install.  （即上图报错）

​		一开始以为，这个报错我以为是因为服务器装的cuda和cudnn与我装的pytorch不匹配，因为他是9.1的，官网没有，我就按照9.2的配置，后面想想也不对，应该配置是没有错的。

​		百度没找到，就找学长，查了一下，发现，我用的环境是base环境，而且pip list 和 conda list 没对应上（猜测，可能是环境有点问题，想初始化一下conda，即conda init，但是没有成功），pip的是另一个版本的 torchvision，刚好我用的环境就是pip的那个torchvision，卸了重装，再运行，还是不行，还是用base环境，而且它居然还在！后面试试删掉torch，conda和pip list都没有了，后面直接删掉 pip uninstall torch torchvision torchaudio ，到官网下载torch和torchvision（torchaudio时间原因，这个音频相关的就不下了）到本地，然后 conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=9.2 -c pytorch 发现，在conda list 安装好了，但是pip list还是没变，然后检测一下：

(hlh_torch10.2) lcx@lcx:~$ python
Python 3.8.13 (default, Mar 28 2022, 11:38:47)
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.

>>> import torch
>>> torch.cuda.is_available()
>>> True
>>> torch.tensor([1,23])
>>> tensor([ 1, 23])
>>> a = torch.tensor([1,23])
>>> a.cuda()
>>> tensor([ 1, 23], device='cuda:0')

以为可以用了，后面跑项目还是不行。有如下报错。

ValueError: Error initializing torch.distributed using env:// rendezvous: environment variable RANK expected, but not set  （这个是分布式，需要设置rank变量，但是我没有设置，我后面就不用分布式了，学长说yolo是一个小网络，一张卡就可以了）

以下是下载torch和torchvision的过程，快一点的，先到官网，下载到本地，再从桌面上传到服务器

先用输入 pwd 回车 查看当前目录，另开本地窗口（不是在服务器上的！）上传文件。

![image-20220721223716557](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220721223716557.png)



![image-20220721221803727](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220721221803727.png)![image-20220721221841589](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220721221841589.png)

上传完毕以后，ls 查看当前目录，然后直接

 pip install torch-1.7.1+cu92-cp38-cp38-linux_x86_64.whl  torchvision-0.8.2+cu92-cp38-cp38-linux_x86_64.whl  下载就好

--------------



## 一些基本的命令



1. which python： 当前python的路径

2. whereis python:   所有python的路径

3. ipython ： 很方便的交互式脚本，详情：  https://blog.csdn.net/gavin_john/article/details/53086766

   ​                  下载方式  pip install ipython    or   conda install ipython

4. ip a：    在linux系统可以查看ip地址信息的命令 类似 linux的 ifconfig，windows ipconfig

5. conda update -n base：    更新basse环境   base是安装anaconda中的默认python版本

6. conda update conda：    conda更新

7. ks ：     是 [KubeSphere](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fkubesphere%2Fkubesphere) 的命令行客户端，可以简化用户、开发者的日常操作。 许多系统管理员宁愿使

   ​             用自动化的安装方法来安装红帽  企业 Linux.为了满足这种需要,红帽创建了kickstart安装方法                 

8. kls

9. watch nvidia-smi：  查看显卡（2s刷新）  ctrl c 退出  建议不要使用，会影响cuda的操作

10. ctrl x：  切换终端

11.  CUDA_VISIBLE_DEVICES=0 python train.py：    指定GPU，运行根目录下的py文件

12. kill -9 PID：  完全杀掉进程

---------





