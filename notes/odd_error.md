## Descriptors cannot not be created directly.

## Descriptors cannot not be created directly.



sudo+ssh://lcx@172.25.6.99:22/home/lcx/miniconda3/envs/hlh_torch10.2/bin/python3.8 -u /home/lcx/shuqikaohe/train.py

TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:

 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates

解决： 按提示下载更低版本的protobuf



## 使用vim文本编辑器

先 cd 来到所要修改文件的根目录，之后再 vim + filename 进入文件，在你进入insert模式的前一页面就是正常模式，然后就可以根据百度教程修改你想要修改的内容了，保存后退出是，先按 ESC 键回到正常状态，再按一次，然后输入 :wq 回车 即可保存并退出文件。



## 分布式训练报错 

RuntimeError： Address already in use

 raise subprocess.CalledProcessError(returncode=process.returncode,
subprocess.CalledProcessError: Command '['/home/lcx/miniconda3/envs/hlh_torch10.2/bin/python', '-u', 'train.py', '--local_rank=1']' returned non-zero exit status 1.

 RuntimeError: NCCL error in: /pytorch/torch/lib/c10d/ProcessGroupNCCL.cpp:784, unhandled system error, NCCL version 2.7.8



### 尝试1 

在运行的代码前加  export NCCL_BI_DISABLE=1 （可能是在内部加）

```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2  export NCCL_BI_DISABLE=1 train.py
```

但还是报错



### 尝试2

输入命令如下

export OPM_NUM_THREAD=1

ech $OPM_NUM_THREAD

1

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2  train.py

还是如上报错



### 尝试3   

- 出处   https://zhuanlan.zhihu.com/p/296803907

htop

**htop是top的升级版,允许用户监视系统上运行的进程及其完整的命令行**

1. 系统不会自带，若是Ubuntu，sudo apt, 若是Centos， yum 可以安装。根据不过系统来进行安装
2. 支持用户交互，可以通过鼠标来kill进程而不用通过输入其PID，支持用鼠标上下拖动，且不同的颜色代表不同的意思。
3. 允许用户根据CPU，内存和时间间隔对进程进行排序

htop安装

Ubuntu安装

sudo adt install htop

htop 进程显示

F10 退出界面

sudo netstat -ano      netstat 命令为你的 Linux 服务器提供了监测和网络故障排除的重要观察手段。

sudo netstat -anop | grep "127.0.0.1"

![image-20220802102435256](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220802102435256.png)



![image-20220802102519016](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220802102519016.png)

![image-20220802102547898](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220802102547898.png)

CUDA_VISIBLE_DEVICES=2,3;NCCL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=2 train.py --master_port 2950

还是报错

 lcx:28468:28468 [1] NCCL INFO Bootstrap : Using [0]eno2:172.25.6.99<0

拒绝连上

![image-20220802102902007](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220802102902007.png)

![image-20220802103636866](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220802103636866.png)

KO！

### 失败总结

最后，可能是环境问题，建议可以的话，remake。

也可能是不能两个人同时使用ddp分布式训练， 那个端口同时传送的问题。

还有装软件啥的不装最新，装第二新就好，太老的不兼容，容易出现问题！



最后实验一次，不可以同时进行ddp分布式训练，可以ddp加dp训练，当ddp同时训练时就会出现bug。



















