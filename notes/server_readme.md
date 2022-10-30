Microsoft Windows [版本 10.0.19044.1766]
(c) Microsoft Corporation。保留所有权利。

C:\Users\Happy>ssh lcx@172.25.6.99
The authenticity of host '172.25.6.99 (172.25.6.99)' can't be established.
ECDSA key fingerprint is SHA256:6DeJKXzUbyMP1JiSIx3g3BH7XOyanAGgI9cXrSy3fMU.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added '172.25.6.99' (ECDSA) to the list of known hosts.
lcx@172.25.6.99's password:
Welcome to Ubuntu 18.04.6 LTS (GNU/Linux 5.4.0-42-generic x86_64)sshss

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

 * Canonical Livepatch is available for installation.
   - Reduce system reboots and improve kernel security. Activate at:
     https://ubuntu.com/livepatch

29 updates can be applied immediately.
1 of these updates is a standard security update.
To see these additional updates run: apt list --upgradable

Your Hardware Enablement Stack (HWE) is supported until April 2023.
*** System restart required ***
Last login: Sun Jul 10 12:58:48 2022 from 10.33.107.110
(base) lcx@lcx:~$ ls
dataset  examples.desktop  miniconda3  segmenter-master  公共的  模板  视频  图片  文档  下载  音乐  智能垃圾分类  桌面
(base) lcx@lcx:~$ mkdir shuqikaohe
(base) lcx@lcx:~$ conda inf0 list

CommandNotFoundError: No command 'conda inf0'.
Did you mean 'conda info'?

(base) lcx@lcx:~$ conda info list
^Z
[1]+  已停止               ( __add_sys_prefix_to_path; "$CONDA_EXE" $_CE_M $_CE_CONDA "$@" )
(base) lcx@lcx:~$ conda env list
# conda environments:
#
base                  *  /home/lcx/miniconda3
tensorflow2.4.1          /home/lcx/miniconda3/envs/tensorflow2.4.1
torch10.2                /home/lcx/miniconda3/envs/torch10.2

(base) lcx@lcx:~$



其实服务器是一个具有很好显卡配置的计算机，如这个就是使用 linux 系统的一个主机，而我们只是远程链接到他那里，利用他的算力和空间，在那里创建一个文件夹，把我们想要跑的项目上传到上面，在这个文件夹里的操作跟在本机Linux系统的文件操作是一样的。我们先要在上面配置我们需要的环境，nvcc -V 可以查看 cuda 的版本（这里的cuda是预先已配置好的，如果你没有配置的话，你可以百度试试），然后就用conda创建并配置环境，其操作和 windows 差不多，配置好以后，在专业版的 pycharm 中链接服务器，主要是解释器，这里推荐一篇文章https://blog.csdn.net/hehedadaq/article/details/118737855，这篇文章写得不错，配置好以后就可以运行项目了。

不知道什么原因，小编的pycharm有时候自动上传失败，于是就要到服务器上修改文件了，vim是一个很好用的文本编辑器，不用担心觉得没见过，不会用，小编一开始连他是啥玩意都不知道，是一位小伙伴推荐的，我说不会，他直接让我百度，没辙后面区百度查，大胆尝试，还是成功改好了，举一个例子（今天改的那个utils_fit.py）

加油！今天还是学到东西的！