7月13日

今天开始正式训练，用本机训练，服务器还要再学学才会用。

第一个10个epoch

![image-20220713203648641](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220713203648641.png)



第20个epoch

![image-20220713203722787](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220713203722787.png)

第30个epoch

![image-20220713203747801](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220713203747801.png)



总结：

- 一共30个epoch，训练的指标很令人难受，因为他实在是太奇怪了，看到我自闭，觉得没救了，后面我随意看看，发现标签与图片序号没对上，我瞬间明白了，应该是我没有导入成功的原因，后面又去看了代码，发现我参数设置得不合理，改正过来应该没有什么问题。