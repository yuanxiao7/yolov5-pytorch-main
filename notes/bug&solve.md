报错：NotImplementedError

- 原因：
  1. 父类中的方法，在子类中没有重写，但在在子类中调用父类的该方法
  2. 没有调用forward方法

