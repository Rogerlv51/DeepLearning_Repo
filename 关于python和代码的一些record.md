## 什么是相对导入，为什么要使用相对导入？

- **使用方法为：from .tools import sum 就是在包前面加一个.即可**<br> 
- **好处是当我们自己定义的包名字和例如python自带的一些包名字相同时，如果用直接导入的方法import一个包名，那么系统可能会判定为python自带的包，无法使用正确的包名。注意相对导入只能使用from...import...这种方式** </br>

## Python中argparse 模块的使用（会发现很多深度学习项目中都有这个用法，比如yolov5）

- **argparse是Python内置的一个用于命令项选项与参数解析的模块，argparse模块可以让人轻松编写用户友好的命令接口。**
- **程序定义它需要的参数，然后argparse将弄清如何从sys.argv解析出那些参数。argparse模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息**

``` python
import argparse
# 首先，创建解析器ArgumentParser()对象
parser = argparse.ArgumentParser(description='CV Train')
# ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息
# 其次，添加参数调用add_argument()方法添加参数
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
# 这个epochs属性的名字叫做epochs，类型为int，默认情况下其值为10，对其的解释为Number of epochs to train->训练的epoch数
args = parser.parse_args()
# ArgumentParser 通过 parse_args() 方法解析参数
print(args.epochs)
```

- **具体的一些用法参见：https://zhuanlan.zhihu.com/p/513300085**