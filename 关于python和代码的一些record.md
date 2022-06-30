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
```
name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
action - 当参数在命令行中出现时使用的动作基本类型，其默认值是store。
nargs - 命令行参数应当消耗的数目。
const - 被一些 action 和 nargs 选择所需求的常数。
default - 当参数未在命令行中出现并且也不存在于命名空间对象时所产生的值。[不指定参数时的默认值]
type - 命令行参数应当被转换成的类型。
choices - 可用的参数的容器，即参数只能在这里面选择
required - 此命令行选项是否可省略 （仅选项可用）。
help - 一个此选项作用的简单描述。
metavar - 在使用方法消息中使用的参数值示例。
dest - 被添加到 parse_args() 所返回对象上的属性名。
```

- **具体的一些用法参见：https://zhuanlan.zhihu.com/p/513300085**