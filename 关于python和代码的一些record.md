## **什么是相对导入，为什么要使用相对导入？**

- **使用方法为：from .tools import sum 就是在包前面加一个.即可**<br> 
- **好处是当我们自己定义的包名字和例如python自带的一些包名字相同时，如果用直接导入的方法import一个包名，那么系统可能会判定为python自带的包，无法使用正确的包名。注意相对导入只能使用from...import...这种方式** </br>

## **Python中argparse 模块的使用（会发现很多深度学习项目中都有这个用法，比如yolov5）**

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

## **为什么有的项目里面在训练的文件出现def main(args)？**

- **这种方式用于和argparse联动，同时方便我们使用终端运行python文件并指定参数**
- **我们可以在def main()中处理需要给程序传入的参数，把parser中的参数传到自己的函数或者变量中**
- **当然如果这些参数的使用仅限于当前这个python文件，也可以不写def main()函数，直接在if __name__ == "__main__"中传入参数即可**

```python
import argparse
def do_something(arg):
    print(arg)

def do_something_else(arg):
    print(arg)

def main(args):
    do_something(args.an_arg)
    do_something_else(args.another_arg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--an_arg", type=int, default=10, help="第一个参数")
    parser.add_argument("--another_arg", type=int, default=10, help="另一个参数")
    args = parser.parse_args()
    main(args)
```
</br>

## **关于python中map函数还有lambda表达式**
- **python3中map函数返回的是一个迭代器，想要打印输出的话必须使用循环的方式，或者转化成list**
- **lambda表达式实际上就是一个函数，只是形式上看起来比def简单**
```python
def square(x) :         # 计算平方数
    return x ** 2

# 第一个参数是一个函数，第二个参数是要传入函数的值
map(square, [1,2,3,4,5])    # 计算列表各个元素的平方
<map object at 0x100d3d550>     # 返回迭代器

list(map(square, [1,2,3,4,5]))   # 使用 list() 转换为列表
[1, 4, 9, 16, 25]

list(map(lambda x: x ** 2, [1, 2, 3, 4, 5]))   # 使用 lambda 匿名函数
[1, 4, 9, 16, 25]
```