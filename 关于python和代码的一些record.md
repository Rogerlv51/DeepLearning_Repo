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
- **当然如果这些参数的使用仅限于当前这个python文件，也可以不写def main()函数，直接在if \_\_name\_\_ == "\_\_main\_\_"中传入参数即可**

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
</br>

## **关于python中使用iter迭代器和next提取**
- **iter( object )：生成可迭代对象的迭代器；object必须是可迭代对象，比如list、tuple、dict等，深度学习中我们就多用Dataloader就是一个迭代器**
- **next( iter, end_num )：每执行依次，按顺序每次从迭代器中提取一个元素**

```python
b = [1,3,4,5]
b = iter(b)   # 获得可迭代对象的迭代器
for x in range(8):
    print( next(x) )   # 每执行一次next()函数，就依次抽一个元素出来

# 结果：
1
3
4
5
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-30-a8068fff7e9c> in <module>
      2 b = iter(b)
      3 for x in range(8):
StopIteration:
```
</br>

## **关于python中常用的property装饰器，classmethond装饰器，staticmethod装饰器**
- **这个装饰器第一个作用是可以在调用方法的时候强制省略()，也就是说不允许用户传递任何参数，否则会报错**
```python
class DataSet(object):
  @property
  def method_with_property(self): ##含有@property
      return 15
  def method_without_property(self): ##不含@property
      return 15

l = DataSet()
print(l.method_with_property)   # 加了@property后，可以用调用属性的形式来调用方法,后面不需要加()
print(l.method_without_property())   # 没有加@property , 必须使用正常的调用方法的形式，即在后面加()
```
- **另一方面在于python进行属性的定义时，没办法设置私有属性，因此要通过@property的方法来进行设置。这样可以隐藏属性名，让用户进行使用的时候无法随意修改。即使用一个方法来包装了属性，实际上该方法返回的就是类的一个属性**
```python
class DataSet(object):
    def __init__(self):
        self._images = 1
        self._labels = 2 #定义属性的名称
    @property
    def images(self): #方法加入@property后，这个方法相当于一个属性，这个属性可以让用户进行使用，而且用户有没办法随意修改。
        return self._images 
    @property
    def labels(self):
        return self._labels
    @labels.setter   # 使用这个装饰器，则可以改变私有属性labels的值
    def labels(self, name):
        self._labels = name
l = DataSet()
#用户进行属性调用的时候，直接调用images即可，而不用知道属性名_images，因此用户无法更改属性，从而保护了类的属性。
print(l.images)  # 加了@property后，可以用调用属性的形式来调用方法,后面不需要加()，这就显得好像在调用属性一样
```
</br>

- **使用classmethod装饰器可以把方法变成实例化对象的函数使用，而staticmethod可以把方法变成静态，可以不需要实例化对象直接通过类调用**
```python
class User:
    def __init__(self, name):
        self._name = name
    
    @classmethod
    def gen_user(cls):
        return cls('handsome')
    
    @staticmethod
    def length():
        return 18

if __name__ == '__main__':
    new = User.gen_user()
    print(new.name)
    print(User.length())
```
</br>

## **关于tqdm库，非常重要，深度学习常用库**
- **Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，总之是用来显示进度条的，jupyter中也常用**
```python
    ## 用法可以是下面的
    with tqdm(train_dataloader, unit='batch', leave=False) as pbar:
        pbar.set_description(f'training')
    # 也可以循环里面用
    for i in tqdm(range(100), desc="循环")    # desc就是个命令行输出的打印描述
    ## 还有很多用法去看官方文档
```


## **关于语句后面出现“ ; ”的作用**
- **有时我们会发现很多语句的末尾会出现一个分号，按道理来说python语句是不需要写分号的，那么其作用是什么呢？**
- **主要是为了抑制函数最后一行的输出，有一些函数比如plt.hist在输出直方图的时候还会附加的打印函数的输出值，但往往我们不需要看函数的输出，只是想关注图像长啥样，因此我们会在后面加一个“;”来组织函数输出**


## **关于torch.nn.Identity()**
```python
关于torch.nn.Identity()，这个函数的作用是返回一个输入的副本，即不做任何操作，
但是这个函数可以用来做一个占位符，比如在模型中有一些层是可选的，那么我们可以在这些层的位置上使用这个函数来占位，
这样就不用在代码中写if else语句来判断是否使用这些层了。目的是忽略一些层的输出，加载预训练模型时常用
```