## 文本预处理
## 对于序列数据处理问题，文本是最常⻅例子之一。例如，一篇文章可以被简单地看作是一串单词序列，甚至是一串字符序列
## 文本的常⻅预处理步骤：1. 将文本作为字符串加载到内存中。2. 将字符串拆分为词元（如单词和字符）
## 3. 建立一个词表，将拆分的词元映射到数字索引。4. 将文本转换为数字索引序列，方便模型操作

import collections
import re 
from d2l import torch as d2l

## 第一步：首先读取一个文本数据集
## 下面的函数将数据集读取到由多条文本行组成 的列表中，其中每条文本行都是一个字符串
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine(): 
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f: 
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]  # 忽略大小写和标点符号读取

lines = read_time_machine() 
print('# 文本总行数: {}'.format(len(lines)))
print(lines[0]) 
print(lines[10])

## 第二步：词元化
## 下面的tokenize函数将文本行列表（lines）作为输入，列表中的每个元素是一个文本序列（如一条文本行）。
## 每个文本序列又被拆分成一个词元列表，词元（token）是文本的基本单位。最后，返回一个由词元列表组成的列表

def tokenize(lines, token = 'word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]     # 直接用空格分开单词即可
    if token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
print("\n")


## 第三步：构建词表
## 词元的类型是字符串，而模型需要的输入是数字，因此这种类型不方便模型使用。
## 现在，让我们构建一个字 典，通常也叫做词表（vocabulary），用来将字符串类型的词元映射到从0开始的数字索引中
def count_corpus(tokens):
    """统计词元的频率""" 
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
    # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]   # 把两级列表直接展开成一个list统计元素个数
    return collections.Counter(tokens)   # 这个函数可以统计每个单词出现的次数

class Vocab: 
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None: 
            tokens = []
        if reserved_tokens is None: 
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens) 
        # 对它们的唯一词元进行统计，得到的统计结果称之为语料（corpus）
        # 然后根 据每个唯一词元的出现频率，为其分配一个数字索引。很少出现的词元通常被移除，这可以降低复杂性
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # 语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元"<unk>"。
        # 我们可以选择增加一个列表，用于保存那些被保留的词元，即前面已经定义的reserved_tokens
        # 例如：填充词元（"<pad>"）；序列开始词元（"<bos>"）；序列结束词元 （"<eos>"）
        self.idx_to_token = ['<unk>'] + reserved_tokens    # 未知词元的索引为0，因此这里把<unk>放在列表的第一个
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:    # 把那些出现次数低于预设值的词元都丢掉
                break 
            if token not in self.token_to_idx:    # 去重，已经在tokenlist中的标好索引的就没必要再放进去
                self.idx_to_token.append(token) 
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property 
    def unk(self): # 未知词元的索引为0
        return 0
    
    @property 
    def token_freqs(self):
        return self._token_freqs

vocab = Vocab(tokens) 
print(list(vocab.token_to_idx.items())[:10])

## 最后，我们可以将每一条文本行转换成一个数字索引列表。
for i in range(len(tokens)): 
    print('文本:', tokens[i]) 
    print('索引:', vocab[tokens[i]])