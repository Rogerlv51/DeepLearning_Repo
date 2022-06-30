# 使用nni库来对模型超参数调优，https://nni.readthedocs.io/zh/stable/index.html
import nni
from nni.experiment import Experiment

# 第一步，定义搜索空间search_space，也可以额外用一个json文件，具体看官方文档
'''
在模型代码中,我们准备了三个需要调优的超参: features、lr、momentum。
现在，我们需要定义的它们的“搜索空间”，指定它们的取值范围和分布规律。
假设我们对三个超参有以下先验知识：
features的取值可以为128、256、512、1024;
lr的取值在0.0001到0.1之间，其取值符合指数分布；
momentum的取值在0到1之间

在NNI中,features的取值范围称为 choice, lr的取值范围称为 loguniform, 
momentum的取值范围称为 uniform 。 您可能已经注意到了，这些名称和 numpy.random 中的函数名一致。
'''
searchspace = {
    'features': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}

# 第二步，配置实验，设置一些参数
# 在本教程中我们使用 local 模式的实验，这意味着实验只在本机运行，不使用任何特别的训练平台
experiment = Experiment('local') 
# 在nni中评估一组超参的过程称为一个trial，模型的代码称为trail代码
experiment.config.trial_command = 'python NNI_Tutorials.py'
experiment.config.trial_code_directory = '.'
# 如果 trial_code_directory 是一个相对路径，它被认为相对于当前的工作目录

# 配置搜索空间
experiment.config.search_space = searchspace
# 配置调优算法
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
# 尝试10组超参，并且每次并行地评估2组超参
experiment.config.max_trial_number = 20
experiment.config.trial_concurrency = 2

# 如果max_trial_number和max_experiment_duration都没有设置，实验将会一直运行，直到您按下 Ctrl-C


# 第三步，运行实验
# 可以指定一个端口来运行它，教程中我们使用8080端口
experiment.run(8080)
input('Press enter to quit')
experiment.stop()
# 实验完全停止之后，可以使用nni.experiment.Experiment.view()重新启动网页控制台