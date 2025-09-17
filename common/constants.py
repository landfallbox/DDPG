"""
@Project     : api-annotator 
@File        : constants.py
@IDE         : PyCharm 
@Author      : landfallbox
@Date        : 2025/9/1 星期一 15:53 
@Description : 常量定义
"""
import os
import torch
import torch.nn as nn

from collections import namedtuple


# 项目的根目录
basedir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 存放实验结果的目录
results_dir: str = os.path.join(basedir, "output")

env_pendulm_v1 = "Pendulum-v1"

# 定义一个命名元组来存储经验
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

# 均方误差损失函数
criterion = nn.MSELoss()