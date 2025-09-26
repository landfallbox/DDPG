"""
常量定义 (ASHRAE 专用版本)
移除 Gym 环境相关常量。
"""
import os
import torch
import torch.nn as nn
from collections import namedtuple

basedir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir: str = os.path.join(basedir, "output")

# 经验元组
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
criterion = nn.MSELoss()

# ASHRAE 数据路径
ashrae_data_dir = os.path.join(basedir, "dataset", "ASHRAE", "ashrae-energy-prediction")
ashrae_data_processed_path = os.path.join(basedir, "dataset", "ASHRAE", "ashrae-energy-prediction-processed.csv")
ashrae_building100_processed_path = os.path.join(basedir, "dataset", "ASHRAE", "ashrae-energy-prediction-processed-building100.csv")

target_col = 'meter_reading'
col_timestamp = 'timestamp'

"""默认超参数"""
default_seed = 42
