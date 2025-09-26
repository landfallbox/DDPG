# DDPG (ASHRAE Only Edition)

本版本已彻底去除对 Gym / gymnasium 的依赖，仅支持对 ASHRAE (energy prediction) 处理后的表格数据进行基于 DDPG 的策略学习/回归式预测（可选 bandit 单步模式）。

## 特性
- 特征筛选：按与 `meter_reading` 的 Pearson 相关系数阈值过滤列
- 特征编码：DeepForest (若可用) 或 RandomForest + 原始标准化特征拼接
- 奖励定义：`reward = - (pred - target)^2`，动作即预测值 ∈ [-1, 1]
- 单步模式 (bandit)：`--single_step` 每条样本独立 episode，适合监督回归风格
- 评估指标：自动统计 MSE / MAE，并保存曲线 (.png) 与 .mat 文件（含 reward/mse/mae）
- 仅使用 PyTorch 与 sklearn 生态；无外部强化学习框架依赖

## 环境要求
- Python >= 3.13
- 依赖见 `pyproject.toml`

安装（可选本地开发模式）：
```bash
pip install -e .
```
或直接用源代码运行（无需安装）。

## 数据准备
默认使用仓库中 `dataset/ASHRAE/ashrae-energy-prediction-processed-building100.csv`。
若需更新/扩展数据：
1. 放置新的处理后 CSV 到同目录
2. 修改 `common/constants.py` 中 `ashrae_building100_processed_path` 指向你的文件

## 运行示例
### 训练（单步模式 + 折扣=0）
```bash
python main.py \
  --mode train \
  --train_iter 30000 \
  --single_step \
  --discount 0.0 \
  --corr_threshold 0.05 \
  --validate_steps 1000 \
  --warmup 200 \
  --epsilon 3000 \
  --hidden1 128 --hidden2 64 \
  --debug
```

### 测试
假设最新 run 目录为 `output/ASHRAE-runX`：
```bash
python main.py --mode test --resume output/ASHRAE-runX --single_step --debug
```

### 重要参数说明
| 参数 | 说明 |
|------|------|
| `--single_step` | 将问题视为 bandit（每条样本=1 step），Critic 拟合即时 -MSE，稳定性更好 |
| `--corr_threshold` | Pearson 相关性绝对值筛选阈值，过高可能删掉过多特征 |
| `--epsilon` | 噪声线性衰减步数，衰减至 0 后不再注入 OU 噪声 |
| `--ou_sigma` | 噪声强度，表格回归通常可设为 0~0.05 |
| `--discount` | 建议 0.0（避免跨样本回报耦合）；非单步长轨迹慎用 >0 值 |
| `--validate_steps` | 评估间隔步数；单步模式下可适当增大以减少开销 |

## 输出结果
`output/ASHRAE-runN/` 包含：
- `actor.pkl`, `critic.pkl`：网络权重
- `validate_reward.png`：评估曲线（左轴 reward，右轴 MSE）
- `validate_reward.mat`：矩阵数据，包含 `reward`, `mse`, `mae`

## 典型调参建议
1. 首先用 `--single_step --discount 0.0` 验证 MSE 是否随训练下降
2. 观察 `validate_reward.png` 中 MSE 曲线是否收敛；若震荡：
   - 降低 `ou_sigma` 或直接设为 0
   - 减小网络规模 (64/32)
   - 增大 `warmup` 到 500
3. 调整 `corr_threshold`，若特征过少导致欠拟合可降低阈值

## 目录结构简述
```
common/      工具、日志、常量、评估器与特征编码
env/         ASHRAE 专用表格环境 (无 Gym)
model/       DDPG 组件（Actor/Critic/Replay Buffer/Noise）
main.py      入口脚本 (仅 ASHRAE)
```

## 许可证
（根据你的项目实际需要补充）

