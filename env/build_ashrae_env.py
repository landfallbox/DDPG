"""
@Project     : DDPG 
@File        : build_ashrae_env.py
@IDE         : PyCharm 
@Author      : landfallbox
@Date        : 2025/9/25 星期四 14:07 
@Description :
"""

import json
import os

import numpy as np
import pandas as pd

from common.constants import ashrae_building100_processed_path, target_col, col_timestamp, default_seed
from common.feature_encoder import DeepForestEncoder
from common.logger import logger
from env.tabular_env import TabularPredictionEnv


def build_ashrae_envs(args):
    """构建 ASHRAE 数据训练/测试环境。
    处理流程:
    1. 读取 CSV (building100 processed)
    2. 按行顺序前 80% 为训练, 后 20% 为测试 (不打乱)
    3. 选择数值特征列 (包含 timestamp，若非数值则转为 epoch 秒) 并基于与目标的 Pearson 相关性进行筛选
    4. 使用 DeepForestEncoder (可回退到 RandomForest) 以 (X_train, y_train) 拟合并编码为特征表示
    5. y 归一化到 [-1,1] (基于训练集 min/max)
    6. 构建 TabularPredictionEnv (动作空间 [-1,1], reward = - (pred - target)^2)
    返回: train_env, test_env, 状态维度, 动作维度
    """
    # 生成随机建筑 ID
    # building_id = random.Random(args.seed if args.seed > 0 else 42).randint(0, 1448)
    # logger.info(f"选择建筑 ID {building_id} 进行训练 (随机种子 {args.seed})")
    #
    # process_ashrae(ashrae_data_dir, ashrae_data_processed_path, 100)

    csv_path = ashrae_building100_processed_path
    logger.info(f"读取 ASHRAE 数据: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("ASHRAE 数据为空")

    # 目标列，使用 meter_reading 作为预测目标
    if target_col not in df.columns:
        raise RuntimeError(f"数据缺少目标列 {target_col}")

    # 构建候选特征列（去除目标列和 timestamp，仅考虑数值列）
    candidate_cols = [c for c in df.columns if c != target_col and c != col_timestamp]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise RuntimeError("未找到可用的数值特征列用于相关性筛选")

    # 转换为 pandas float，用于后续相关性计算
    target_series = df[target_col].astype(float)

    # 计算每个数值列与目标的 Pearson 相关系数
    pearson_corr = {}
    for col in numeric_cols:
        series = df[col].astype(float)
        if series.nunique() <= 1:
            corr_val = 0.0  # 常数列无信息
        else:
            try:
                corr_val = target_series.corr(series)  # pandas 会自动忽略 NaN
            except Exception as e:
                logger.warning(f"计算列 {col} Pearson 相关失败: {e}")
                corr_val = 0.0
        if pd.isna(corr_val):
            corr_val = 0.0
        pearson_corr[col] = float(corr_val)

    # 将全部相关系数写入 JSON 文件
    corr_path = os.path.join(args.output, 'pearson_correlations.json')
    try:
        with open(corr_path, 'w', encoding='utf-8') as f:
            json.dump({k: round(v, 6) for k, v in pearson_corr.items()}, f, ensure_ascii=False, indent=2)
        logger.info(f"相关系数字典已保存: {corr_path}")
    except Exception as e:
        logger.warning(f"保存相关系数 JSON 失败: {e}")

    # 按阈值过滤
    corr_threshold = args.corr_threshold
    feature_cols = [c for c, v in pearson_corr.items() if abs(v) >= corr_threshold]

    # 若没有任何列达到阈值, 退化为选择绝对相关性最高的前 1 列，避免空特征
    if not feature_cols:
        # 取绝对值最大的一列
        top_col = max(pearson_corr.items(), key=lambda kv: abs(kv[1]))[0]
        feature_cols = [top_col]
        logger.warning(
            f"无特征达到阈值 {corr_threshold}, 回退保留相关性最高列: {top_col} (corr={pearson_corr[top_col]:.4f})")

    # 统计信息与日志
    sorted_selected = sorted([(c, pearson_corr[c]) for c in feature_cols], key=lambda x: -abs(x[1]))
    sorted_dropped = sorted([(c, pearson_corr[c]) for c in numeric_cols if c not in feature_cols],
                            key=lambda x: -abs(x[1]))
    selected_str = ', '.join([f"{c}:{v:.3f}" for c, v in sorted_selected])
    dropped_str = ', '.join([f"{c}:{v:.3f}" for c, v in sorted_dropped]) if sorted_dropped else '(无)'
    logger.info(f"Pearson 筛选: 阈值={corr_threshold}, 保留 {len(feature_cols)}/{len(numeric_cols)} 数值列")
    logger.info(f"保留特征({len(sorted_selected)}): {selected_str}")
    logger.info(f"被删除特征({len(sorted_dropped)}): {dropped_str}")

    # 按行顺序切分 (不打乱): 80% 训练 / 20% 测试
    n_total = len(df)
    n_train = int(n_total * 0.8)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df = df.iloc[n_train:].reset_index(drop=True)

    x_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[target_col].values.astype(float)
    x_test_raw = test_df[feature_cols].values
    y_test_raw = test_df[target_col].values.astype(float)

    # 编码器: 深度森林 (可选) / 随机森林 fallback
    encoder = DeepForestEncoder(add_original=True, random_state=args.seed if args.seed > 0 else default_seed)
    x_train_enc = encoder.fit_transform(x_train_raw, y_train_raw)
    x_test_enc = encoder.transform(x_test_raw)

    # y 归一化到 [-1,1] 使用训练集 min/max
    y_min, y_max = y_train_raw.min(), y_train_raw.max()
    if y_max - y_min < 1e-9:
        # 避免除零 (若全为常数, 直接设为 0)
        y_train_norm = np.zeros_like(y_train_raw, dtype=np.float32)
        y_test_norm = np.zeros_like(y_test_raw, dtype=np.float32)
    else:
        y_train_norm = ((y_train_raw - y_min) / (y_max - y_min) * 2.0 - 1.0).astype(np.float32)
        y_test_norm = ((y_test_raw - y_min) / (y_max - y_min) * 2.0 - 1.0).astype(np.float32)

    train_env = TabularPredictionEnv(x_train_enc, y_train_norm, shuffle=False,
                                     seed=args.seed if args.seed > 0 else None,
                                     sample_mode=(
                                         'single_step' if getattr(args, 'ashrae_single_step', False) else 'sequential'),
                                     max_episode_length=(1 if getattr(args, 'ashrae_single_step', False) else None))
    test_env = TabularPredictionEnv(x_test_enc, y_test_norm, shuffle=False, seed=args.seed if args.seed > 0 else None,
                                    sample_mode=(
                                        'single_step' if getattr(args, 'ashrae_single_step', False) else 'sequential'),
                                    max_episode_length=(1 if getattr(args, 'ashrae_single_step', False) else None))

    if getattr(args, 'ashrae_single_step', False):
        logger.info('ASHRAE 环境使用 single_step 模式: 每条样本独立 episode (bandit 化)')

    nb_states = train_env.observation_space.shape[0]
    nb_actions = train_env.action_space.shape[0]

    meta = {
        'n_total': n_total,
        'n_train': n_train,
        'n_test': n_total - n_train,
        'feature_raw_dim': x_train_raw.shape[1],
        'feature_enc_dim': nb_states,
    }
    logger.info(f"ASHRAE 数据概况: {meta}")
    return train_env, test_env, nb_states, nb_actions
