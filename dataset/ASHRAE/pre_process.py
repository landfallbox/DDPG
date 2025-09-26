"""
@Project     : DDPG 
@File        : pre_process.py
@IDE         : PyCharm 
@Author      : landfallbox
@Date        : 2025/9/23 星期二 15:37 
@Description : 
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar

from common.constants import ashrae_data_dir, ashrae_data_processed_path
from common.logger import logger


def safe_interpolate(group, limit=48):
    """安全插值函数，处理全空分组"""
    if group.isnull().all():
        # 如果整个分组都是空值，使用前向+后向填充
        return group.ffill().bfill()
    try:
        return group.interpolate(method='linear', limit=limit, limit_direction='both')
    except ValueError:
        # 线性插值失败时使用最近邻插值
        return group.interpolate(method='nearest', limit=limit, limit_direction='both')


def fill_weather_missing_by_nearest_avg(series):
    """向量化填充天气数据缺失值

    对每个缺失位置，寻找最近的前一个与后一个非缺失值：
    若两者都存在，取两者平均；若仅一侧存在，取该侧值；若两侧均不存在，保持缺失。

    Parameters:
    series : pd.Series
        需要填充的时间序列数据。

    Returns:
    pd.Series
        填充后的时间序列数据。
    """
    values = series.to_numpy().astype(float)  # 确保可放 NaN
    mask = np.isnan(values)
    if not mask.any():
        return series  # 无缺失
    n = len(values)
    # 若全部缺失，直接返回（与原逻辑一致）
    if mask.all():
        return series

    indices = np.arange(n)

    # 计算前一个非缺失值索引（prev_idx）
    # 使用最大累积传播最后出现的有效索引
    valid_idx = np.where(~mask, indices, -1)
    prev_idx = np.maximum.accumulate(valid_idx)

    # 计算后一个非缺失值索引（next_idx）——使用从右到左单次线性扫描
    next_idx = np.empty(n, dtype=int)
    last = -1
    for i in range(n - 1, -1, -1):
        if not mask[i]:
            last = i
        next_idx[i] = last

    # 需要填充的位置
    miss_positions = np.where(mask)[0]
    fill_vals = values.copy()

    pidx = prev_idx[miss_positions]
    nidx = next_idx[miss_positions]

    pvals = np.where(pidx != -1, values[pidx], np.nan)
    nvals = np.where(nidx != -1, values[nidx], np.nan)

    both = ~np.isnan(pvals) & ~np.isnan(nvals)
    only_prev = ~np.isnan(pvals) & np.isnan(nvals)
    only_next = np.isnan(pvals) & ~np.isnan(nvals)

    new_vals = np.empty(len(miss_positions))
    new_vals[both] = (pvals[both] + nvals[both]) / 2
    new_vals[only_prev] = pvals[only_prev]
    new_vals[only_next] = nvals[only_next]
    # 两侧都 NaN 的保持 NaN
    both_nan = np.isnan(pvals) & np.isnan(nvals)
    if both_nan.any():
        new_vals[both_nan] = np.nan

    fill_vals[miss_positions] = new_vals

    return pd.Series(fill_vals, index=series.index)


def process_ashrae(data_dir, output_path=ashrae_data_processed_path, building_id=None):
    """完整数据处理流水线

    Parameters:
    data_dir : str or Path
        原始 ASHRAE 数据所在目录。
    output_path : str, Path, default ashrae_data_processed_path
        输出文件路径（若传入默认路径且指定 building_id，则自动在文件名中追加 -building{building_id}）
    building_id : int or None, default None
        若指定，仅处理该建筑的数据。
    """
    try:
        # 统一 Path 处理 & 记录默认基准路径
        default_output_path = Path(ashrae_data_processed_path)
        output_path = Path(output_path)  # 用户即便传入字符串也统一转换

        # ========== 数据加载 ==========
        logger.info("开始加载原始数据...")
        data_path = Path(data_dir)

        # 加载训练数据
        train_df = pd.read_csv(
            data_path / "train.csv",
            parse_dates=["timestamp"],
            dtype={'meter': 'category', 'building_id': 'uint16'}
        )
        # 预先初始化 bid 以避免静态分析器未定义警告
        bid = None

        # 按需过滤建筑
        if building_id is not None:
            logger.info(f"仅处理指定建筑 building_id={building_id} 数据…")
            # 确保类型兼容
            try:
                bid = np.uint16(building_id)
            except Exception:
                raise ValueError(f"building_id={building_id} 无法转换为 uint16")

            train_df = train_df[train_df['building_id'] == bid]
            if train_df.empty:
                raise ValueError(f"未在数据集中找到 building_id={building_id} 的记录")
            # 如果用户未自定义输出文件名（仍是默认路径），自动追加建筑标识
            if output_path == default_output_path:
                output_path = output_path.with_name(f"{output_path.stem}-building{building_id}.csv")
                logger.info(f"检测到默认输出路径，已自动改为按建筑输出: {output_path}")

        # 加载建筑元数据
        building_df = pd.read_csv(
            data_path / "building_metadata.csv",
            dtype={'site_id': 'uint8', 'primary_use': 'category'}
        )

        # 若指定 building_id，提前裁剪 building_df 以减少 merge 体积
        if building_id is not None:
            building_df = building_df[building_df['building_id'] == bid]

        # 加载天气数据
        weather_df = pd.read_csv(
            data_path / "weather_train.csv",
            parse_dates=["timestamp"],
            dtype={'site_id': 'uint8'}
        )

        # ========== 数据合并 ==========
        logger.info("开始数据合并...")
        merged = train_df.merge(
            building_df,
            on="building_id",
            how="left",
            validate="m:1"  # 验证一对多关系
        ).merge(
            weather_df,
            on=["site_id", "timestamp"],
            how="left",
            validate="m:1"
        )

        # ========== 时间特征工程 ==========
        logger.info("生成时间特征...")
        timestamp = pd.to_datetime(merged['timestamp'])
        merged['hour'] = timestamp.dt.hour.astype('uint8')
        merged['day_of_week'] = timestamp.dt.weekday.astype('uint8')  # 0=周一
        merged['month'] = timestamp.dt.month.astype('uint8')
        merged['day_of_month'] = timestamp.dt.day.astype('uint8')
        merged['day_of_year'] = timestamp.dt.dayofyear.astype('uint16')
        merged['week_of_year'] = timestamp.dt.isocalendar().week.astype('uint8')
        merged['is_holiday'] = timestamp.isin(
            USFederalHolidayCalendar().holidays(start=timestamp.min(), end=timestamp.max())).astype(
            'uint8')  # 节假日特征（美国联邦假日）

        # ========== 数据清洗 ==========
        logger.info("数据清洗中...")
        # 移除无效电表读数
        initial_count = len(merged)
        merged = merged[merged['meter_reading'] >= 0]
        logger.info(f"移除无效读数记录：{initial_count - len(merged)} 条")

        # ========== 天气数据处理 ==========
        logger.info("处理天气数据...")
        weather_cols = [
            'air_temperature', 'dew_temperature',
            'precip_depth_1_hr', 'sea_level_pressure', 'wind_speed'
        ]
        # 步骤：分组，前后最近有值取平均填充
        for col in weather_cols:
            merged[col] = merged.groupby("site_id")[col].transform(fill_weather_missing_by_nearest_avg)
        # 处理云覆盖率
        merged['cloud_coverage'] = merged.groupby("site_id")['cloud_coverage'].transform(
            lambda x: x.fillna(x.median()) if x.notnull().any() else x.fillna(0)
        ).fillna(0).astype('uint8')

        # ========== 建筑特征处理 ==========
        logger.info("处理建筑特征...")
        # 建筑年龄处理
        merged['age'] = (2025 - merged['year_built']).clip(lower=0)
        merged['age'] = merged['age'].fillna(merged['age'].median()).astype('uint16')

        # 楼层数处理
        merged['floor_count'] = merged['floor_count'].fillna(0).astype('uint8')

        # ========== 特征编码 ==========
        logger.info("编码分类特征...")
        le = LabelEncoder()
        merged['primary_use_code'] = le.fit_transform(merged['primary_use']).astype('uint8')

        # 保存建筑类型映射关系（可选）
        pd.DataFrame({
            'primary_use': le.classes_,
            'code': le.transform(le.classes_)
        }).to_csv('primary_use_mapping.csv', index=False)
        logger.info("建筑类型映射表已保存至 primary_use_mapping.csv")

        # ========== 数据排序 ==========
        logger.info("按建筑和时间排序...")
        merged = merged.sort_values(['building_id', 'timestamp']).reset_index(drop=True)

        # ========== 最终特征选择 ==========
        final_cols = [
            'building_id', 'timestamp', 'meter',
            'meter_reading', 'primary_use_code',
            'hour', 'day_of_week', 'month',
            'day_of_month', 'day_of_year', 'week_of_year', 'is_holiday',
            'air_temperature', 'dew_temperature',
            'wind_speed', 'square_feet', 'age'
        ]

        # ========== 数据验证 ==========
        logger.info("执行最终数据验证...")
        # 检查空值
        assert merged[final_cols].isnull().sum().sum() == 0, "存在未处理的缺失值"
        # 检查新增时间特征无缺失
        for col in ['day_of_month', 'day_of_year', 'week_of_year', 'is_holiday']:
            assert merged[col].isnull().sum() == 0, f"{col} 存在缺失值"
        # 检查电表读数有效性
        assert (merged['meter_reading'] < 0).sum() == 0, "存在无效电表读数"
        # 验证映射关系在当前数据子集内唯一
        unique_pairs = merged[['primary_use', 'primary_use_code']].drop_duplicates()
        assert unique_pairs['primary_use'].nunique() == len(unique_pairs), "映射关系不唯一！"

        # ========== 保存结果 ==========
        logger.info(f"保存处理结果到 {output_path}")
        merged[final_cols].to_csv(output_path, index=False)

        logger.info(f"处理完成！最终数据维度：{merged.shape}")
        return merged[final_cols]

    except Exception as e:
        logger.error(f"处理过程中发生错误：{str(e)}")
        raise


if __name__ == "__main__":
    process_ashrae(ashrae_data_dir, ashrae_data_processed_path, 100)
