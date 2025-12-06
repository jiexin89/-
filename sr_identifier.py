# sr_identifier.py
import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks # 需要安装 scipy: pip install scipy

logger = logging.getLogger(__name__)

def identify_support_resistance(df: pd.DataFrame, window=20, min_distance=5, strength_threshold=2) -> tuple[list[float], list[float]]:
    """
    基于价格高低点识别支撑和阻力位。

    Args:
        df: 包含 'high', 'low', 'close' 列的 DataFrame。
        window: 用于查找峰值/谷值的滑动窗口大小。
        min_distance: 峰值/谷值之间的最小索引距离。
        strength_threshold: 一个价位需要被触及多少次才被视为有效支撑/阻力。

    Returns:
        tuple: (support_levels, resistance_levels)
    """
    if df is None or len(df) < window:
        logger.debug("📊 K线数据不足，无法识别 S/R 位")
        return [], []

    highs = df['high'].values
    lows = df['low'].values
    # closes = df['close'].values # 暂时未用到

    # 查找潜在的峰值（阻力）和谷值（支撑）
    # prominence 控制了峰/谷的显著性，可以根据需要调整
    resistance_indices, _ = find_peaks(highs, distance=min_distance, prominence=np.std(highs) * 0.15)
    support_indices, _ = find_peaks(-lows, distance=min_distance, prominence=np.std(lows) * 0.15) # 对 lows 取负号

    # 筛选峰值和谷值，只保留最近的 window 个数据点内的
    recent_highs_indices = [i for i in resistance_indices if i >= len(highs) - window]
    recent_lows_indices = [i for i in support_indices if i >= len(lows) - window]

    resistance_levels = [highs[i] for i in recent_highs_indices]
    support_levels = [lows[i] for i in recent_lows_indices]

    # 确保价位唯一且排序 (从高到低)
    support_levels = sorted(list(set(support_levels)), reverse=True)
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)

    logger.debug(f"📊 识别到支撑位: {support_levels[:5]}") # 只打印前5个
    logger.debug(f"📊 识别到阻力位: {resistance_levels[:5]}") # 只打印前5个

    return support_levels, resistance_levels

def find_nearest_sr_level(price: float, levels: list[float], side: str, tolerance_percent: float = 0.005) -> float | None:
    """
    在给定的 S/R 水平中，找到最接近当前价格且有利于平仓的价位。

    Args:
        price: 当前价格。
        levels: S/R 水平列表 (例如，支撑位列表或阻力位列表)。
        side: 平仓方向 ("buy" for平空, "sell" for平多)。
        tolerance_percent: 价格距离 S/R 位的容忍度百分比 (例如 0.005 表示 0.5%)。

    Returns:
        float | None: 找到的最优价格，如果没找到则返回 None。
    """
    if not levels:
        return None

    # 确定目标方向
    # 平多 (sell): 寻找 *高于* 当前价的 *阻力位*
    # 平空 (buy): 寻找 *低于* 当前价的 *支撑位*
    if side == "sell": # 平多，寻找阻力位
        target_levels = [level for level in levels if level > price and (level - price) / price <= tolerance_percent]
        if target_levels:
            return min(target_levels) # 返回最接近当前价的 *更高* 的阻力位
    elif side == "buy": # 平空，寻找支撑位
        target_levels = [level for level in levels if level < price and (price - level) / price <= tolerance_percent]
        if target_levels:
            return max(target_levels) # 返回最接近当前价的 *更低* 的支撑位
    else:
        logger.warning(f"⚠️ 未知平仓方向: {side}")
        return None

    logger.debug(f"🔍 未找到符合 {side} 单条件的 S/R 位 (当前价: {price}, 容忍度: {tolerance_percent*100:.2f}%)")
    return None

# 如果需要单独测试该模块
# if __name__ == "__main__":
#     # 示例：创建一个模拟的 df
#     np.random.seed(42)
#     close = 100 + np.cumsum(np.random.randn(100) * 0.1)
#     high = close + np.abs(np.random.randn(100) * 0.2)
#     low = close - np.abs(np.random.randn(100) * 0.2)
#     df_test = pd.DataFrame({'close': close, 'high': high, 'low': low})
#     supports, resistances = identify_support_resistance(df_test)
#     print(f"Supports: {supports}")
#     print(f"Resistances: {resistances}")
#     print(find_nearest_sr_level(100.5, resistances, "sell"))
#     print(find_nearest_sr_level(99.5, supports, "buy"))
