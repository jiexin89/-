# core/factor_bundler.py
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import joblib

logger = logging.getLogger(__name__)

class FactorBundle:
    def __init__(self):
        self.scaler = StandardScaler() # 或者使用 QuantileTransformer
        self.bundle = {}
        self.normalized_bundle = {}

    def add_factor(self, name, value, source='unknown'):
        """
        添加一个因子到 bundle。
        Args:
            name (str): 因子名称。
            value (float or int or bool): 因子原始值。
            source (str): 因子来源，如 'xgb', 'mtf', 'ta', 'orderbook', 'kline', 'rps', 'channel' 等。
        """
        self.bundle[name] = {'value': value, 'source': source}
        logger.debug(f"📊 FactorBundle: Added factor '{name}' from '{source}' with value {value}")

    def normalize(self):
        """
        对 bundle 中的所有因子进行标准化/归一化。
        """
        if not self.bundle:
            logger.warning("⚠️ FactorBundle is empty, cannot normalize.")
            return

        names = list(self.bundle.keys())
        values = np.array([[self.bundle[name]['value']] for name in names]) # reshape for scaler

        # Fit and transform
        normalized_values = self.scaler.fit_transform(values).flatten()

        # Update normalized_bundle
        self.normalized_bundle = {}
        for i, name in enumerate(names):
            self.normalized_bundle[name] = {
                'original_value': self.bundle[name]['value'],
                'normalized_value': normalized_values[i],
                'source': self.bundle[name]['source']
            }
        logger.info(f"✅ FactorBundle normalized {len(names)} factors.")

    def get_normalized_factors(self):
        """
        获取标准化后的因子字典。
        Returns:
            dict: {name: {'original_value': val, 'normalized_value': norm_val, 'source': src}}
        """
        return self.normalized_bundle

    def get_raw_factors(self):
        """
        获取原始因子字典。
        Returns:
            dict: {name: {'value': val, 'source': src}}
        """
        return self.bundle

    def save_scaler(self, filepath):
        """保存标准化器"""
        joblib.dump(self.scaler, filepath)
        logger.info(f"✅ Scaler saved to {filepath}")

    def load_scaler(self, filepath):
        """加载标准化器"""
        try:
            self.scaler = joblib.load(filepath)
            logger.info(f"✅ Scaler loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"⚠️ Scaler file not found: {filepath}, using default StandardScaler.")

# 全局实例（可选，方便在不同模块间共享）
# global_factor_bundler = FactorBundle()