# rl_decision_engine.py
import gym
from gym import spaces
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

logger = logging.getLogger("RLDecisionEngine")
logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

class TradingEnv(gym.Env):
    """自定义交易环境，用于训练PPO策略"""
    def __init__(self, history_file: Path):
        super().__init__()
        self.history_file = history_file
        self.load_history()
        self.current_step = 0
        # 观察空间：10个特征（与 decision_features 对齐）
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(10,), dtype=np.float32
        )
        # 动作空间：0=持有, 1=买入, 2=卖出
        self.action_space = spaces.Discrete(3)
        self.action_map = {0: "hold", 1: "buy", 2: "sell"}

    def load_history(self):
        if not self.history_file.exists():
            self.history = []
            return
        with open(self.history_file, "r", encoding="utf-8") as f:
            self.history = json.load(f)
        # 只保留已完成且有 decision_features 的记录
        self.history = [
            r for r in self.history
            if r.get("status") == "completed"
            and r.get("decision_features")
            and r.get("win") is not None
        ]
        logger.info(f"✅ 加载 {len(self.history)} 条有效历史记录用于RL训练")

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        if self.current_step >= len(self.history):
            return self._get_obs(), 0, True, {}
        record = self.history[self.current_step]
        win = record["win"]
        trade_action = record["trade_action"]
        # 动作映射
        action_str = self.action_map[action]
        reward = 0
        if action_str == "buy" and win == 1:
            reward = 1.0
        elif action_str == "sell" and win == 0:
            reward = 1.0
        elif action_str != "hold":
            reward = -0.5  # 错误操作惩罚
        self.current_step += 1
        done = self.current_step >= len(self.history)
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        if self.current_step >= len(self.history):
            return np.zeros(10, dtype=np.float32)
        record = self.history[self.current_step]
        df = record["decision_features"]
        # 特征顺序必须与训练时一致
        obs = np.array([
            df.get("xgb_proba", 0.5) * 2 - 1,  # 映射到 [-1, 1]
            df.get("buy_timing_proba", 0.5) * 2 - 1,
            df.get("sell_timing_proba", 0.5) * 2 - 1,
            (df.get("mtf_predicted_price", 0) / record.get("entry_price", 1) - 1) * 10,  # 预测收益率*10
            df.get("orderbook_imbalance", 0.0),
            df.get("sentiment_score", 0.0),
            df.get("crash_proba", 0.0) * 2 - 1,
            0.0,  # 预留
            0.0,  # 预留
            0.0,  # 预留
        ], dtype=np.float32)
        # 裁剪到 [-2, 2]
        return np.clip(obs, -2.0, 2.0)

class RLDecisionEngine:
    def __init__(self, symbol_name: str = "default"):
        self.symbol_name = symbol_name
        self.history_file = DATA_DIR / "ai_signal_history.json"
        self.model_path = MODEL_DIR / f"ppo_model_{symbol_name}.zip"
        self.model = None
        self.load_model()

    def load_model(self):
        if self.model_path.exists():
            try:
                self.model = PPO.load(self.model_path)
                logger.info(f"✅ 加载PPO模型: {self.model_path}")
            except Exception as e:
                logger.warning(f"❌ 加载PPO模型失败: {e}")

    def train(self, timesteps: int = 10000):
        env = TradingEnv(self.history_file)
        if len(env.history) < 100:
            logger.warning("⚠️ RL训练数据不足（<100），跳过训练")
            return False
        # 检查环境
        try:
            check_env(env)
        except Exception as e:
            logger.error(f"❌ 环境检查失败: {e}")
            return False
        # 训练模型
        self.model = PPO("MlpPolicy", env, verbose=0, n_steps=128, batch_size=64)
        self.model.learn(total_timesteps=timesteps)
        # 保存模型
        self.model.save(self.model_path)
        logger.info(f"✅ PPO模型已保存至 {self.model_path}")
        return True

    def predict_action(self, decision_features: dict, entry_price: float) -> int:
        """
        预测动作
        Returns:
            int: 0=持有, 1=买入, 2=卖出
        """
        if self.model is None:
            return 0  # 默认持有
        # 构建观测
        obs = np.array([
            decision_features.get("xgb_proba", 0.5) * 2 - 1,
            decision_features.get("buy_timing_proba", 0.5) * 2 - 1,
            decision_features.get("sell_timing_proba", 0.5) * 2 - 1,
            (decision_features.get("mtf_predicted_price", entry_price) / entry_price - 1) * 10,
            decision_features.get("orderbook_imbalance", 0.0),
            decision_features.get("sentiment_score", 0.0),
            decision_features.get("crash_proba", 0.0) * 2 - 1,
            0.0, 0.0, 0.0
        ], dtype=np.float32)
        obs = np.clip(obs, -2.0, 2.0).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def get_adaptive_weights(self) -> dict:
        """
        从RL策略中提取权重（简化版：通过模拟动作频率反推）
        """
        if self.model is None:
            return {}
        # 模拟1000次动作
        env = TradingEnv(self.history_file)
        if len(env.history) == 0:
            return {}
        buy_count, sell_count, hold_count = 0, 0, 0
        for _ in range(min(1000, len(env.history))):
            action = self.predict_action(
                env.history[_]["decision_features"],
                env.history[_]["entry_price"]
            )
            if action == 1:
                buy_count += 1
            elif action == 2:
                sell_count += 1
            else:
                hold_count += 1
        total = buy_count + sell_count + hold_count
        if total == 0:
            return {}
        # 假设权重与动作频率正相关
        return {
            "xgb_proba": buy_count / total,
            "timing_predictor": (buy_count + sell_count) / total,
            "mtf_predictor": (buy_count + sell_count) / total,
            "orderbook": (buy_count + sell_count) / total,
            "sentiment": (buy_count + sell_count) / total,
            "crash_boom": sell_count / total,
        }