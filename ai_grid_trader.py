import logging
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np

class AIGridTrader:
    def __init__(self, trader, symbol: str, ai_engine: 'AILearningEngine', logger: logging.Logger):
        self.trader = trader
        self.symbol = symbol
        self.ai_engine = ai_engine
        self.logger = logger

        # ========== 核心网格参数（可动态调整）==========
        self.grid_upper = None  # 网格上界（止盈/阻力）
        self.grid_lower = None  # 网格下界（止损/支撑）
        self.base_price_interval = 0.01  # 基础价格间隔（1%）
        self.safe_gap = 0.005            # 安全间隔（0.5%）
        self.max_multiplier = 3.0        # 最大风险倍数
        self.order_count = 5             # 默认订单数量
        self.grid_orders = {}            # {order_id: {"price", "size", "side", "status"}}
        self.ai_direction = "neutral"    # "bullish", "bearish", "neutral"
        self.current_price = None
        self.last_adjust_time = None     # 上次调整时间

    def update_grid_bounds(self, upper: float, lower: float, grid_count: int = 5):
        """更新网格边界和订单数量"""
        self.grid_upper = upper
        self.grid_lower = lower
        self.order_count = grid_count
        self.logger.debug(f"🔄【网格参数更新】{self.symbol} 新边界: {lower:.5f} - {upper:.5f}, 订单数: {grid_count}")

    def deploy_buy_grid(self):
        """部署做多网格（在支撑区挂买单）"""
        if self.grid_lower is None or self.grid_upper is None:
            self.logger.warning("⚠️【网格部署失败】网格边界未设置")
            return

        # ========== 取消旧订单（避免重复）==========
        self._cancel_all_orders()

        # ========== 计算动态订单 ==========
        prices = self._calculate_grid_prices(is_buy=True)
        total_balance = self.trader.balance
        base_size = (total_balance * 0.005) / self.current_price  # 基础仓位 0.5%
        total_size = base_size * self.max_multiplier  # 动态仓位（根据风险）

        # ========== 下单 ==========
        for i, price in enumerate(prices):
            # 越靠近支撑，仓位越大（金字塔加仓）
            size = total_size * (i + 1) / len(prices) * 0.8  # 80% 仓用于网格
            if size * price < 10:  # 最小交易额 $10
                continue
            try:
                order = self.trader.place_order(
                    symbol=self.symbol,
                    side="buy",
                    price=price,
                    size=size,
                    order_type="limit"
                )
                self.grid_orders[order["id"]] = {
                    "price": price,
                    "size": size,
                    "side": "buy",
                    "status": "open",
                    "created_at": datetime.now()
                }
                self.logger.info(f"✅【网格挂单】{self.symbol} 买单 @ {price:.5f}, 量: {size:.4f}")
            except Exception as e:
                self.logger.error(f"❌【网格挂单失败】{e}")

    def deploy_sell_grid(self):
        """部署做空网格（在阻力区挂卖单）"""
        if self.grid_lower is None or self.grid_upper is None:
            self.logger.warning("⚠️【网格部署失败】网格边界未设置")
            return

        # ========== 取消旧订单 ==========
        self._cancel_all_orders()

        # ========== 计算动态订单 ==========
        prices = self._calculate_grid_prices(is_buy=False)
        total_balance = self.trader.balance
        base_size = (total_balance * 0.005) / self.current_price
        total_size = base_size * self.max_multiplier

        for i, price in enumerate(prices):
            # 越靠近阻力，仓位越大
            size = total_size * (len(prices) - i) / len(prices) * 0.8
            if size * price < 10:
                continue
            try:
                order = self.trader.place_order(
                    symbol=self.symbol,
                    side="sell",
                    price=price,
                    size=size,

                )
                self.grid_orders[order["id"]] = {
                    "price": price,
                    "size": size,
                    "side": "sell",
                    "status": "open",
                    "created_at": datetime.now()
                }
                self.logger.info(f"✅【网格挂单】{self.symbol} 卖单 @ {price:.5f}, 量: {size:.4f}")
            except Exception as e:
                self.logger.error(f"❌【网格挂单失败】{e}")

    def _calculate_grid_prices(self, is_buy: bool) -> list:
        """计算网格价格"""
        if is_buy:
            # 买单：从 grid_lower 到 grid_upper
            prices = np.linspace(self.grid_lower, self.grid_upper, self.order_count)
        else:
            # 卖单：从 grid_upper 到 grid_lower
            prices = np.linspace(self.grid_upper, self.grid_lower, self.order_count)
        return prices.tolist()

    def _cancel_all_orders(self):
        """取消所有网格订单"""
        for order_id in list(self.grid_orders.keys()):
            try:
                self.trader.cancel_order(self.symbol, order_id)
                self.logger.debug(f"🗑️【取消网格订单】{order_id}")
            except Exception as e:
                self.logger.warning(f"⚠️【取消订单失败】{e}")
        self.grid_orders.clear()

    def adjust_grid_on_strong_signal(self, is_strong_bullish: bool, is_strong_bearish: bool):
        """
        【核心新增】根据强信号动态调整现有网格
        """
        if not self.grid_orders:
            self.logger.debug("🔍【网格调整】无活跃网格，跳过调整")
            return

        if self.last_adjust_time and (datetime.now() - self.last_adjust_time).total_seconds() < 300:
            self.logger.debug("⏳【网格调整】冷却中，跳过")
            return

        self.last_adjust_time = datetime.now()
        self.logger.info(f"🔄【网格动态调整】{self.symbol} 检测到强信号，正在优化...")

        current_position = self.trader.get_position(self.symbol)

        if is_strong_bullish:
            # ========== 强看涨：增强多头网格 ==========
            if current_position >= 0:
                # 如果已有仓位，向上移动网格上界，增加潜在盈利
                new_upper = self.grid_upper * 1.02  # 上移 2%
                self.grid_upper = new_upper
                self.deploy_buy_grid()  # 重建网格（含新边界）
                self.logger.info(f"📈【网格增强】多头网格上界上移至 {new_upper:.5f}")
            else:
                # 转为多头网格
                self.ai_direction = "bullish"
                self.deploy_buy_grid()

        elif is_strong_bearish:
            # ========== 强看跌：增强空头网格 ==========
            if current_position <= 0:
                new_lower = self.grid_lower * 0.98  # 下移 2%
                self.grid_lower = new_lower
                self.deploy_sell_grid()
                self.logger.info(f"📉【网格增强】空头网格下界下移至 {new_lower:.5f}")
            else:
                self.ai_direction = "bearish"
                self.deploy_sell_grid()

    def monitor_orders(self):
        """监控订单成交与止损"""
        for order_id, order in list(self.grid_orders.items()):
            if order["status"] != "open":
                continue
            try:
                status = self.trader.get_order_status(self.symbol, order_id)
                if status in ["filled", "partially_filled"]:
                    self.grid_orders[order_id]["status"] = status
                    self.logger.info(f"💰【网格成交】{self.symbol} {order['side']} @ {order['price']:.5f}")
                elif status == "canceled":
                    del self.grid_orders[order_id]
            except Exception as e:
                self.logger.warning(f"⚠️【订单查询失败】{e}")

    def update_pnl(self, current_price: float):
        """更新盈亏"""
        self.current_price = current_price
        # 可扩展：计算未实现盈亏