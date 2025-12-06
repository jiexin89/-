# -*- coding: utf-8 -*-
"""强化风险管理模块：多维度风险控制"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class EnhancedRiskManager:
    def __init__(self, paper_engine, ai_engine):
        self.paper_engine = paper_engine
        self.ai_engine = ai_engine
        
        # ========== 风险参数 ==========
        self.max_position_ratio = 0.8  # 单币种最大仓位占比
        self.max_total_risk = 0.2      # 总资金最大风险敞口
        self.max_daily_loss = 0.05     # 每日最大亏损 5%
        self.max_consecutive_loss = 3  # 最大连续亏损次数
        self.max_drawdown = 0.15       # 最大回撤 15%
        self.min_confidence = 0.5      # 最小置信度
        self.max_leverage = 1.0        # 最大杠杆倍数（实盘用）
        
        # ========== 临时变量 ==========
        self.daily_loss = 0.0
        self.last_day = datetime.now().date()
        self.starting_balance = paper_engine.balance if hasattr(paper_engine, 'balance') else 10000.0

    def assess_risk(self, 
                   symbol: str, 
                   final_signal: str, 
                   latest_price: float, 
                   confidence: float,
                   extreme_features: Dict[str, Any],
                   current_regime: int,
                   current_position: float = 0.0,
                   entry_price: Optional[float] = None) -> Dict[str, Any]:
        """
        评估当前交易的风险水平
        
        Args:
            symbol: 交易对
            final_signal: AI信号
            latest_price: 当前价格
            confidence: 综合置信度
            extreme_features: 极端特征
            current_regime: 市场状态
            current_position: 当前持仓
            entry_price: 开仓价格（如有）
        
        Returns:
            Dict: 风险评估结果
        """
        assessment = {
            "allow_trade": True,
            "risk_level": "low",  # "low", "medium", "high", "critical"
            "adjusted_size": 1.0,  # 调整后的仓位系数
            "reasons": [],
            "account_risk": 0.0,
            "market_risk": 0.0,
            "strategy_risk": 0.0,
            "total_risk": 0.0
        }
        
        # ========== 1. 账户风险评估 ==========
        account_risk = self._assess_account_risk()
        assessment["account_risk"] = account_risk
        
        # ========== 2. 市场风险评估 ==========
        market_risk = self._assess_market_risk(extreme_features, current_regime)
        assessment["market_risk"] = market_risk
        
        # ========== 3. 策略风险评估 ==========
        strategy_risk = self._assess_strategy_risk(symbol, final_signal, confidence, current_position)
        assessment["strategy_risk"] = strategy_risk
        
        # ========== 4. 计算总风险 ==========
        total_risk = (account_risk + market_risk + strategy_risk) / 3
        assessment["total_risk"] = total_risk
        
        # ========== 5. 风险等级划分 ==========
        if total_risk > 0.8:
            assessment["risk_level"] = "critical"
        elif total_risk > 0.6:
            assessment["risk_level"] = "high"
        elif total_risk > 0.4:
            assessment["risk_level"] = "medium"
        else:
            assessment["risk_level"] = "low"
        
        # ========== 6. 决策逻辑 ==========
        if total_risk >= 0.8:
            assessment["allow_trade"] = False
            assessment["reasons"].append("总风险过高，禁止交易")
        elif confidence < self.min_confidence:
            assessment["allow_trade"] = False
            assessment["reasons"].append(f"AI置信度{confidence:.2f}低于阈值{self.min_confidence}")
        elif account_risk > 0.7:
            assessment["allow_trade"] = False
            assessment["reasons"].append("账户风险过高，禁止交易")
        elif market_risk > 0.7:
            assessment["allow_trade"] = False
            assessment["reasons"].append("市场风险过高，禁止交易")
        elif strategy_risk > 0.7:
            assessment["allow_trade"] = False
            assessment["reasons"].append("策略风险过高，禁止交易")
        
        # ========== 7. 调整仓位大小 ==========
        if assessment["allow_trade"]:
            assessment["adjusted_size"] = max(0.1, min(1.0, 1.0 - total_risk))
        
        return assessment

    def _assess_account_risk(self) -> float:
        """评估账户层面风险"""
        current_balance = self.paper_engine.balance
        starting_balance = self.starting_balance
        
        # 计算当前回撤
        drawdown = (starting_balance - current_balance) / starting_balance if starting_balance > 0 else 0.0
        drawdown_risk = min(1.0, drawdown / self.max_drawdown)
        
        # 计算当前持仓风险（总仓位/总资产）
        total_position_value = 0.0
        for symbol_key in ["POL", "ARB", "ETH", "ACT", "BTC"]:  # MONITOR_SYMBOLS
            pos = self.paper_engine.get_position(f"{symbol_key}USDT")
            if pos != 0:
                # 获取当前价格（这里需要一个获取价格的函数，假设存在）
                # current_price = get_current_price(f"{symbol_key}USDT")
                # total_position_value += abs(pos) * current_price
                pass  # 暂时跳过，因为无法获取实时价格
        position_risk = min(1.0, total_position_value / current_balance) if current_balance > 0 else 0.0
        
        # 计算连续亏损
        # (需要 paper_engine 有连续亏损统计功能)
        consecutive_loss_risk = 0.0
        # for symbol_key in MONITOR_SYMBOLS:
        #     losses = self.paper_engine.consecutive_losses(f"{symbol_key}USDT")
        #     if losses >= self.max_consecutive_loss:
        #         consecutive_loss_risk = 1.0
        #         break
        
        return (drawdown_risk + position_risk + consecutive_loss_risk) / 3

    def _assess_market_risk(self, extreme_features: Dict[str, Any], current_regime: int) -> float:
        """评估市场层面风险"""
        risk_score = 0.0
        
        # 波动率压缩
        vol_comp = extreme_features.get("volatility_compression", 1.0)
        if vol_comp > 2.0:
            risk_score += 0.3
        
        # RSI背离
        rsi_div = extreme_features.get("rsi_divergence", 0)
        if abs(rsi_div) > 1:
            risk_score += 0.2
        
        # 巨量异常
        vol_spike = extreme_features.get("volume_spike", 1.0)
        if vol_spike > 3.0:
            risk_score += 0.3
        
        # 市场状态（震荡市风险更高）
        if current_regime == 0:  # 震荡市
            risk_score += 0.2
        
        return min(1.0, risk_score)

    def _assess_strategy_risk(self, symbol: str, final_signal: str, confidence: float, current_position: float) -> float:
        """评估策略层面风险"""
        risk_score = 0.0
        
        # 低置信度
        if confidence < 0.5:
            risk_score += 0.4
        
        # 逆势交易（当前持仓与信号方向相反）
        if (current_position > 0 and final_signal in ["卖出", "强烈卖出"]) or \
           (current_position < 0 and final_signal in ["买入", "强烈买入"]):
            risk_score += 0.3
        
        # 胜率过低
        win_rate = self.paper_engine.get_symbol_win_rate(f"{symbol}USDT")
        if win_rate < 0.3:
            risk_score += 0.3
        
        return min(1.0, risk_score)

    def update_daily_tracking(self):
        """更新每日跟踪（如亏损统计）"""
        today = datetime.now().date()
        if today != self.last_day:
            self.daily_loss = 0.0
            self.last_day = today