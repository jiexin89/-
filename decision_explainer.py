# -*- coding: utf-8 -*-
"""决策解释器：将AI的黑盒决策转化为人类可读的解释"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DecisionExplainer:
    def __init__(self):
        self.explanation_templates = {
            "strong_buy": [
                "🟢 强看涨信号：K线形态与多周期共振，AI判定为强烈买入！",
                "🚀 AI预测未来价格将上涨，基于趋势和形态双重确认。",
                "📈 多周期指标共振向上，市场情绪乐观，建议开多。"
            ],
            "buy": [
                "🟡 看涨信号：AI预测价格上涨，但强度中等。",
                "💰 技术指标显示潜在上涨机会，可适量建仓。",
                "✅ AI投票显示轻微上涨倾向，建议关注支撑位入场。"
            ],
            "strong_sell": [
                "🔴 强看跌信号：K线形态与多周期共振，AI判定为强烈卖出！",
                "🚨 AI预测未来价格将下跌，基于趋势和形态双重确认。",
                "📉 多周期指标共振向下，市场情绪悲观，建议开空。"
            ],
            "sell": [
                "🟠 看跌信号：AI预测价格下跌，但强度中等。",
                "💸 技术指标显示潜在下跌风险，可适量减仓。",
                "❌ AI投票显示轻微下跌倾向，建议关注阻力位出场。"
            ],
            "hold": [
                "⚠️ 观望信号：多空力量均衡，趋势不明朗。",
                "⏸️ AI预测方向不明确，建议等待更清晰信号。",
                "💤 市场处于震荡状态，暂时无明确交易机会。"
            ]
        }

    def explain_decision(self, 
                        final_signal: str, 
                        decision_features: Dict[str, Any],
                        extreme_features: Dict[str, Any],
                        current_analyses: Dict[str, Any],
                        kline_alerts: List[str],
                        confidence: float,
                        current_regime: int) -> str:
        """
        解释AI的最终决策
        
        Args:
            final_signal: AI最终信号 ("买入"/"卖出"/"观望")
            decision_features: 决策特征（XGB概率、Timing概率、MTF预测等）
            extreme_features: 极端特征
            current_analyses: 当前多周期分析
            kline_alerts: K线形态警报
            confidence: 综合置信度
            current_regime: 当前市场状态（0=震荡，1=趋势）
        
        Returns:
            str: 决策解释文本
        """
        explanation_parts = []
        
        # ========== 1. 信号来源解释 ==========
        source_reasons = []
        
        # XGBoost 贡献
        xgb_prob = decision_features.get("xgb_proba", 0.5)
        if xgb_prob > 0.7:
            source_reasons.append(f"XGBoost预测上涨概率高达{xgb_prob*100:.0f}%")
        elif xgb_prob < 0.3:
            source_reasons.append(f"XGBoost预测下跌概率高达{(1-xgb_prob)*100:.0f}%")
        
        # CNN 贡献
        cnn_prob = decision_features.get("cnn_proba", 0.5)
        if cnn_prob > 0.7:
            source_reasons.append(f"CNN识别到看涨形态，置信度{cnn_prob*100:.0f}%")
        elif cnn_prob < 0.3:
            source_reasons.append(f"CNN识别到看跌形态，置信度{(1-cnn_prob)*100:.0f}%")
        
        # MTF 贡献
        mtf_pred = decision_features.get("mtf_predicted_price", 0.0)
        latest_price = decision_features.get("latest_price", 0.0)
        if mtf_pred > 0 and latest_price > 0:
            mtf_change = (mtf_pred - latest_price) / latest_price
            if mtf_change > 0.005:
                source_reasons.append(f"MTF预测未来上涨{mtf_change*100:.1f}%")
            elif mtf_change < -0.005:
                source_reasons.append(f"MTF预测未来下跌{-mtf_change*100:.1f}%")
        
        if source_reasons:
            explanation_parts.append("📊【信号来源】" + "，".join(source_reasons))
        
        # ========== 2. K线形态解释 ==========
        if kline_alerts:
            bullish_alerts = [a for a in kline_alerts if any(k in a for k in ["启明星", "红三兵", "锤头线", "看涨"])]
            bearish_alerts = [a for a in kline_alerts if any(k in a for k in ["黄昏星", "三只乌鸦", "射击之星", "看跌"])]
            
            if bullish_alerts and final_signal in ["买入", "强烈买入"]:
                explanation_parts.append(f"📈【K线形态】检测到看涨形态: {', '.join(bullish_alerts[:2])}")
            elif bearish_alerts and final_signal in ["卖出", "强烈卖出"]:
                explanation_parts.append(f"📉【K线形态】检测到看跌形态: {', '.join(bearish_alerts[:2])}")
        
        # ========== 3. 极端事件解释 ==========
        extreme_events = []
        if extreme_features.get("volatility_compression", 0) > 2.0:
            extreme_events.append("市场波动率压缩，即将爆发")
        if extreme_features.get("rsi_divergence", 0) != 0:
            divergence_type = "顶背离" if extreme_features.get("rsi_divergence") > 0 else "底背离"
            extreme_events.append(f"检测到{divergence_type}，趋势可能反转")
        if extreme_features.get("volume_spike", 1.0) > 2.0:
            extreme_events.append("成交量激增，资金活跃")
        
        if extreme_events:
            explanation_parts.append("🚨【极端事件】" + "，".join(extreme_events))
        
        # ========== 4. 市场状态解释 ==========
        regime_names = {0: "震荡市", 1: "趋势市"}
        regime_desc = regime_names.get(current_regime, "未知市场")
        explanation_parts.append(f"🌐【市场状态】当前为{regime_desc}，AI模型已针对性调整")
        
        # ========== 5. 置信度解释 ==========
        if confidence > 0.8:
            explanation_parts.append("🎯【置信度】极高，信号可靠性强")
        elif confidence > 0.6:
            explanation_parts.append("✅【置信度】较高，可谨慎执行")
        elif confidence > 0.4:
            explanation_parts.append("⚠️【置信度】中等，建议观望")
        else:
            explanation_parts.append("❌【置信度】较低，建议忽略信号")
        
        # ========== 6. 综合解释模板 ==========
        template_key = final_signal.lower().replace(" ", "_").replace("强烈", "strong_")
        if template_key in ["买入", "卖出", "观望"]:
            template_key = "strong_" + template_key if "强烈" in final_signal else template_key
        template = self.explanation_templates.get(template_key, self.explanation_templates["hold"])
        
        final_explanation = "\n".join(explanation_parts) + f"\n💡【AI建议】{template[0]}"
        
        return final_explanation