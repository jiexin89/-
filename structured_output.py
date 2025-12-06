# -*- coding: utf-8 -*-
"""结构化决策输出：统一交易决策的返回格式"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import json

class StructuredDecisionOutput:
    def __init__(self, 
                 timestamp: datetime,
                 symbol: str,
                 current_price: float,
                 final_signal: str,
                 confidence: float,
                 explanation: str,
                 risk_assessment: Dict[str, Any],
                 entry_price: Optional[float] = None,
                 predicted_price: Optional[float] = None,
                 indicators: Optional[Dict] = None,
                 kline_alerts: Optional[List[str]] = None,
                 decision_features: Optional[Dict] = None):
        self.data = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "current_price": current_price,
            "entry_price": entry_price,
            "predicted_price": predicted_price,
            "final_signal": final_signal,
            "confidence": confidence,
            "explanation": explanation,
            "risk_assessment": risk_assessment,
            "indicators": indicators or {},
            "kline_alerts": kline_alerts or [],
            "decision_features": decision_features or {},
            "execution_status": "pending"  # "pending", "executed", "rejected"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """返回字典格式"""
        return self.data
    
    def to_json(self) -> str:
        """返回 JSON 格式"""
        return json.dumps(self.data, ensure_ascii=False, indent=2)
    
    def set_execution_status(self, status: str, execution_details: Optional[Dict] = None):
        """设置执行状态"""
        self.data["execution_status"] = status
        if execution_details:
            self.data.update(execution_details)
    
    def get_signal_for_trading(self) -> Optional[str]:
        """获取可执行的交易信号"""
        if not self.data["risk_assessment"].get("allow_trade", True):
            return None
        return self.data["final_signal"]
    
    def get_adjusted_size(self) -> float:
        """获取调整后的仓位大小"""
        return self.data["risk_assessment"].get("adjusted_size", 1.0)
    
    def print_readable(self):
        """打印可读格式"""
        print("=" * 60)
        print(f"🧠【AI决策输出】{self.data['symbol']} | {self.data['timestamp']}")
        print(f"💰 当前价格: {self.data['current_price']:.6f}")
        print(f"🎯 最终信号: {self.data['final_signal']}")
        print(f"🎯 置信度: {self.data['confidence']:.1%}")
        print(f"⚠️ 风险等级: {self.data['risk_assessment'].get('risk_level', 'unknown')}")
        print(f"📊 解释: \n{self.data['explanation']}")
        print("=" * 60)