# real_trader.py
import logging
from typing import Optional
from binance import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger("TradingBot")

class RealTrader:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.fee_rate = 0.0004  # Binance futures 手续费 0.04%
        self.testnet = testnet
        self.balance = self.get_account_balance()
        logger.info(f"✅ {'Testnet' if testnet else 'Mainnet'} RealTrader 初始化成功 | 余额: {self.balance:.2f} USDT")

    def get_account_balance(self) -> float:
        """获取账户 USDT 余额"""
        try:
            account = self.client.futures_account()
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['walletBalance'])
        except Exception as e:
            logger.error(f"⚠️ 获取账户余额失败: {e}")
        return 10000.0  # fallback

    def get_position(self, symbol: str) -> float:
        """获取当前持仓（多头为正，空头为负）"""
        try:
            pos = self.client.futures_position_information(symbol=symbol)
            for p in pos:
                if p['symbol'] == symbol:
                    return float(p['positionAmt'])
            return 0.0
        except Exception as e:
            logger.error(f"⚠️ 获取 {symbol} 持仓失败: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, price: float, size: float,
                    take_profit: Optional[float] = None, stop_loss: Optional[float] = None):
        """执行实盘交易（支持限价单 + 止盈止损）"""
        if size <= 0:
            logger.warning("⚠️ 订单大小无效，跳过交易")
            return

        # ========== 1. 精度校验 ==========
        try:
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if not symbol_info:
                raise ValueError(f"❌ 未找到交易对 {symbol} 的精度信息")

            price_precision = symbol_info['pricePrecision']
            qty_precision = symbol_info['quantityPrecision']
            price = round(price, price_precision)
            size = round(size, qty_precision)
        except Exception as e:
            logger.error(f"⚠️ 精度校验失败: {e}")
            return

        # ========== 2. 下单 ==========
        try:
            # 开仓限价单
            order = self.client.futures_create_order(
                symbol=symbol,
                side="BUY" if side == "buy" else "SELL",
                positionSide="BOTH",
                type="LIMIT",
                timeInForce="GTC",
                quantity=size,
                price=price
            )
            logger.info(f"✅ 实盘下单成功 | {side.upper()} {symbol} | 价格: {price} | 数量: {size}")

            # 下止盈止损单（OCO）
            if take_profit and stop_loss:
                self.client.futures_create_order(
                    symbol=symbol,
                    side="SELL" if side == "buy" else "BUY",
                    positionSide="BOTH",
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=take_profit,
                    closePosition=True
                )
                self.client.futures_create_order(
                    symbol=symbol,
                    side="SELL" if side == "buy" else "BUY",
                    positionSide="BOTH",
                    type="STOP_MARKET",
                    stopPrice=stop_loss,
                    closePosition=True
                )
                logger.info(f"🎯 止盈止损已设置 | TP: {take_profit} | SL: {stop_loss}")

            # 更新余额
            self.balance = self.get_account_balance()

        except BinanceAPIException as e:
            logger.error(f"❌ Binance API 错误: {e.status_code} - {e.message}")
        except Exception as e:
            logger.error(f"💥 实盘下单失败: {e}")