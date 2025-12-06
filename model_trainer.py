# model_trainer.py
import logging
import traceback

logger = logging.getLogger(__name__)

def pre_train_models(ai_engine, xgb_predictor, price_drop_predictor, lstm_predictor):
    """预训练所有模型，确保系统启动时模型已准备好"""
    logger.info("🔄 开始预训练所有模型...")
    
    try:
        # 1. 预训练XGBoost模型
        logger.info("🔄 正在预训练XGBoost模型...")
        xgb_predictor.train(ai_engine)
        logger.info("✅ XGBoost模型预训练完成")
        
        # 2. 预训练价格下跌预测模型
        logger.info("🔄 正在预训练价格下跌预测模型...")
        price_drop_predictor.train(ai_engine)
        logger.info("✅ 价格下跌预测模型预训练完成")
        
        # 3. 预训练LSTM模型
        logger.info("🔄 正在预训练LSTM模型...")
        historical_data = ai_engine.get_historical_data()
        if historical_data is not None and len(historical_data) > lstm_predictor.sequence_length + 10:
            lstm_predictor.train(historical_data)
            logger.info(f"✅ LSTM模型预训练完成，使用 {len(historical_data)} 条历史数据")
        else:
            logger.warning("⚠️ 历史数据不足，无法完成LSTM模型预训练")
        
        logger.info("✅ 所有模型预训练完成")
        return True
    except Exception as e:
        logger.error(f"❌ 模型预训练失败: {str(e)}")
        logger.error(f"❌ 错误详细信息: {traceback.format_exc()}")
        return False

def train_xgb_model(ai_engine, xgb_predictor):
    """单独训练XGBoost模型"""
    try:
        logger.info("🔄 正在训练XGBoost模型...")
        xgb_predictor.train(ai_engine)
        logger.info("✅ XGBoost模型训练完成")
        return True
    except Exception as e:
        logger.error(f"❌ XGBoost模型训练失败: {str(e)}")
        logger.error(f"❌ 错误详细信息: {traceback.format_exc()}")
        return False

# 类似地定义其他训练函数...