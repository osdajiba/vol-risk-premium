"""
交易策略模块 - 实现波动率期限结构交易和ETF对冲策略
"""

import numpy as np
import pandas as pd
import config

def term_structure_strategy(df):
    """波动率期限结构交易策略
    
    基于VIX期货期限结构形态捕捉波动率风险溢价。
    当期限结构陡峭正向且趋势增强时，做多近月VIX期货；
    当期限结构陡峭反向且趋势增强时，做空近月VIX期货。
    
    Args:
        df: 包含市场数据的DataFrame
        
    Returns:
        DataFrame: 添加了策略信号和收益的DataFrame
    """
    # 初始化信号序列
    df['ts_signal'] = 0
    
    # 使用向量化操作生成交易信号
    # 陡峭正向期限结构，做多VIX期货
    long_condition = (df['term_structure'] < config.TS_LOW_THRESHOLD) & (df['vix_change'] > 1)
    
    # 陡峭反向期限结构，做空VIX期货
    short_condition = (df['term_structure'] > config.TS_HIGH_THRESHOLD) & (df['vix_change'] < -1)
    
    # 应用信号
    df.loc[long_condition, 'ts_signal'] = 1
    df.loc[short_condition, 'ts_signal'] = -1
    
    # 填充信号，保持持仓直到出现反向信号
    df['ts_signal'] = df['ts_signal'].replace(to_replace=0, method='ffill')
    
    # 计算波动率目标仓位调整
    # 使用20天滚动窗口计算年化波动率
    df['vix_futures_vol'] = df['vix_futures_f1'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    # 应用波动率目标方法调整仓位大小
    df['ts_position_size'] = config.TARGET_VOL / df['vix_futures_vol'].clip(lower=0.01)
    
    # 限制杠杆在规定范围内
    df['ts_position_size'] = df['ts_position_size'].clip(lower=0.5, upper=config.MAX_LEVERAGE)
    
    # 计算最终仓位
    df['ts_position'] = df['ts_signal'] * df['ts_position_size']
    
    # 计算策略收益（不考虑交易成本）
    df['ts_returns'] = df['ts_position'].shift(1) * df['vix_futures_f1'].pct_change()
    
    # 计算交易成本
    calculate_trading_costs(df, 'ts_signal', 'ts_returns', config.FUTURES_COST + config.SLIPPAGE, 0)
    
    return df

def etf_hedge_strategy(df):
    """波动率ETF对冲策略
    
    该策略利用波动率ETF的期货展期负效应捕捉风险溢价。
    在低波动环境且期限结构正向时做空VXX。
    包含基于VIX水平的动态对冲比例调整和跳跃风险应对机制。
    
    Args:
        df: 包含市场数据的DataFrame
        
    Returns:
        DataFrame: 添加了策略信号和收益的DataFrame
    """
    # 初始化信号序列
    df['etf_signal'] = 0
    
    # 低波动环境下做空VXX
    short_condition = (df['vix'] < 20) & (df['term_structure'] < 1)
    df.loc[short_condition, 'etf_signal'] = -1
    
    # 填充信号，保持持仓直到出现平仓条件
    df['etf_signal'] = df['etf_signal'].replace(to_replace=0, method='ffill')
    
    # 风险控制：VIX快速上升时减仓或平仓
    vix_spike1 = df['vix_change'] > config.VIX_SPIKE_THRESHOLD_1  # VIX日涨幅>10%
    vix_spike2 = df['vix_change'] > config.VIX_SPIKE_THRESHOLD_2  # VIX日涨幅>20%
    
    # 应用风险控制规则
    current_signal = df['etf_signal'].copy()
    for i in range(1, len(df)):
        if current_signal[i] == -1:  # 目前有空头仓位
            if vix_spike2.iloc[i]:  # VIX大幅上涨，全部平仓
                current_signal.iloc[i] = 0
            elif vix_spike1.iloc[i]:  # VIX中幅上涨，减仓50%
                current_signal.iloc[i] = -0.5
    
    df['etf_signal'] = current_signal
    
    # 计算动态对冲比例
    df['hedge_ratio'] = 0.1 + 0.04 * df['vix']
    df.loc[df['vix_change'] > 5, 'hedge_ratio'] += 0.1  # VIX跳跃时增加对冲
    df['hedge_ratio'] = df['hedge_ratio'].clip(lower=0.1, upper=0.9)  # 限制在10%-90%之间
    
    # 计算最终仓位
    df['etf_position'] = df['etf_signal'] * df['hedge_ratio']
    
    # 计算策略收益（不考虑交易成本）
    df['etf_returns'] = df['etf_position'].shift(1) * df['vxx'].pct_change()
    
    # 计算交易成本（ETF交易成本和做空成本）
    calculate_trading_costs(df, 'etf_signal', 'etf_returns', config.ETF_COST + config.SLIPPAGE, config.SHORT_COST)
    
    return df

def calculate_trading_costs(df, signal_col, returns_col, trade_cost, short_cost=0):
    """计算交易成本和做空成本
    
    Args:
        df: 数据框
        signal_col: 信号列名
        returns_col: 收益列名
        trade_cost: 交易成本率
        short_cost: 做空成本率（日化）
    """
    # 计算交易成本（只在开仓和平仓时产生）
    df[f'{signal_col}_change'] = df[signal_col].diff().abs()
    df[f'{returns_col}_trade_cost'] = df[f'{signal_col}_change'] * trade_cost
    
    # 计算做空成本（只在持有空头仓位时产生）
    if short_cost > 0:
        df[f'{returns_col}_short_cost'] = np.where(df[signal_col] < 0, 
                                                 abs(df[signal_col]) * short_cost, 0)
    else:
        df[f'{returns_col}_short_cost'] = 0
    
    # 计算净收益
    df[f'{returns_col}_net'] = df[returns_col] - df[f'{returns_col}_trade_cost'] - df[f'{returns_col}_short_cost']