"""
交易策略模块 - 实现波动率期限结构交易和ETF对冲策略
"""

import numpy as np
import pandas as pd
import config

def term_structure_strategy(df, custom_config=None):
    """波动率期限结构交易策略
    
    Args:
        df: 包含市场数据的DataFrame
        custom_config: 自定义配置对象 (用于稳健性测试)
        
    Returns:
        DataFrame: 添加了策略信号和收益的DataFrame
    """
    cfg = custom_config if custom_config is not None else config
    
    df['ts_signal'] = 0
    
    long_condition = (df['term_structure'] < cfg.TS_LOW_THRESHOLD) & (df['vix_change'] > 1)
    short_condition = (df['term_structure'] > cfg.TS_HIGH_THRESHOLD) & (df['vix_change'] < -1)
    
    df.loc[long_condition, 'ts_signal'] = 1
    df.loc[short_condition, 'ts_signal'] = -1
    
    df['ts_signal'] = df['ts_signal'].replace(0, np.nan).ffill().fillna(0)
    
    df['vix_futures_vol'] = df['vix_futures_f1'].pct_change().rolling(window=20).std() * np.sqrt(252)
    df['ts_position_size'] = cfg.TARGET_VOL / df['vix_futures_vol'].clip(lower=0.01)
    df['ts_position_size'] = df['ts_position_size'].clip(lower=0.5, upper=cfg.MAX_LEVERAGE)
    df['ts_position'] = df['ts_signal'] * df['ts_position_size']
    df['ts_returns'] = df['ts_position'].shift(1) * df['vix_futures_f1'].pct_change()
    
    calculate_trading_costs(df, 'ts_signal', 'ts_returns', cfg.FUTURES_COST + cfg.SLIPPAGE, 0, cfg)
    
    return df

def etf_hedge_strategy(df, custom_config=None):
    """波动率 ETF 对冲策略
    
    Args:
        df: 包含市场数据的DataFrame
        custom_config: 自定义配置对象 (用于稳健性测试)
        
    Returns:
        DataFrame: 添加了策略信号和收益的DataFrame
    """
    cfg = custom_config if custom_config is not None else config
    
    df['etf_signal'] = 0
    short_condition = (df['vix'] < 20) & (df['term_structure'] < 1)
    df.loc[short_condition, 'etf_signal'] = -1

    df['etf_signal'] = df['etf_signal'].replace(0, np.nan).ffill().fillna(0)
    
    vix_spike1 = df['vix_change'] > cfg.VIX_SPIKE_THRESHOLD_1
    vix_spike2 = df['vix_change'] > cfg.VIX_SPIKE_THRESHOLD_2
    
    current_signal = df['etf_signal'].copy().astype(float)
    for i in range(1, len(df)):
        if current_signal.iloc[i] == -1:
            if vix_spike2.iloc[i]:
                current_signal.iloc[i] = 0.0
            elif vix_spike1.iloc[i]:
                current_signal.iloc[i] = -0.5
    
    df['etf_signal'] = current_signal
    df['hedge_ratio'] = 0.1 + 0.04 * df['vix']
    df.loc[df['vix_change'] > 5, 'hedge_ratio'] += 0.1
    df['hedge_ratio'] = df['hedge_ratio'].clip(lower=0.1, upper=0.9)
    df['etf_position'] = df['etf_signal'] * df['hedge_ratio']
    df['etf_returns'] = df['etf_position'].shift(1) * df['vxx'].pct_change()
    
    calculate_trading_costs(df, 'etf_signal', 'etf_returns', cfg.ETF_COST + cfg.SLIPPAGE, cfg.SHORT_COST, cfg)
    
    return df

def calculate_trading_costs(df, signal_col, returns_col, trade_cost, short_cost=0, custom_config=None):
    """计算交易成本和做空成本
    
    Args:
        df: 数据框
        signal_col: 信号列名
        returns_col: 收益列名
        trade_cost: 交易成本率
        short_cost: 做空成本率（日化）
        custom_config: 自定义配置对象 (用于稳健性测试)
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