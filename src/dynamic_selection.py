"""
动态策略选择模块 - 基于市场状态的动态策略组合系统
"""

import numpy as np
import pandas as pd
import config

def dynamic_strategy_selection(df):
    """基于市场状态实现动态策略选择
    
    根据市场状态为期限结构策略和ETF对冲策略分配权重，
    使用渐进式调整方法避免频繁切换带来的交易成本。
    
    Args:
        df: 包含市场状态和单一策略收益的DataFrame
        
    Returns:
        DataFrame: 添加了动态策略权重和组合收益的DataFrame
    """
    # 初始化权重为等权
    df['ts_weight'] = 0.5
    df['etf_weight'] = 0.5
    
    # 根据市场状态分配目标权重
    for state, (ts_w, etf_w) in config.STATE_WEIGHTS.items():
        mask = (df['market_state_smooth'] == state)
        df.loc[mask, 'ts_target_weight'] = ts_w
        df.loc[mask, 'etf_target_weight'] = etf_w
    
    # 渐进式权重调整（每日最多调整MAX_DAILY_WEIGHT_CHANGE）
    ts_weight = df['ts_weight'].copy()
    etf_weight = df['etf_weight'].copy()
    
    # 递归调整权重
    for i in range(1, len(df)):
        ts_diff = df['ts_target_weight'].iloc[i] - ts_weight.iloc[i-1]    # 计算目标权重和当前权重的差距, 限制每日调整幅度
        if abs(ts_diff) > config.MAX_DAILY_WEIGHT_CHANGE:
            ts_diff = np.sign(ts_diff) * config.MAX_DAILY_WEIGHT_CHANGE

        ts_weight.iloc[i] = ts_weight.iloc[i-1] + ts_diff
        etf_weight.iloc[i] = 1 - ts_weight.iloc[i]  # 确保权重和为1
    df['ts_weight'] = ts_weight
    df['etf_weight'] = etf_weight
    
    # 确保ETF策略在状态6（恢复期）的权重为0
    recovery_mask = (df['market_state_smooth'] == 6)
    df.loc[recovery_mask, 'etf_weight'] = 0
    df.loc[recovery_mask, 'ts_weight'] = 1
    
    df['dynamic_returns'] = (df['ts_weight'] * df['ts_returns_net'] + df['etf_weight'] * df['etf_returns_net'])    # 计算动态组合收益
    df['equal_weight_returns'] = 0.5 * df['ts_returns_net'] + 0.5 * df['etf_returns_net']    # 额外计算等权重组合收益（作为 baseline 比较）
    
    return df

def calculate_strategy_exposure(df):
    """计算策略风险敞口
    
    Args:
        df: 包含策略权重的DataFrame
        
    Returns:
        DataFrame: 添加了风险敞口指标的DataFrame
    """
    # 计算各策略的波动率
    df['ts_vol'] = df['ts_returns_net'].rolling(window=20).std() * np.sqrt(252)
    df['etf_vol'] = df['etf_returns_net'].rolling(window=20).std() * np.sqrt(252)
    
    # 计算风险贡献
    df['ts_risk_contrib'] = df['ts_weight'] * df['ts_vol']
    df['etf_risk_contrib'] = df['etf_weight'] * df['etf_vol']
    df['total_risk'] = df['ts_risk_contrib'] + df['etf_risk_contrib']
    
    # 计算风险占比
    df['ts_risk_pct'] = df['ts_risk_contrib'] / df['total_risk']
    df['etf_risk_pct'] = df['etf_risk_contrib'] / df['total_risk']
    
    return df

def analyze_state_transitions(df):
    """分析市场状态转换与策略权重变化
    
    Args:
        df: 包含市场状态和权重的DataFrame
        
    Returns:
        dict: 状态转换分析结果
    """
    # 找出所有状态转换点
    state_changes = df['market_state_smooth'].diff() != 0
    transitions = df[state_changes].copy()
    
    # 计算转换前后的平均收益
    results = []
    for i, row in transitions.iterrows():
        if i <= df.index[20] or i >= df.index[-20]:
            continue  # 跳过数据边界
            
        # 获取转换点前后各20个交易日的数据
        pre_period = df.loc[:i].iloc[-20:]
        post_period = df.loc[i:].iloc[:20]
        
        # 计算各策略在转换前后的表现
        result = {
            'date': i,
            'from_state': df.loc[:i, 'market_state_smooth'].iloc[-2],
            'to_state': row['market_state_smooth'],
            'pre_ts_return': pre_period['ts_returns_net'].mean() * 100,
            'post_ts_return': post_period['ts_returns_net'].mean() * 100,
            'pre_etf_return': pre_period['etf_returns_net'].mean() * 100,
            'post_etf_return': post_period['etf_returns_net'].mean() * 100,
            'pre_dynamic_return': pre_period['dynamic_returns'].mean() * 100,
            'post_dynamic_return': post_period['dynamic_returns'].mean() * 100,
            'weight_change': abs(row['ts_weight'] - df.loc[:i, 'ts_weight'].iloc[-2])
        }
        results.append(result)
    
    return pd.DataFrame(results) if results else pd.DataFrame()