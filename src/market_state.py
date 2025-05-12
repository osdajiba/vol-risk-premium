"""
市场状态分类模块 - 基于VIX水平、市场趋势和期限结构形态分类市场状态
"""

import numpy as np
import pandas as pd
import config

def classify_market_states(df):
    """根据VIX水平、市场趋势和期限结构识别市场状态
    
    Args:
        df: 包含市场数据的DataFrame
        
    Returns:
        DataFrame: 添加了市场状态分类的DataFrame
    """
    # 使用向量化操作分类市场状态
    conditions = [
        # 状态1：平静上涨 - 低VIX，上升趋势，正向期限结构
        (df['vix'] < config.VIX_LOW_THRESHOLD) & 
        (df['spx_trend'] == 1) & 
        (df['term_structure'] < 1),
        
        # 状态2：平静整理 - 低/中VIX，横盘趋势，正向期限结构
        (df['vix'] < config.VIX_MID_THRESHOLD) & 
        (df['spx_trend'] == 0) & 
        (df['term_structure'] < 1),
        
        # 状态3：轻微压力 - 中等VIX，下跌趋势，正向/平坦期限结构
        (df['vix'] >= config.VIX_LOW_THRESHOLD) & 
        (df['vix'] < config.VIX_MID_THRESHOLD) & 
        (df['spx_trend'] == -1) & 
        (df['term_structure'] <= config.TS_HIGH_THRESHOLD),
        
        # 状态4：明显压力 - 中/高VIX，下跌趋势，平坦期限结构
        ((df['vix'] >= config.VIX_MID_THRESHOLD) | 
         (df['spx_trend'] == -1)) & 
        (df['term_structure'] >= config.TS_LOW_THRESHOLD) & 
        (df['term_structure'] <= config.TS_HIGH_THRESHOLD),
        
        # 状态5：恐慌 - 高VIX，急跌趋势，反向期限结构
        (df['vix'] > config.VIX_MID_THRESHOLD) & 
        (df['spx_trend'] == -1) & 
        (df['term_structure'] > 1),
        
        # 状态6：恢复 - 高/中VIX，反弹趋势，反向/平坦期限结构
        (df['vix'] > config.VIX_LOW_THRESHOLD) & 
        (df['spx_trend'] == 1) & 
        (df['term_structure'] >= config.TS_LOW_THRESHOLD)
    ]
    
    states = [1, 2, 3, 4, 5, 6]
    
    # 使用np.select进行向量化分类
    df['market_state'] = np.select(conditions, states, default=3)
    
    # 应用状态平滑处理
    df['market_state_smooth'] = smooth_market_state(df['market_state'], window=config.SMOOTH_WINDOW)
    
    return df

def smooth_market_state(states, window=3):
    """使用滑动窗口平滑市场状态，避免频繁切换
    
    Args:
        states: 市场状态序列
        window: 平滑窗口大小
        
    Returns:
        平滑后的市场状态序列
    """
    smoothed = states.copy()
    
    # 转换为numpy数组以提高处理速度
    states_array = states.values
    smoothed_array = smoothed.values
    
    for i in range(window, len(states_array)):
        # 只有当同一状态持续出现时才切换
        if states_array[i] != smoothed_array[i-1]:
            # 检查前window个状态是否一致
            if np.all(states_array[i-window+1:i+1] == states_array[i]):
                smoothed_array[i] = states_array[i]
            else:
                smoothed_array[i] = smoothed_array[i-1]
    
    return pd.Series(smoothed_array, index=states.index)

def analyze_market_states(df):
    """分析市场状态分布和转换
    
    Args:
        df: 包含市场状态的DataFrame
        
    Returns:
        dict: 包含市场状态分析的字典
    """
    # 计算各市场状态的出现次数和比例
    state_counts = df['market_state_smooth'].value_counts().sort_index()
    state_percent = state_counts / len(df) * 100
    
    # 计算市场状态转换矩阵
    transitions = pd.crosstab(
        df['market_state_smooth'].shift(1),
        df['market_state_smooth'],
        normalize='index'
    )
    
    # 计算各状态的平均持续时间
    durations = []
    current_state = df['market_state_smooth'].iloc[0]
    current_duration = 1
    
    for i in range(1, len(df)):
        if df['market_state_smooth'].iloc[i] == current_state:
            current_duration += 1
        else:
            durations.append((current_state, current_duration))
            current_state = df['market_state_smooth'].iloc[i]
            current_duration = 1
    
    # 添加最后一个状态的持续时间
    durations.append((current_state, current_duration))
    
    # 转换为DataFrame
    durations_df = pd.DataFrame(durations, columns=['state', 'duration'])
    avg_duration = durations_df.groupby('state')['duration'].mean()
    
    # 返回分析结果
    return {
        'state_counts': state_counts,
        'state_percent': state_percent,
        'transitions': transitions,
        'avg_duration': avg_duration
    }