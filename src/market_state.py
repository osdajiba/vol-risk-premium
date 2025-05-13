"""
市场状态分类模块 - 基于VIX水平、市场趋势和期限结构形态分类市场状态
"""

import numpy as np
import pandas as pd
import config
import logging


logger = logging.getLogger('market_state')


def classify_market_states(df, custom_config=None):
    """根据VIX水平、市场趋势和期限结构识别市场状态
    
    Args:
        df: 包含市场数据的DataFrame
        custom_config: 自定义配置对象 (用于稳健性测试)
        
    Returns:
        DataFrame: 添加了市场状态分类的DataFrame
    """
    cfg = custom_config if custom_config is not None else config
    
    if df.empty:
        logger.warning("输入的DataFrame为空，无法进行市场状态分类")
        return df
    
    required_columns = ['vix', 'spx_trend', 'term_structure']
    for col in required_columns:
        if col not in df.columns:
            logger.warning(f"缺少必要的列: {col}，无法进行市场状态分类")
            # 添加默认状态列
            df['market_state'] = np.nan
            df['market_state_smooth'] = np.nan
            return df
    
    conditions = [
        # 状态1：平静上涨 - 低VIX，上升趋势，正向期限结构
        (df['vix'] < cfg.VIX_LOW_THRESHOLD) & 
        (df['spx_trend'] == 1) & 
        (df['term_structure'] < 1),
        
        # 状态2：平静整理 - 低/中VIX，横盘趋势，正向期限结构
        (df['vix'] < cfg.VIX_MID_THRESHOLD) & 
        (df['spx_trend'] == 0) & 
        (df['term_structure'] < 1),
        
        # 状态3：轻微压力 - 中等VIX，下跌趋势，正向/平坦期限结构
        (df['vix'] >= cfg.VIX_LOW_THRESHOLD) & 
        (df['vix'] < cfg.VIX_MID_THRESHOLD) & 
        (df['spx_trend'] == -1) & 
        (df['term_structure'] <= cfg.TS_HIGH_THRESHOLD),
        
        # 状态4：明显压力 - 中/高VIX，下跌趋势，平坦期限结构
        ((df['vix'] >= cfg.VIX_MID_THRESHOLD) | 
         (df['spx_trend'] == -1)) & 
        (df['term_structure'] >= cfg.TS_LOW_THRESHOLD) & 
        (df['term_structure'] <= cfg.TS_HIGH_THRESHOLD),
        
        # 状态5：恐慌 - 高VIX，急跌趋势，反向期限结构
        (df['vix'] > cfg.VIX_MID_THRESHOLD) & 
        (df['spx_trend'] == -1) & 
        (df['term_structure'] > 1),
        
        # 状态6：恢复 - 高/中VIX，反弹趋势，反向/平坦期限结构
        (df['vix'] > cfg.VIX_LOW_THRESHOLD) & 
        (df['spx_trend'] == 1) & 
        (df['term_structure'] >= cfg.TS_LOW_THRESHOLD)
    ]
    
    states = [1, 2, 3, 4, 5, 6]
    
    df['market_state'] = np.select(conditions, states, default=3)    # 使用np.select进行向量化分类
    df['market_state_smooth'] = smooth_market_state(df['market_state'], window=cfg.SMOOTH_WINDOW)    # 应用状态平滑处理
    
    return df

def smooth_market_state(states, window=3):
    """使用滑动窗口平滑市场状态，避免频繁切换
    
    Args:
        states: 市场状态序列
        window: 平滑窗口大小
        
    Returns:
        平滑后的市场状态序列
    """
    if states.empty:
        logger.warning("市场状态序列为空，无法进行平滑处理")
        return states
    
    smoothed = states.copy()
    
    # 如果序列长度小于窗口大小，直接返回原序列
    if len(states) < window:
        logger.warning(f"市场状态序列长度 {len(states)} 小于平滑窗口 {window}，跳过平滑处理")
        return smoothed
    
    states_array = states.values
    smoothed_array = smoothed.values
    
    for i in range(window, len(states_array)):
        # 只有当同一状态持续出现时才切换
        if states_array[i] != smoothed_array[i-1]:
            # 检查前 window 个状态是否一致
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
    if df.empty or 'market_state_smooth' not in df.columns:
        logger.warning("输入的DataFrame为空或不包含market_state_smooth列，无法分析市场状态")
        return {
            'state_counts': pd.Series(),
            'state_percent': pd.Series(),
            'transitions': pd.DataFrame(),
            'avg_duration': pd.Series()
        }
    
    if df['market_state_smooth'].isna().all():
        logger.warning("market_state_smooth列全为NaN，无法分析市场状态")
        return {
            'state_counts': pd.Series(),
            'state_percent': pd.Series(),
            'transitions': pd.DataFrame(),
            'avg_duration': pd.Series()
        }
    
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
    
    if len(df) > 0:
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
    
    durations_df = pd.DataFrame(durations, columns=['state', 'duration']) if durations else pd.DataFrame(columns=['state', 'duration'])
    if durations_df.empty:
        avg_duration = pd.Series(dtype='float64')
    else:
        avg_duration = durations_df.groupby('state')['duration'].mean()
    
    return {
        'state_counts': state_counts,
        'state_percent': state_percent,
        'transitions': transitions,
        'avg_duration': avg_duration
    }