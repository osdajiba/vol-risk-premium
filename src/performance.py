"""
绩效评估模块 - 计算策略绩效指标并进行比较分析
"""

import numpy as np
import pandas as pd
from scipy import stats

def calculate_performance_metrics(returns, risk_free_rate=0.02/252):
    """计算策略绩效指标集
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险日收益率，默认2%年化
        
    Returns:
        dict: 包含各项绩效指标的字典
    """
    returns = returns.dropna()
    
    # 检查收益率序列是否为空
    if returns.empty:
        return {
            'Total Return(%)': np.nan,
            'Annual Return(%)': np.nan,
            'Annual Volatility(%)': np.nan,
            'Sharpe Ratio': np.nan,
            'Sortino Ratio': np.nan,
            'Max Drawdown(%)': np.nan,
            'Max Drawdown Date': pd.NaT,
            'Calmar Ratio': np.nan,
            'Win Rate(%)': np.nan,
            'Profit/Loss Ratio': np.nan,
            'Max Consecutive Wins': np.nan,
            'Max Consecutive Losses': np.nan,
            'Return Mean': np.nan,
            'Return Std': np.nan,
            'Return Skewness': np.nan,
            'Return Kurtosis': np.nan
        }
    
    # 累积收益
    cum_returns = (1 + returns).cumprod()
    
    # 总收益率
    total_return = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else np.nan
    
    # 年化收益率
    total_days = len(returns)
    annual_return = (cum_returns.iloc[-1] ** (252 / total_days) - 1) if total_days > 0 else np.nan
    
    # 年化波动率
    annual_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else np.nan
    
    # 夏普比率
    excess_returns = returns - risk_free_rate
    sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else np.nan
    
    # 最大回撤
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns / rolling_max - 1)
    max_drawdown = drawdown.min() if not drawdown.empty else np.nan
    max_drawdown_date = drawdown.idxmin() if not drawdown.empty and not pd.isna(drawdown.min()) else pd.NaT
    
    # Calmar 比率
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 and not pd.isna(max_drawdown) else np.nan
    
    # 索提诺比率
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else np.nan
    sortino_ratio = (annual_return - risk_free_rate * 252) / downside_deviation if downside_deviation and downside_deviation > 0 else np.nan
    
    # 胜率
    win_rate = len(returns[returns > 0]) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else np.nan
    
    # 盈亏比
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else np.nan
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else np.nan
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss and avg_loss != 0 else np.nan
    
    # 最大连续盈利/亏损
    pos_streak, neg_streak = calculate_streaks(returns)
    
    # 日收益分布
    return_mean = returns.mean() if len(returns) > 0 else np.nan
    return_std = returns.std() if len(returns) > 1 else np.nan
    # 使用scipy.stats计算偏度和峰度
    if len(returns) > 2:
        try:
            return_skew = stats.skew(returns)
            return_kurt = stats.kurtosis(returns)
        except:
            return_skew = np.nan
            return_kurt = np.nan
    else:
        return_skew = np.nan
        return_kurt = np.nan
    
    return {
        'Total Return(%)': total_return * 100 if not pd.isna(total_return) else np.nan,
        'Annual Return(%)': annual_return * 100 if not pd.isna(annual_return) else np.nan,
        'Annual Volatility(%)': annual_vol * 100 if not pd.isna(annual_vol) else np.nan,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown(%)': max_drawdown * 100 if not pd.isna(max_drawdown) else np.nan,
        'Max Drawdown Date': max_drawdown_date,
        'Calmar Ratio': calmar_ratio,
        'Win Rate(%)': win_rate * 100 if not pd.isna(win_rate) else np.nan,
        'Profit/Loss Ratio': profit_loss_ratio,
        'Max Consecutive Wins': pos_streak,
        'Max Consecutive Losses': neg_streak,
        'Return Mean': return_mean,
        'Return Std': return_std,
        'Return Skewness': return_skew,
        'Return Kurtosis': return_kurt
    }

def calculate_streaks(returns):
    """计算最大连续盈利和亏损次数
    
    Args:
        returns: 收益率序列
        
    Returns:
        tuple: (最大连续盈利次数, 最大连续亏损次数)
    """
    # 检查收益率序列是否为空
    if returns.empty:
        return np.nan, np.nan
    
    # 将收益转换为二元序列：1表示盈利，-1表示亏损
    binary_returns = np.sign(returns)
    binary_returns = binary_returns.replace(0, np.nan).ffill()
    
    # 检查二元序列是否为空
    if binary_returns.empty or binary_returns.isna().all():
        return np.nan, np.nan
    
    # 识别连续序列
    pos_streaks = []
    neg_streaks = []
    
    # 确保第一个元素不是NaN
    if binary_returns.iloc[0] is not pd.NA and not pd.isna(binary_returns.iloc[0]):
        current_streak = 1
        current_sign = binary_returns.iloc[0] 
    else:
        return np.nan, np.nan
    
    for i in range(1, len(binary_returns)):
        if pd.isna(binary_returns.iloc[i]):
            continue
            
        if binary_returns.iloc[i] == current_sign:
            current_streak += 1
        else:
            if current_sign == 1:
                pos_streaks.append(current_streak)
            elif current_sign == -1:
                neg_streaks.append(current_streak)
                
            current_streak = 1
            current_sign = binary_returns.iloc[i]
    
    # 添加最后一个连续序列
    if current_sign == 1:
        pos_streaks.append(current_streak)
    elif current_sign == -1:
        neg_streaks.append(current_streak)
    
    # 计算最大连续次数
    max_pos_streak = max(pos_streaks) if pos_streaks else np.nan
    max_neg_streak = max(neg_streaks) if neg_streaks else np.nan
    
    return max_pos_streak, max_neg_streak

def compare_strategies(df, strategies):
    """比较多个策略的绩效
    
    Args:
        df: 包含策略收益的DataFrame
        strategies: 策略列表，每个元素为(策略名称, 收益列名)的元组
        
    Returns:
        DataFrame: 包含各策略绩效指标对比的DataFrame
    """
    performance = {}
    
    for name, col in strategies:
        if col in df.columns:
            performance[name] = calculate_performance_metrics(df[col].dropna())
        else:
            # 如果收益列不存在，返回NA值
            performance[name] = calculate_performance_metrics(pd.Series([]))
    
    return pd.DataFrame(performance)

def analyze_by_market_state(df, strategies):
    """按市场状态分析策略绩效
    
    Args:
        df: 包含市场状态和策略收益的DataFrame
        strategies: 策略列表，每个元素为(策略名称, 收益列名)的元组
        
    Returns:
        DataFrame: 按市场状态分组的策略月均收益
    """
    state_performance = {}
    
    for state in range(1, 7):
        state_mask = (df['market_state_smooth'] == state)
        
        if sum(state_mask) > 0:
            state_df = df[state_mask]
            
            state_data = {
                'Days': len(state_df),
                'Percentage(%)': len(state_df) / len(df) * 100
            }
            
            # 计算各策略在该状态下的平均月收益
            for name, col in strategies:
                if col in state_df.columns and not state_df[col].dropna().empty:
                    state_data[f'{name} Monthly Return(%)'] = state_df[col].mean() * 21 * 100  # 乘以21个交易日，转换为月收益
                else:
                    state_data[f'{name} Monthly Return(%)'] = np.nan
            
            state_performance[f'State {state}'] = state_data
    
    return pd.DataFrame(state_performance).T

def analyze_covid_period(df, strategies, covid_start, covid_end, recovery_end):
    """分析COVID-19期间的策略表现
    
    Args:
        df: 包含策略收益的DataFrame
        strategies: 策略列表
        covid_start: COVID-19危机开始日期
        covid_end: COVID-19危机结束日期
        recovery_end: 恢复期结束日期
        
    Returns:
        DataFrame: COVID-19期间各策略表现
    """
    covid_start = pd.Timestamp(covid_start)
    covid_end = pd.Timestamp(covid_end)
    recovery_end = pd.Timestamp(recovery_end)
    crisis_df = df[(df.index >= covid_start) & (df.index <= covid_end)]
    recovery_df = df[(df.index > covid_end) & (df.index <= recovery_end)]
    
    covid_performance = {}
    
    # 计算危机期间表现
    covid_performance['Crisis Period'] = {
        'Start Date': covid_start,
        'End Date': covid_end,
        'Duration (Days)': len(crisis_df),
        'SPX Return(%)': (crisis_df['spx'].iloc[-1] / crisis_df['spx'].iloc[0] - 1) * 100 if len(crisis_df) > 0 and 'spx' in crisis_df.columns else np.nan,
        'VIX Change': crisis_df['vix'].iloc[-1] - crisis_df['vix'].iloc[0] if len(crisis_df) > 0 and 'vix' in crisis_df.columns else np.nan
    }
    
    for name, col in strategies:
        if col in crisis_df.columns and len(crisis_df) > 0 and not crisis_df[col].dropna().empty:
            covid_performance['Crisis Period'][f'{name} Return(%)'] = \
                ((1 + crisis_df[col].dropna()).prod() - 1) * 100
        else:
            covid_performance['Crisis Period'][f'{name} Return(%)'] = np.nan
    
    # 计算恢复期表现
    covid_performance['Recovery Period'] = {
        'Start Date': covid_end,
        'End Date': recovery_end,
        'Duration (Days)': len(recovery_df),
        'SPX Return(%)': (recovery_df['spx'].iloc[-1] / recovery_df['spx'].iloc[0] - 1) * 100 if len(recovery_df) > 0 and 'spx' in recovery_df.columns else np.nan,
        'VIX Change': recovery_df['vix'].iloc[-1] - recovery_df['vix'].iloc[0] if len(recovery_df) > 0 and 'vix' in recovery_df.columns else np.nan
    }
    
    for name, col in strategies:
        if col in recovery_df.columns and len(recovery_df) > 0 and not recovery_df[col].dropna().empty:
            covid_performance['Recovery Period'][f'{name} Return(%)'] = \
                ((1 + recovery_df[col].dropna()).prod() - 1) * 100
        else:
            covid_performance['Recovery Period'][f'{name} Return(%)'] = np.nan
    
    return pd.DataFrame(covid_performance)

def train_test_split_analysis(df, strategies, split_date):
    """样本内外绩效对比分析
    
    Args:
        df: 包含策略收益的DataFrame
        strategies: 策略列表
        split_date: 训练测试集分割日期
        
    Returns:
        tuple: (样本内绩效DataFrame, 样本外绩效DataFrame)
    """
    split_date = pd.Timestamp(split_date)
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]
    
    print(f"样本内(训练集)数据: {len(train_df)}天, {train_df.index[0] if not train_df.empty else 'NA'}至{train_df.index[-1] if not train_df.empty else 'NA'}")
    print(f"样本外(测试集)数据: {len(test_df)}天, {test_df.index[0] if not test_df.empty else 'NA'}至{test_df.index[-1] if not test_df.empty else 'NA'}")
    
    # 计算样本内绩效
    train_performance = {}
    for name, col in strategies:
        if col in train_df.columns:
            train_performance[name] = calculate_performance_metrics(train_df[col].dropna())
        else:
            train_performance[name] = calculate_performance_metrics(pd.Series([]))
    train_performance_df = pd.DataFrame(train_performance)
    
    # 计算样本外绩效
    test_performance = {}
    for name, col in strategies:
        if col in test_df.columns:
            test_performance[name] = calculate_performance_metrics(test_df[col].dropna())
        else:
            test_performance[name] = calculate_performance_metrics(pd.Series([]))
    test_performance_df = pd.DataFrame(test_performance)
    
    return train_performance_df, test_performance_df