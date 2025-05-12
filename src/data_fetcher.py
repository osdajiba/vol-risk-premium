"""
数据获取模块 - 负责从网络API获取所需的金融数据
添加了重试逻辑和请求间延迟以处理速率限制
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
import config

def fetch_market_data(start_date=config.START_DATE, end_date=config.END_DATE, max_retries=5, min_delay=1, max_delay=5):
    """获取回测所需的市场数据，支持重试逻辑
    
    Args:
        start_date: 回测起始日期
        end_date: 回测结束日期
        max_retries: 最大重试次数
        min_delay: 最小延迟秒数
        max_delay: 最大延迟秒数
    
    Returns:
        pandas.DataFrame: 包含所有必要市场数据的DataFrame
    """
    print(f"获取市场数据，起始日期: {start_date}，结束日期: {end_date}")
    
    # 获取更多历史数据用于计算技术指标
    adjusted_start = pd.Timestamp(start_date) - timedelta(days=100)
    adjusted_start_str = adjusted_start.strftime('%Y-%m-%d')
    
    # 定义下载单个数据的函数，包含重试逻辑
    def download_with_retry(ticker, start, end, name):
        for attempt in range(max_retries):
            try:
                print(f"尝试下载{name}数据 (尝试 {attempt+1}/{max_retries})...")
                data = yf.download(ticker, start=start, end=end, progress=False)
                print(f"成功获取{name}数据，共{len(data)}条记录")
                return data
            except Exception as e:
                print(f"下载{name}数据失败: {str(e)}")
                if attempt < max_retries - 1:
                    # 使用指数退避策略，延迟时间随重试次数增加
                    delay = min_delay + (max_delay - min_delay) * random.random() * (2 ** attempt)
                    print(f"等待 {delay:.2f} 秒后重试...")
                    time.sleep(delay)
                else:
                    print(f"已达到最大重试次数，无法获取{name}数据")
                    raise
    
    # 依次获取各项数据，并在请求间增加延迟
    vix = download_with_retry('^VIX', adjusted_start_str, end_date, 'VIX指数')
    time.sleep(min_delay + random.random() * (max_delay - min_delay))
    
    spx = download_with_retry('^GSPC', adjusted_start_str, end_date, 'S&P500指数')
    time.sleep(min_delay + random.random() * (max_delay - min_delay))
    
    vxx = download_with_retry('VXX', adjusted_start_str, end_date, 'VXX ETF')
    
    # 创建主数据框
    df = pd.DataFrame()
    df['date'] = vix.index
    df['vix'] = vix['Close']
    df['spx'] = spx['Close']
    df['vxx'] = vxx['Close']
    
    # 设置日期为索引
    df.set_index('date', inplace=True)
    
    # 只保留交易日数据
    df = df.dropna(subset=['vix', 'spx'])
    
    print(f"获取到 {len(df)} 个交易日的数据")
    
    # 生成模拟VIX期货数据和计算技术指标
    generate_synthetic_vix_futures(df)
    calculate_technical_indicators(df)
    
    # 只保留回测期间的数据
    df = df[df.index >= start_date]
    
    return df

def generate_synthetic_vix_futures(df):
    """基于VIX和VXX生成模拟的VIX期货数据
    
    这是一个简化模型，用于生成近似的VIX期货数据。
    实际应用中应使用真实的期货数据。
    """
    # 计算VIX的20日历史波动率
    df['vix_vol'] = df['vix'].rolling(window=20).std()
    
    # 假设近月期货的升贴水与VIX水平和历史波动率相关
    # 低VIX时期，期货通常有升水；高VIX时期，期货可能有贴水
    vix_premium = np.where(df['vix'] < 20, 
                         0.05 * (20 - df['vix']), 
                         -0.03 * (df['vix'] - 20))
    
    # 添加小幅随机扰动以模拟市场噪音
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, len(df))
    
    # 生成近月VIX期货价格
    df['vix_futures_f1'] = df['vix'] + vix_premium + noise
    
    # 生成次近月VIX期货价格（通常有轻微升水）
    df['vix_futures_f2'] = df['vix_futures_f1'] * np.where(df['vix'] < 25, 1.03, 0.98) + np.random.normal(0, 0.3, len(df))
    
    # 在极端VIX情况下模拟期限结构反转
    high_vix_mask = df['vix'] > 30
    reversal_prob = (df['vix'] - 30) / 50
    reversal_prob = reversal_prob.clip(0, 0.8)
    
    # 生成反转样本
    np.random.seed(43)
    reversal = np.random.random(len(df)) < reversal_prob
    reversal_mask = high_vix_mask & reversal
    
    # 应用反转
    temp = df.loc[reversal_mask, 'vix_futures_f1'].copy()
    df.loc[reversal_mask, 'vix_futures_f1'] = df.loc[reversal_mask, 'vix_futures_f2']
    df.loc[reversal_mask, 'vix_futures_f2'] = temp
    
    # 计算期限结构指标
    df['term_structure'] = df['vix_futures_f1'] / df['vix_futures_f2']

def calculate_technical_indicators(df):
    """计算回测所需的技术指标"""
    # 计算均线
    df['spx_ma50'] = df['spx'].rolling(window=50).mean()
    
    # 计算趋势指标
    df['spx_trend'] = np.where(df['spx'] > df['spx_ma50'] * (1 + config.TREND_STRENGTH), 1,  # 上升趋势
                      np.where(df['spx'] < df['spx_ma50'] * (1 - config.TREND_STRENGTH), -1,  # 下降趋势
                               0))  # 横盘整理
    
    # 计算VIX变化率
    df['vix_change'] = df['vix'].pct_change() * 100
    df['vix_ma10'] = df['vix'].rolling(window=10).mean()
    
    return df