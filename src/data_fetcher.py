"""
数据获取模块 - 负责从网络API或本地文件获取所需的金融数据。优先使用本地文件，网络获取时添加数据缓存和健壮的重试机制
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
import os
import config
import requests
from io import StringIO
import logging
import json

# 配置日志
logger = logging.getLogger('data_fetcher')

# 确保数据缓存目录存在
CACHE_DIR = config.DATA_CACHE_DIR
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
    logger.info(f"创建数据缓存目录: {CACHE_DIR}")

def get_cache_path(ticker, start_date, end_date):
    """获取缓存文件路径"""
    # 替换不合法的文件名字符
    ticker_safe = ticker.replace('^', '').replace('.', '_')
    return os.path.join(CACHE_DIR, f"{ticker_safe}_{start_date}_{end_date}.csv")

def is_cache_valid(cache_path, max_age_days=config.CACHE_MAX_AGE_DAYS):
    """检查缓存是否有效"""
    if not os.path.exists(cache_path):
        return False
    
    # 检查文件是否为空
    if os.path.getsize(cache_path) == 0:
        logger.warning(f"缓存文件 {cache_path} 为空，需要重新下载")
        return False
    
    # 检查缓存是否过期（可选）
    file_time = os.path.getmtime(cache_path)
    file_age = (time.time() - file_time) / (60 * 60 * 24)  # 转换为天数
    
    # 仅当请求当前或最近数据时检查缓存年龄
    # 对于历史回测数据，可以使用更长的缓存期
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    if today in cache_path or yesterday in cache_path:
        if file_age > max_age_days:
            logger.info(f"缓存文件 {cache_path} 已过期 ({file_age:.1f} 天)")
            return False
    
    return True

def load_from_cache(cache_path):
    """从缓存加载数据"""
    try:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        logger.info(f"成功从缓存加载数据: {cache_path}")
        return df
    except Exception as e:
        logger.error(f"从缓存加载数据失败: {str(e)}")
        return None

def save_to_cache(df, cache_path):
    """保存数据到缓存"""
    try:
        # 确保父目录存在
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path)
        logger.info(f"成功保存数据到缓存: {cache_path}")
        return True
    except Exception as e:
        logger.error(f"保存数据到缓存失败: {str(e)}")
        return False

def fetch_from_yfinance(ticker, start_date, end_date, max_retries=config.DATA_MAX_RETRIES, 
                       min_delay=config.DATA_MIN_DELAY, max_delay=config.DATA_MAX_DELAY):
    """从Yahoo Finance获取数据，包含增强的重试逻辑"""
    cache_path = get_cache_path(ticker, start_date, end_date)
    
    # 检查缓存
    if is_cache_valid(cache_path):
        cached_data = load_from_cache(cache_path)
        if cached_data is not None and not cached_data.empty:
            return cached_data
    
    logger.info(f"开始从Yahoo Finance下载 {ticker} 数据，日期范围: {start_date} - {end_date}")
    
    # 设置更大的初始延迟时间
    base_delay = min_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"尝试下载 {ticker} 数据 (尝试 {attempt}/{max_retries})...")
            
            # 添加一个初始延迟，避免立即发送请求
            wait_time = base_delay * random.uniform(0.8, 1.2)
            logger.info(f"预防性延迟 {wait_time:.2f} 秒...")
            time.sleep(wait_time)
            
            # 下载数据
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=30)
            
            if data.empty:
                logger.warning(f"{ticker} 数据下载为空，重试...")
                # 数据为空，可能是临时问题，增加延迟并重试
                base_delay = min(base_delay * 2, max_delay)
                continue
            
            logger.info(f"成功下载 {ticker} 数据，共 {len(data)} 条记录")
            
            # 保存到缓存
            save_to_cache(data, cache_path)
            
            return data
            
        except Exception as e:
            # 详细记录错误信息
            if hasattr(e, 'status_code'):
                logger.error(f"下载 {ticker} 数据失败: HTTP状态码 {e.status_code}, {str(e)}")
            else:
                logger.error(f"下载 {ticker} 数据失败: {str(e)}")
            
            if attempt < max_retries:
                # 使用指数退避策略，延迟时间随重试次数增加
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                delay = min(delay, max_delay * 2)  # 设置最大延迟上限
                logger.info(f"等待 {delay:.2f} 秒后重试...")
                time.sleep(delay)
                # 增加基本延迟用于下次尝试
                base_delay = min(base_delay * 1.5, max_delay)
            else:
                logger.error(f"已达到最大重试次数，无法获取 {ticker} 数据")
                # 如果所有重试都失败，尝试返回可能的缓存数据，即使它可能已过期
                cached_data = load_from_cache(cache_path)
                if cached_data is not None and not cached_data.empty:
                    logger.warning(f"返回可能过期的缓存数据作为后备")
                    return cached_data
                raise
    
    return pd.DataFrame()  # 如果所有尝试都失败，返回空DataFrame

def try_alternative_source(ticker, start_date, end_date):
    """从FRED等替代数据源获取数据"""
    logger.info(f"尝试从替代数据源获取 {ticker} 数据")
    
    # 映射YF代码到替代来源
    alt_sources = {
        '^VIX': {
            'source': 'fred',
            'id': 'VIXCLS'
        },
        '^GSPC': {
            'source': 'fred',
            'id': 'SP500'
        }
    }
    
    if ticker not in alt_sources:
        logger.warning(f"没有 {ticker} 的替代数据源")
        return None
    
    source_info = alt_sources[ticker]
    
    if source_info['source'] == 'fred':
        return fetch_from_fred(source_info['id'], start_date, end_date)
    
    return None

def fetch_from_fred(series_id, start_date, end_date, max_retries=3):
    """从FRED（美联储经济数据）获取数据"""
    cache_path = get_cache_path(f"FRED_{series_id}", start_date, end_date)
    
    # 检查缓存
    if is_cache_valid(cache_path):
        cached_data = load_from_cache(cache_path)
        if cached_data is not None:
            return cached_data
    
    # FRED API接口URL
    # 注意：FRED提供免费API，但需要注册获取API密钥
    # 这里使用公共访问URL，但有调用限制
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"从FRED获取 {series_id} 数据，尝试 {attempt+1}/{max_retries}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # 解析CSV数据
            data = pd.read_csv(StringIO(response.text), parse_dates=True, index_col=0)
            data.columns = ['Close']  # 与YF格式一致
            
            # 过滤日期范围
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            data = data[(data.index >= start) & (data.index <= end)]
            
            # 保存到缓存
            save_to_cache(data, cache_path)
            
            return data
            
        except Exception as e:
            logger.error(f"从FRED获取数据失败: {str(e)}")
            if attempt < max_retries - 1:
                delay = 5 + random.uniform(0, 5)
                logger.info(f"等待 {delay:.2f} 秒后重试...")
                time.sleep(delay)
            else:
                logger.error("已达到FRED最大重试次数")
                return None
    
    return None

def check_for_local_data(ticker, start_date, end_date):
    """检查本地数据文件是否存在"""
    # 映射ticker到本地文件名
    ticker_map = {
        '^VIX': config.DEFAULT_DATA_FILES.get('vix'),
        '^GSPC': config.DEFAULT_DATA_FILES.get('spx'),
        'VXX': config.DEFAULT_DATA_FILES.get('vxx')
    }
    
    if ticker not in ticker_map or ticker_map[ticker] is None:
        return None
    
    local_file = ticker_map[ticker]
    
    if os.path.exists(local_file) and os.path.getsize(local_file) > 0:
        try:
            logger.info(f"尝试从本地文件加载 {ticker} 数据: {local_file}")
            df = pd.read_csv(local_file, index_col=0, parse_dates=True)
            
            # 过滤日期范围
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            if df.index.min() <= start and df.index.max() >= end:
                filtered_df = df[(df.index >= start) & (df.index <= end)]
                logger.info(f"成功从本地文件加载 {ticker} 数据，共 {len(filtered_df)} 条记录")
                return filtered_df
            else:
                logger.warning(f"本地 {ticker} 数据日期范围不足，需要从网络获取")
                return None
        except Exception as e:
            logger.error(f"从本地文件加载 {ticker} 数据失败: {str(e)}")
            return None
    
    return None

def fetch_market_data(start_date=config.START_DATE, end_date=config.END_DATE, 
                     use_cache=True, force_download=False, alt_sources=True):
    """获取回测所需的市场数据，优先使用本地文件，支持缓存和多数据源
    
    Args:
        start_date: 回测起始日期
        end_date: 回测结束日期
        use_cache: 是否使用缓存数据
        force_download: 是否强制重新下载数据
        alt_sources: 在主数据源失败时使用替代数据源
    
    Returns:
        pandas.DataFrame: 包含所有必要市场数据的DataFrame
    """
    logger.info(f"获取市场数据，起始日期: {start_date}，结束日期: {end_date}")
    logger.info(f"缓存设置: use_cache={use_cache}, force_download={force_download}")
    
    # 获取更多历史数据用于计算技术指标
    adjusted_start = pd.Timestamp(start_date) - timedelta(days=100)
    adjusted_start_str = adjusted_start.strftime('%Y-%m-%d')
    
    # 尝试从本地合并文件加载
    combined_file = config.DEFAULT_COMBINED_DATA
    if not force_download and os.path.exists(combined_file) and os.path.getsize(combined_file) > 0:
        try:
            logger.info(f"尝试从本地合并文件加载数据: {combined_file}")
            combined_df = pd.read_csv(combined_file, index_col=0, parse_dates=True)
            
            # 过滤日期范围
            start = pd.to_datetime(adjusted_start_str)
            end = pd.to_datetime(end_date)
            
            if combined_df.index.min() <= start and combined_df.index.max() >= end:
                filtered_df = combined_df[(combined_df.index >= start) & (combined_df.index <= end)]
                logger.info(f"成功从本地合并文件加载数据，共 {len(filtered_df)} 条记录")
                
                # 确保数据已经包含所有必需的指标
                required_columns = ['vix', 'spx', 'vxx', 'term_structure', 'vix_futures_f1', 'vix_futures_f2']
                if all(col in filtered_df.columns for col in required_columns):
                    logger.info("本地数据包含所有必需指标")
                    return filtered_df
                else:
                    logger.info("本地数据缺少部分指标，将进行补充计算")
                    # 补充计算期货数据和技术指标
                    if 'vix_futures_f1' not in filtered_df.columns or 'vix_futures_f2' not in filtered_df.columns:
                        generate_synthetic_vix_futures(filtered_df)
                    
                    calculate_technical_indicators(filtered_df)
                    
                    # 保存更新后的数据
                    save_to_cache(filtered_df, combined_file)
                    return filtered_df
            else:
                logger.warning(f"本地合并数据日期范围不足，将尝试其他数据源")
        except Exception as e:
            logger.error(f"从本地合并文件加载数据失败: {str(e)}")
    
    # 组合数据缓存路径
    combined_cache_path = get_cache_path("combined_market_data", adjusted_start_str, end_date)
    
    # 检查组合数据缓存是否有效
    if use_cache and not force_download and is_cache_valid(combined_cache_path):
        combined_data = load_from_cache(combined_cache_path)
        if combined_data is not None and not combined_data.empty:
            logger.info(f"从缓存加载组合市场数据，共 {len(combined_data)} 条记录")
            
            # 确保数据已经包含所有必需的指标
            if 'vix' in combined_data.columns and 'spx' in combined_data.columns and 'vxx' in combined_data.columns:
                if ('term_structure' in combined_data.columns and 
                    'vix_futures_f1' in combined_data.columns and 
                    'vix_futures_f2' in combined_data.columns):
                    logger.info("缓存数据已包含所有必需指标")
                    return combined_data
                else:
                    logger.info("缓存数据缺少部分技术指标，将重新计算")
                    # 生成期货数据和计算技术指标
                    generate_synthetic_vix_futures(combined_data)
                    calculate_technical_indicators(combined_data)
                    # 保存更新后的数据
                    save_to_cache(combined_data, combined_cache_path)
                    return combined_data
    
    # 使用单独数据源获取数据
    tickers = ['^VIX', '^GSPC', 'VXX']
    data_dict = {}
    
    # 尝试从本地文件加载各个数据
    if not force_download:
        for ticker in tickers:
            local_data = check_for_local_data(ticker, adjusted_start_str, end_date)
            if local_data is not None and not local_data.empty:
                data_dict[ticker] = local_data['Close'] if 'Close' in local_data.columns else local_data.iloc[:, 0]
                logger.info(f"从本地文件成功加载 {ticker} 数据，共 {len(local_data)} 条记录")
    
    # 对于没有从本地加载的数据，从网络获取
    missing_tickers = [ticker for ticker in tickers if ticker not in data_dict]
    
    if missing_tickers:
        logger.info(f"尝试从网络获取以下数据: {', '.join(missing_tickers)}")
        
        # 使用批量下载模式获取多个股票数据
        try:
            if len(missing_tickers) > 1:  # 如果有多个缺失的股票，尝试批量下载
                batch_delay = random.uniform(1, 3)
                logger.info(f"批量下载前等待 {batch_delay:.2f} 秒...")
                time.sleep(batch_delay)
                
                batch_data = yf.download(missing_tickers, start=adjusted_start_str, end=end_date, 
                                       progress=False, group_by='ticker', timeout=45)
                
                if not batch_data.empty:
                    for ticker in missing_tickers:
                        if ticker in batch_data.columns.levels[0]:
                            ticker_data = batch_data[ticker].copy()
                            if not ticker_data.empty and 'Close' in ticker_data.columns:
                                data_dict[ticker] = ticker_data['Close']
                                logger.info(f"批量下载成功获取 {ticker} 数据, 共 {len(ticker_data)} 条记录")
                else:
                    logger.warning("批量下载返回空数据，将尝试单独下载")
        except Exception as e:
            logger.warning(f"批量下载失败: {str(e)}，将尝试单独下载每个数据源")
        
        # 对于未通过批量下载获取的股票，尝试单独下载
        for ticker in missing_tickers:
            if ticker not in data_dict or data_dict[ticker].empty:
                logger.info(f"单独下载 {ticker} 数据...")
                
                # 从Yahoo Finance获取数据
                try:
                    ticker_data = fetch_from_yfinance(ticker, adjusted_start_str, end_date)
                    if not ticker_data.empty and 'Close' in ticker_data.columns:
                        data_dict[ticker] = ticker_data['Close']
                        logger.info(f"成功获取 {ticker} 数据，共 {len(ticker_data)} 条记录")
                    else:
                        logger.warning(f"{ticker} 数据下载为空或缺少Close列")
                        
                        # 尝试替代数据源
                        if alt_sources:
                            alt_data = try_alternative_source(ticker, adjusted_start_str, end_date)
                            if alt_data is not None and not alt_data.empty:
                                data_dict[ticker] = alt_data['Close']
                                logger.info(f"从替代源成功获取 {ticker} 数据，共 {len(alt_data)} 条记录")
                
                except Exception as e:
                    logger.error(f"获取 {ticker} 数据失败: {str(e)}")
                    
                    # 尝试替代数据源
                    if alt_sources:
                        alt_data = try_alternative_source(ticker, adjusted_start_str, end_date)
                        if alt_data is not None and not alt_data.empty:
                            data_dict[ticker] = alt_data['Close']
                            logger.info(f"从替代源成功获取 {ticker} 数据，共 {len(alt_data)} 条记录")
    
    # 检查是否获取了所有必需的数据
    if '^VIX' not in data_dict or '^GSPC' not in data_dict:
        raise ValueError("无法获取必需的VIX或S&P500数据")
    
    # 创建主数据框
    df = pd.DataFrame()
    
    # 添加收盘价数据
    df['vix'] = data_dict['^VIX']
    df['spx'] = data_dict['^GSPC']
    
    # VXX是可选的，如果没有获取到，可以使用合成数据
    if 'VXX' in data_dict and not data_dict['VXX'].empty:
        df['vxx'] = data_dict['VXX']
    else:
        logger.warning("未能获取VXX数据，将使用合成数据")
        # 基于VIX创建合成VXX
        df['vxx'] = generate_synthetic_vxx(df['vix'])
    
    # 确保数据没有缺失
    df = df.dropna(subset=['vix', 'spx'])
    
    if df.empty:
        raise ValueError("获取的市场数据为空")
    
    logger.info(f"获取到 {len(df)} 个交易日的数据")
    
    # 生成模拟VIX期货数据和计算技术指标
    generate_synthetic_vix_futures(df)
    calculate_technical_indicators(df)
    
    # 保存组合数据到缓存和本地合并文件
    if use_cache:
        save_to_cache(df, combined_cache_path)
        save_to_cache(df, config.DEFAULT_COMBINED_DATA)  # 同时保存到默认合并文件
    
    # 只保留回测期间的数据
    df = df[df.index >= start_date]
    
    return df

def generate_synthetic_vxx(vix_series):
    """基于VIX创建合成VXX数据
    
    VXX通常与VIX相关，但由于期货展期成本，VXX会有持续的下行偏差
    """
    # 使用随机游走模型，初始化为100
    np.random.seed(42)  # 固定随机种子以确保可重复性
    
    # 从初始值开始
    synthetic_vxx = [100]
    
    # 定义年化衰减率（由于期货展期成本）
    decay_annual = 0.3  # 30%的年化衰减率
    decay_daily = (1 - decay_annual) ** (1/252) - 1  # 转换为日衰减率
    
    # 叠加VIX的日变化率和衰减率，再添加一些随机扰动
    for i in range(1, len(vix_series)):
        # 获取VIX的日涨跌幅
        vix_change = vix_series.iloc[i] / vix_series.iloc[i-1] - 1
        
        # 叠加衰减和随机扰动
        random_component = np.random.normal(0, 0.005)  # 每日0.5%的波动
        vxx_change = vix_change * 0.9 + decay_daily + random_component
        
        # 计算新的VXX值
        new_vxx = synthetic_vxx[-1] * (1 + vxx_change)
        synthetic_vxx.append(new_vxx)
    
    return pd.Series(synthetic_vxx, index=vix_series.index)

def generate_synthetic_vix_futures(df):
    """基于VIX和VXX生成模拟的VIX期货数据
    
    这是一个简化模型，用于生成近似的VIX期货数据。
    实际应用中应使用真实的期货数据。
    """
    # 检查DataFrame是否为空
    if df.empty or 'vix' not in df.columns:
        logger.warning("无法生成VIX期货数据：DataFrame为空或缺少VIX列")
        return df
    
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
    
    return df

def calculate_technical_indicators(df):
    """计算回测所需的技术指标"""
    # 检查DataFrame是否为空
    if df.empty or 'spx' not in df.columns or 'vix' not in df.columns:
        logger.warning("无法计算技术指标：DataFrame为空或缺少必要列")
        return df
    
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

def fetch_data_from_file(file_path, start_date=None, end_date=None):
    """从本地文件中加载数据（用于已有的历史数据）"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return None
            
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"从文件加载数据: {file_path}")
        
        # 筛选日期范围
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]
            
        return df
    except Exception as e:
        logger.error(f"从文件加载数据失败: {str(e)}")
        return None