"""
稳健性测试模块 - 测试策略对参数变化和替代指标的敏感性

通过对关键参数进行敏感性分析和使用替代指标测试系统的稳健性，
验证策略在不同市场情境和参数设置下的表现
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import copy
import time
import logging
from datetime import datetime
import config
from market_state import classify_market_states
from strategies import term_structure_strategy, etf_hedge_strategy
from dynamic_selection import dynamic_strategy_selection
from performance import compare_strategies, train_test_split_analysis
from data_fetcher import fetch_market_data, fetch_data_from_file


logger = logging.getLogger('robustness')


def run_parameter_sensitivity(df, base_config, parameter_name, values, strategies, output_dir=None):
    """运行参数敏感性测试
    
    Args:
        df: 市场数据
        base_config: 基础配置对象
        parameter_name: 要测试的参数名称
        values: 参数值列表
        strategies: 策略列表，每个元素为(策略名称, 收益列名)的元组
        output_dir: 输出目录
        
    Returns:
        dict: 不同参数值对应的性能指标
    """
    logger.info(f"开始 {parameter_name} 参数敏感性测试, 测试值: {values}")
    
    results = {}
    best_sharpe = -float('inf')
    best_value = None
    
    # 获取原始参数值（用于测试后恢复）
    original_value = getattr(base_config, parameter_name, None)
    
    # 遍历每个参数值
    for value in values:
        # 直接修改原始配置对象的参数值而不是创建副本
        setattr(base_config, parameter_name, value)
        
        # 使用修改后的配置运行回测
        test_df = df.copy()
        test_df = classify_market_states(test_df, custom_config=base_config)
        test_df = term_structure_strategy(test_df, custom_config=base_config)
        test_df = etf_hedge_strategy(test_df, custom_config=base_config)
        test_df = dynamic_strategy_selection(test_df, custom_config=base_config)
        
        # 计算策略性能
        performance = compare_strategies(test_df, strategies)
        
        # 记录结果
        results[value] = {
            'dynamic_sharpe': performance['动态策略选择']['Sharpe Ratio'],
            'dynamic_return': performance['动态策略选择']['Annual Return(%)'],
            'dynamic_drawdown': performance['动态策略选择']['Max Drawdown(%)'],
            'ts_sharpe': performance['期限结构策略']['Sharpe Ratio'],
            'etf_sharpe': performance['ETF对冲策略']['Sharpe Ratio'],
            'equal_sharpe': performance['等权组合']['Sharpe Ratio']
        }
        
        # 更新最佳值
        if results[value]['dynamic_sharpe'] > best_sharpe:
            best_sharpe = results[value]['dynamic_sharpe']
            best_value = value
        
        logger.info(f"{parameter_name} = {value}: 动态策略夏普比 = {results[value]['dynamic_sharpe']:.2f}, "
                   f"年化收益 = {results[value]['dynamic_return']:.2f}%, "
                   f"最大回撤 = {results[value]['dynamic_drawdown']:.2f}%")
    
    # 恢复原始参数值
    if original_value is not None:
        setattr(base_config, parameter_name, original_value)
    
    logger.info(f"参数 {parameter_name} 敏感性测试完成, 最佳值 = {best_value} (夏普比 = {best_sharpe:.2f})")
    
    # 如果设置了输出目录, 绘制并保存结果图表
    if output_dir:
        plot_parameter_sensitivity(results, parameter_name, values, 
                                  filename=f"{output_dir}/sensitivity_{parameter_name}.png")
    
    return results


def get_skew_data(start_date, end_date, use_cache=True):
    """获取CBOE SKEW指数数据
    
    Args:
        start_date: 起始日期
        end_date: 结束日期
        use_cache: 是否使用缓存
        
    Returns:
        pandas.Series: SKEW指数数据
    """
    from data_fetcher import fetch_from_yfinance, get_cache_path, is_cache_valid, load_from_cache, save_to_cache
    
    cache_path = get_cache_path("^SKEW", start_date, end_date)
    
    # 检查缓存
    if use_cache and is_cache_valid(cache_path):
        cached_data = load_from_cache(cache_path)
        if cached_data is not None and not cached_data.empty:
            logger.info(f"从缓存加载SKEW数据, 共 {len(cached_data)} 条记录")
            return cached_data['Close']
    
    # 从Yahoo Finance获取SKEW数据
    try:
        skew_data = fetch_from_yfinance("^SKEW", start_date, end_date)
        if skew_data is not None and not skew_data.empty:
            save_to_cache(skew_data, cache_path)
            logger.info(f"成功获取SKEW数据, 共 {len(skew_data)} 条记录")
            return skew_data['Close']
    except Exception as e:
        logger.error(f"获取SKEW数据失败: {str(e)}")
    
    # 如果无法获取SKEW数据, 尝试合成
    logger.warning("尝试基于VIX生成合成SKEW数据")
    
    # 获取VIX数据
    vix_data = fetch_from_yfinance("^VIX", start_date, end_date)
    if vix_data is None or vix_data.empty:
        logger.error("无法获取VIX数据, 无法生成合成SKEW数据")
        return None
    
    # 生成合成SKEW数据 (SKEW通常在100-150范围内, 与VIX相关但不完全相关)
    np.random.seed(42)
    base_skew = 120  # 基础值
    vix_factor = 0.3  # VIX影响因子
    noise = np.random.normal(0, 5, len(vix_data))  # 随机噪声
    
    synthetic_skew = base_skew + vix_factor * vix_data['Close'] + noise
    synthetic_skew = pd.Series(synthetic_skew, index=vix_data.index)
    
    logger.info(f"成功生成合成SKEW数据, 共 {len(synthetic_skew)} 条记录")
    
    return synthetic_skew


def run_alternative_indicators_test(start_date, end_date, strategies, output_dir=None):
    """使用替代指标进行回测
    
    测试使用SKEW代替VIX和200日均线代替50日均线的表现
    
    Args:
        start_date: 回测起始日期
        end_date: 回测结束日期
        strategies: 策略列表
        output_dir: 输出目录
        
    Returns:
        dict: 不同指标组合的性能结果
    """
    logger.info("开始替代指标测试")
    
    # 获取基准数据
    logger.info("获取基准回测数据")
    base_df = fetch_market_data(start_date=start_date, end_date=end_date)
    
    # 使用配置文件中的设置
    use_skew = True
    ma_windows = [50, 200]
    
    if hasattr(config, 'ALT_INDICATORS'):
        use_skew = config.ALT_INDICATORS.get('use_skew', True)
        ma_windows = config.ALT_INDICATORS.get('ma_windows', [50, 200])
    
    # 获取SKEW数据 (如果需要)
    skew_data = None
    if use_skew:
        logger.info("获取SKEW指数数据")
        skew_data = get_skew_data(start_date, end_date)
    
    # 准备测试用例
    test_cases = {
        'baseline': {
            'description': '基准测试 (VIX + 50日均线)',
            'df': base_df.copy(),
            'ma_window': ma_windows[0],
            'use_skew': False
        }
    }
    
    # 根据配置添加测试用例
    if len(ma_windows) > 1 and ma_windows[1] != ma_windows[0]:
        test_cases['ma_alt'] = {
            'description': f'使用{ma_windows[1]}日均线代替{ma_windows[0]}日均线',
            'df': base_df.copy(),
            'ma_window': ma_windows[1],
            'use_skew': False
        }
    
    # 如果有SKEW数据并且配置了使用SKEW
    if skew_data is not None and use_skew:
        test_cases['skew'] = {
            'description': '使用SKEW代替VIX',
            'df': base_df.copy(),
            'ma_window': ma_windows[0],
            'use_skew': True
        }
        
        if len(ma_windows) > 1 and ma_windows[1] != ma_windows[0]:
            test_cases['skew_ma_alt'] = {
                'description': f'同时使用SKEW和{ma_windows[1]}日均线',
                'df': base_df.copy(),
                'ma_window': ma_windows[1],
                'use_skew': True
            }
    
    # 如果有SKEW数据, 添加到相应的测试用例中
    if skew_data is not None and use_skew:
        for case_name, case in test_cases.items():
            if case['use_skew']:
                case['df']['skew'] = skew_data.reindex(case['df'].index)
                # 插值填充缺失值
                case['df']['skew'] = case['df']['skew'].interpolate(method='linear')
    
    # 运行每个测试用例
    results = {}
    for name, case in test_cases.items():
        logger.info(f"运行测试: {case['description']}")
        
        df = case['df']
        
        # 计算指定窗口的移动平均线
        df['spx_ma'] = df['spx'].rolling(window=case['ma_window']).mean()
        
        # 计算趋势指标
        df['spx_trend'] = np.where(df['spx'] > df['spx_ma'] * (1 + config.TREND_STRENGTH), 1,  # 上升趋势
                         np.where(df['spx'] < df['spx_ma'] * (1 - config.TREND_STRENGTH), -1,  # 下降趋势
                                 0))  # 横盘整理
        
        # 使用SKEW代替VIX (如果指定)
        if case['use_skew'] and 'skew' in df.columns:
            # 将SKEW转换为类VIX格式 (反向映射, 因为SKEW与VIX不同)
            # SKEW通常在100-150之间, VIX通常在10-30之间, 且相关性方向相反
            # 简单转换公式: 转换后的VIX = 基础值 + 缩放因子 * (SKEW指数 - 基础SKEW)
            base_value = 15
            scale_factor = 0.5
            base_skew = 120
            
            df['converted_vix'] = base_value + scale_factor * (df['skew'] - base_skew)
            
            # 临时保存原始VIX
            df['original_vix'] = df['vix'].copy()
            
            # 使用转换后的值
            df['vix'] = df['converted_vix']
        
        # 运行市场状态分类和策略
        df = classify_market_states(df)
        df = term_structure_strategy(df)
        df = etf_hedge_strategy(df)
        df = dynamic_strategy_selection(df)
        
        # 计算性能指标
        performance = compare_strategies(df, strategies)
        
        # 记录结果
        results[name] = {
            'description': case['description'],
            'ma_window': case['ma_window'],
            'use_skew': case['use_skew'],
            'dynamic_sharpe': performance['动态策略选择']['Sharpe Ratio'],
            'dynamic_return': performance['动态策略选择']['Annual Return(%)'],
            'dynamic_drawdown': performance['动态策略选择']['Max Drawdown(%)'],
            'ts_sharpe': performance['期限结构策略']['Sharpe Ratio'],
            'etf_sharpe': performance['ETF对冲策略']['Sharpe Ratio'],
            'equal_sharpe': performance['等权组合']['Sharpe Ratio']
        }
        
        logger.info(f"测试 {name} 结果: 动态策略夏普比 = {results[name]['dynamic_sharpe']:.2f}, "
                   f"年化收益 = {results[name]['dynamic_return']:.2f}%, "
                   f"最大回撤 = {results[name]['dynamic_drawdown']:.2f}%")
        
        # 恢复原始VIX (如果使用了SKEW)
        if case['use_skew'] and 'original_vix' in df.columns:
            df['vix'] = df['original_vix']
    
    # 如果设置了输出目录, 绘制并保存结果图表
    if output_dir:
        plot_alternative_indicators(results, 
                                   filename=f"{output_dir}/alternative_indicators.png")
    
    logger.info("替代指标测试完成")
    
    return results


def plot_parameter_sensitivity(results, parameter_name, values, filename=None):
    """绘制参数敏感性测试结果
    
    Args:
        results: 敏感性测试结果
        parameter_name: 参数名称
        values: 参数值列表
        filename: 保存文件名
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 提取性能指标
    x = list(results.keys())
    sharpe_ratios = {
        '动态策略': [results[v]['dynamic_sharpe'] for v in x],
        '期限结构策略': [results[v]['ts_sharpe'] for v in x],
        'ETF对冲策略': [results[v]['etf_sharpe'] for v in x],
        '等权组合': [results[v]['equal_sharpe'] for v in x]
    }
    
    returns = [results[v]['dynamic_return'] for v in x]
    drawdowns = [results[v]['dynamic_drawdown'] for v in x]
    
    # 绘制夏普比率
    for name, values in sharpe_ratios.items():
        if name == '动态策略':
            ax1.plot(x, values, 'o-', linewidth=2, markersize=8, label=name)
        else:
            ax1.plot(x, values, 'o--', alpha=0.7, label=name)
    
    ax1.set_title(f'参数 {parameter_name} 敏感性测试 - 夏普比率', fontsize=16)
    ax1.set_ylabel('夏普比率', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best', fontsize=14)
    
    # 绘制收益和回撤
    ax2_twin = ax2.twinx()
    returns_line, = ax2.plot(x, returns, 'o-', color='green', linewidth=2, markersize=8, label='年化收益')
    ax2.set_ylabel('年化收益 (%)', color='green', fontsize=14)
    ax2.tick_params(axis='y', colors='green')
    
    drawdowns_line, = ax2_twin.plot(x, drawdowns, 'o--', color='red', linewidth=2, markersize=8, label='最大回撤')
    ax2_twin.set_ylabel('最大回撤 (%)', color='red', fontsize=14)
    ax2_twin.tick_params(axis='y', colors='red')
    
    ax2.set_xlabel(f'参数 {parameter_name} 值', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 合并图例
    lines = [returns_line, drawdowns_line]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='best', fontsize=14)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
        logger.info(f"参数敏感性测试图表已保存至 {filename}")


def plot_alternative_indicators(results, filename=None):
    """绘制替代指标测试结果
    
    Args:
        results: 替代指标测试结果
        filename: 保存文件名
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # 提取数据
    test_names = [name for name in results.keys()]
    descriptions = [results[name]['description'] for name in test_names]
    
    # 策略夏普比率对比
    strategies = ['dynamic_sharpe', 'ts_sharpe', 'etf_sharpe', 'equal_sharpe']
    strategy_names = ['动态策略', '期限结构策略', 'ETF对冲策略', '等权组合']
    
    sharpe_data = {}
    for strategy, name in zip(strategies, strategy_names):
        sharpe_data[name] = [results[test]['dynamic_sharpe'] if strategy == 'dynamic_sharpe' else 
                            results[test]['ts_sharpe'] if strategy == 'ts_sharpe' else
                            results[test]['etf_sharpe'] if strategy == 'etf_sharpe' else
                            results[test]['equal_sharpe'] for test in test_names]
    
    # 绘制夏普比率柱状图
    bar_width = 0.2
    x = np.arange(len(test_names))
    
    for i, (name, values) in enumerate(sharpe_data.items()):
        ax1.bar(x + i*bar_width - bar_width*1.5, values, bar_width, label=name)
    
    ax1.set_title('不同指标组合下的夏普比率对比', fontsize=16)
    ax1.set_ylabel('夏普比率', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(descriptions, fontsize=12, rotation=45, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax1.legend(loc='upper left', fontsize=12)
    
    # 绘制动态策略表现对比
    metrics = ['dynamic_return', 'dynamic_drawdown']
    metric_names = ['年化收益(%)', '最大回撤(%)']
    colors = ['green', 'red']
    
    bar_width = 0.35
    x = np.arange(len(test_names))
    
    for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        values = [abs(results[test][metric]) for test in test_names]
        bars = ax2.bar(x + (i-0.5)*bar_width, values, bar_width, 
                     label=name, color=color, alpha=0.7)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=12)
    
    ax2.set_title('不同指标组合下的动态策略表现', fontsize=16)
    ax2.set_ylabel('百分比 (%)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(descriptions, fontsize=12, rotation=45, ha='right')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax2.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
        logger.info(f"替代指标测试图表已保存至 {filename}")


def comprehensive_robustness_test(start_date=config.START_DATE, end_date=config.END_DATE, 
                                 output_dir=config.RESULT_DIR, local_file=None):
    """运行全面的稳健性测试
    
    包括参数敏感性测试、替代指标测试和样本内外测试
    
    Args:
        start_date: 回测起始日期
        end_date: 回测结束日期
        output_dir: 输出目录
        local_file: 本地数据文件路径
        
    Returns:
        dict: 测试结果
    """
    logger.info("开始全面稳健性测试")
    
    # 创建输出目录
    robustness_dir = os.path.join(output_dir, getattr(config, 'ROBUSTNESS_DIR', 'robustness'))
    if not os.path.exists(robustness_dir):
        os.makedirs(robustness_dir)
        logger.info(f"创建稳健性测试结果目录: {robustness_dir}")
    
    # 策略列表
    strategies = [
        ('期限结构策略', 'ts_returns_net'),
        ('ETF对冲策略', 'etf_returns_net'),
        ('等权组合', 'equal_weight_returns'),
        ('动态策略选择', 'dynamic_returns')
    ]
    
    # 1. 获取市场数据
    if local_file:
        logger.info(f"从本地文件加载数据: {local_file}")
        df = fetch_data_from_file(local_file, start_date, end_date)
    else:
        logger.info("从API获取市场数据")
        df = fetch_market_data(start_date=start_date, end_date=end_date)
    
    if df is None or df.empty:
        logger.error("无法获取市场数据, 稳健性测试终止")
        return None
    
    results = {
        'parameter_sensitivity': {},
        'alternative_indicators': None,
        'sample_split': {}
    }
    
    # 2. 参数敏感性测试
    logger.info("开始参数敏感性测试")
    
    # 使用配置文件中的参数范围
    if hasattr(config, 'ROBUSTNESS_PARAMS'):
        for param, values in config.ROBUSTNESS_PARAMS.items():
            logger.info(f"测试参数: {param}, 值范围: {values}")
            results['parameter_sensitivity'][param] = run_parameter_sensitivity(
                df.copy(), config, param, values, strategies, robustness_dir
            )
    else:
        # 如果没有配置参数范围，使用默认值
        # VIX低阈值敏感性测试
        vix_low_values = [12, 14, 15, 17, 19]
        results['parameter_sensitivity']['VIX_LOW_THRESHOLD'] = run_parameter_sensitivity(
            df.copy(), config, 'VIX_LOW_THRESHOLD', vix_low_values, strategies, robustness_dir
        )
        
        # VIX中阈值敏感性测试
        vix_mid_values = [22, 24, 25, 27, 29]
        results['parameter_sensitivity']['VIX_MID_THRESHOLD'] = run_parameter_sensitivity(
            df.copy(), config, 'VIX_MID_THRESHOLD', vix_mid_values, strategies, robustness_dir
        )
        
        # 平滑窗口敏感性测试
        smooth_window_values = [1, 2, 3, 5, 7]
        results['parameter_sensitivity']['SMOOTH_WINDOW'] = run_parameter_sensitivity(
            df.copy(), config, 'SMOOTH_WINDOW', smooth_window_values, strategies, robustness_dir
        )
        
        # 期限结构阈值敏感性测试
        ts_low_values = [0.95, 0.96, 0.97, 0.98, 0.99]
        results['parameter_sensitivity']['TS_LOW_THRESHOLD'] = run_parameter_sensitivity(
            df.copy(), config, 'TS_LOW_THRESHOLD', ts_low_values, strategies, robustness_dir
        )
        
        # 趋势强度敏感性测试
        trend_strength_values = [0.01, 0.015, 0.02, 0.025, 0.03]
        results['parameter_sensitivity']['TREND_STRENGTH'] = run_parameter_sensitivity(
            df.copy(), config, 'TREND_STRENGTH', trend_strength_values, strategies, robustness_dir
        )
    
    # 3. 替代指标测试
    if hasattr(config, 'USE_ALTERNATIVE_INDICATORS') and config.USE_ALTERNATIVE_INDICATORS:
        logger.info("开始替代指标测试")
        results['alternative_indicators'] = run_alternative_indicators_test(
            start_date, end_date, strategies, robustness_dir
        )
    
    # 4. 样本内外测试
    logger.info("开始样本内外测试")
    
    # 运行基准策略
    df = classify_market_states(df)
    df = term_structure_strategy(df)
    df = etf_hedge_strategy(df)
    df = dynamic_strategy_selection(df)
    
    # 使用配置中的分割日期或默认日期
    split_dates = getattr(config, 'SAMPLE_SPLIT_DATES', ['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01'])
    
    for split_date in split_dates:
        logger.info(f"样本内外分割测试: 分割日期 = {split_date}")
        
        train_performance, test_performance = train_test_split_analysis(
            df, strategies, split_date=split_date
        )
        
        results['sample_split'][split_date] = {
            'train': train_performance,
            'test': test_performance
        }
    
    # 保存样本内外测试结果图表
    plot_sample_split_comparison(results['sample_split'], 
                                filename=f"{robustness_dir}/sample_split_comparison.png")
    
    # 5. 生成汇总报告
    report_file = getattr(config, 'ROBUSTNESS_REPORT_FILE', 'robustness_report.txt')
    generate_robustness_report(results, f"{robustness_dir}/{report_file}")
    
    logger.info("全面稳健性测试完成")
    
    return results


def plot_sample_split_comparison(split_results, filename=None):
    """绘制不同分割日期的样本内外测试结果对比
    
    Args:
        split_results: 样本内外测试结果字典
        filename: 保存文件名
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    metrics = ['Sharpe Ratio', 'Annual Return(%)']
    strategies = ['期限结构策略', 'ETF对冲策略', '动态策略选择']
    
    for i, (split_date, results) in enumerate(split_results.items()):
        train_perf = results['train']
        test_perf = results['test']
        
        ax = axs[i]
        x = np.arange(len(strategies))
        bar_width = 0.35
        
        # 绘制夏普比率
        train_sharpe = [train_perf[s]['Sharpe Ratio'] for s in strategies]
        test_sharpe = [test_perf[s]['Sharpe Ratio'] for s in strategies]
        
        bar1 = ax.bar(x - bar_width/2, train_sharpe, bar_width, label='样本内', alpha=0.7)
        bar2 = ax.bar(x + bar_width/2, test_sharpe, bar_width, label='样本外', alpha=0.7)
        
        ax.set_title(f'分割日期: {split_date}', fontsize=14)
        ax.set_ylabel('夏普比率', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        # 添加数值标签
        for bars in [bar1, bar2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=11)
        
        if i == 0:
            ax.legend(loc='upper left', fontsize=12)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
        logger.info(f"样本内外测试对比图表已保存至 {filename}")


def generate_robustness_report(results, filename):
    """生成稳健性测试汇总报告
    
    Args:
        results: 稳健性测试结果
        filename: 保存文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("==== 波动率风险溢价捕捉系统 - 稳健性测试报告 ====\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 参数敏感性测试结果
        f.write("== 1. 参数敏感性测试 ==\n\n")
        
        for param, param_results in results['parameter_sensitivity'].items():
            f.write(f"-- 参数: {param} --\n")
            f.write(f"{'参数值':<10} {'动态策略夏普比':<15} {'期限结构策略夏普比':<20} "
                    f"{'ETF对冲策略夏普比':<20} {'等权组合夏普比':<15}\n")
            
            for value, metrics in param_results.items():
                f.write(f"{value:<10} {metrics['dynamic_sharpe']:<15.2f} "
                        f"{metrics['ts_sharpe']:<20.2f} "
                        f"{metrics['etf_sharpe']:<20.2f} "
                        f"{metrics['equal_sharpe']:<15.2f}\n")
            
            f.write("\n")
        
        # 替代指标测试结果
        f.write("== 2. 替代指标测试 ==\n\n")
        
        if results['alternative_indicators']:
            f.write(f"{'测试组合':<15} {'描述':<30} {'动态策略夏普比':<15} "
                    f"{'年化收益(%)':<15} {'最大回撤(%)':<15}\n")
            
            for name, metrics in results['alternative_indicators'].items():
                f.write(f"{name:<15} {metrics['description']:<30} "
                        f"{metrics['dynamic_sharpe']:<15.2f} "
                        f"{metrics['dynamic_return']:<15.2f} "
                        f"{abs(metrics['dynamic_drawdown']):<15.2f}\n")
        else:
            f.write("替代指标测试未完成或失败\n")
        
        f.write("\n")
        
        # 样本内外测试结果
        f.write("== 3. 样本内外测试 ==\n\n")
        
        for split_date, split_result in results['sample_split'].items():
            train_perf = split_result['train']
            test_perf = split_result['test']
            
            f.write(f"-- 分割日期: {split_date} --\n")
            f.write("样本内结果:\n")
            f.write(f"{'指标':<20} {'期限结构策略':<15} {'ETF对冲策略':<15} "
                    f"{'等权组合':<15} {'动态策略选择':<15}\n")
            
            for metric in ['Annual Return(%)', 'Sharpe Ratio', 'Max Drawdown(%)']:
                f.write(f"{metric:<20} "
                        f"{train_perf['期限结构策略'][metric]:<15.2f} "
                        f"{train_perf['ETF对冲策略'][metric]:<15.2f} "
                        f"{train_perf['等权组合'][metric]:<15.2f} "
                        f"{train_perf['动态策略选择'][metric]:<15.2f}\n")
            
            f.write("\n样本外结果:\n")
            f.write(f"{'指标':<20} {'期限结构策略':<15} {'ETF对冲策略':<15} "
                    f"{'等权组合':<15} {'动态策略选择':<15}\n")
            
            for metric in ['Annual Return(%)', 'Sharpe Ratio', 'Max Drawdown(%)']:
                f.write(f"{metric:<20} "
                        f"{test_perf['期限结构策略'][metric]:<15.2f} "
                        f"{test_perf['ETF对冲策略'][metric]:<15.2f} "
                        f"{test_perf['等权组合'][metric]:<15.2f} "
                        f"{test_perf['动态策略选择'][metric]:<15.2f}\n")
            
            f.write("\n")
        
        # 总结
        f.write("== 4. 稳健性测试总结 ==\n\n")
        
        # VIX阈值敏感性总结
        if 'VIX_LOW_THRESHOLD' in results['parameter_sensitivity'] and 'VIX_MID_THRESHOLD' in results['parameter_sensitivity']:
            vix_low_range = list(results['parameter_sensitivity']['VIX_LOW_THRESHOLD'].keys())
            vix_low_sharpe = [results['parameter_sensitivity']['VIX_LOW_THRESHOLD'][v]['dynamic_sharpe'] for v in vix_low_range]
            vix_low_diff = max(vix_low_sharpe) - min(vix_low_sharpe)
            
            vix_mid_range = list(results['parameter_sensitivity']['VIX_MID_THRESHOLD'].keys())
            vix_mid_sharpe = [results['parameter_sensitivity']['VIX_MID_THRESHOLD'][v]['dynamic_sharpe'] for v in vix_mid_range]
            vix_mid_diff = max(vix_mid_sharpe) - min(vix_mid_sharpe)
            
            f.write(f"1. VIX阈值敏感性:\n")
            f.write(f"   - VIX低阈值 ({min(vix_low_range)}-{max(vix_low_range)}) 夏普比率变化: {vix_low_diff:.2f} "
                    f"(最低: {min(vix_low_sharpe):.2f}, 最高: {max(vix_low_sharpe):.2f})\n")
            f.write(f"   - VIX中阈值 ({min(vix_mid_range)}-{max(vix_mid_range)}) 夏普比率变化: {vix_mid_diff:.2f} "
                    f"(最低: {min(vix_mid_sharpe):.2f}, 最高: {max(vix_mid_sharpe):.2f})\n")
            
            if vix_low_diff < 0.3 and vix_mid_diff < 0.3:
                f.write("   结论: 策略对VIX阈值变化不敏感, 表现稳健\n\n")
            else:
                f.write("   结论: 策略对VIX阈值变化较敏感, 建议优化\n\n")
        
        # 平滑窗口敏感性总结
        if 'SMOOTH_WINDOW' in results['parameter_sensitivity']:
            smooth_range = list(results['parameter_sensitivity']['SMOOTH_WINDOW'].keys())
            smooth_sharpe = [results['parameter_sensitivity']['SMOOTH_WINDOW'][v]['dynamic_sharpe'] for v in smooth_range]
            smooth_diff = max(smooth_sharpe) - min(smooth_sharpe)
            
            f.write(f"2. 状态平滑窗口敏感性:\n")
            f.write(f"   - 窗口范围 ({min(smooth_range)}-{max(smooth_range)}) 夏普比率变化: {smooth_diff:.2f} "
                    f"(最低: {min(smooth_sharpe):.2f}, 最高: {max(smooth_sharpe):.2f})\n")
            
            if smooth_diff < 0.3:
                f.write("   结论: 策略对状态平滑窗口不敏感, 表现稳健\n\n")
            else:
                f.write("   结论: 策略对状态平滑窗口较敏感, 建议优化\n\n")
        
        # 替代指标测试总结
        if results['alternative_indicators']:
            baseline_sharpe = results['alternative_indicators'].get('baseline', {}).get('dynamic_sharpe')
            
            if baseline_sharpe:
                alt_results = []
                
                for name, metrics in results['alternative_indicators'].items():
                    if name != 'baseline':
                        sharpe_diff = metrics['dynamic_sharpe'] - baseline_sharpe
                        alt_results.append((name, metrics['description'], metrics['dynamic_sharpe'], sharpe_diff))
                
                f.write(f"3. 替代指标测试总结:\n")
                f.write(f"   - 基准测试 (VIX + 50日均线) 夏普比率: {baseline_sharpe:.2f}\n")
                
                for name, desc, sharpe, diff in alt_results:
                    f.write(f"   - {desc}: 夏普比率 {sharpe:.2f} (变化: {diff:+.2f})\n")
                
                # 计算平均性能变化
                avg_diff = sum(r[3] for r in alt_results) / len(alt_results) if alt_results else 0
                
                if avg_diff > -0.2:
                    f.write("   结论: 策略在替代指标下仍保持良好表现, 框架稳健\n\n")
                else:
                    f.write("   结论: 策略对指标选择较敏感, 建议进一步研究指标组合\n\n")
        
        # 样本内外测试总结
        if results['sample_split']:
            in_sample_sharpe = []
            out_sample_sharpe = []
            
            for split_date, split_result in results['sample_split'].items():
                in_sample_sharpe.append(split_result['train']['动态策略选择']['Sharpe Ratio'])
                out_sample_sharpe.append(split_result['test']['动态策略选择']['Sharpe Ratio'])
            
            avg_in_sample = sum(in_sample_sharpe) / len(in_sample_sharpe) if in_sample_sharpe else 0
            avg_out_sample = sum(out_sample_sharpe) / len(out_sample_sharpe) if out_sample_sharpe else 0
            sharpe_diff = avg_out_sample - avg_in_sample
            
            f.write(f"4. 样本内外测试总结:\n")
            f.write(f"   - 平均样本内夏普比率: {avg_in_sample:.2f}\n")
            f.write(f"   - 平均样本外夏普比率: {avg_out_sample:.2f}\n")
            f.write(f"   - 样本内外差异: {sharpe_diff:+.2f}\n")
            
            if sharpe_diff > -0.3:
                f.write("   结论: 策略在样本外测试中保持稳定表现, 未出现明显过拟合\n\n")
            else:
                f.write("   结论: 策略在样本外测试中表现下滑, 可能存在过拟合, 建议优化\n\n")
        
        # 最终总结
        f.write("== 最终结论 ==\n\n")
        
        # 基于之前结果计算稳健性综合评分
        robustness_score = 0
        factor_count = 0
        
        # VIX阈值敏感性评分
        if 'VIX_LOW_THRESHOLD' in results['parameter_sensitivity'] and 'VIX_MID_THRESHOLD' in results['parameter_sensitivity']:
            vix_low_diff = max(vix_low_sharpe) - min(vix_low_sharpe)
            vix_mid_diff = max(vix_mid_sharpe) - min(vix_mid_sharpe)
            
            if vix_low_diff < 0.2 and vix_mid_diff < 0.2:
                robustness_score += 5
            elif vix_low_diff < 0.3 and vix_mid_diff < 0.3:
                robustness_score += 4
            elif vix_low_diff < 0.4 and vix_mid_diff < 0.4:
                robustness_score += 3
            else:
                robustness_score += 2
                
            factor_count += 1
        
        # 平滑窗口敏感性评分
        if 'SMOOTH_WINDOW' in results['parameter_sensitivity']:
            if smooth_diff < 0.2:
                robustness_score += 5
            elif smooth_diff < 0.3:
                robustness_score += 4
            elif smooth_diff < 0.4:
                robustness_score += 3
            else:
                robustness_score += 2
                
            factor_count += 1
        
        # 替代指标测试评分
        if results['alternative_indicators']:
            if avg_diff > -0.1:
                robustness_score += 5
            elif avg_diff > -0.2:
                robustness_score += 4
            elif avg_diff > -0.3:
                robustness_score += 3
            else:
                robustness_score += 2
                
            factor_count += 1
        
        # 样本内外测试评分
        if results['sample_split']:
            if sharpe_diff > 0:
                robustness_score += 5
            elif sharpe_diff > -0.2:
                robustness_score += 4
            elif sharpe_diff > -0.3:
                robustness_score += 3
            else:
                robustness_score += 2
                
            factor_count += 1
        
        # 计算最终得分
        final_score = robustness_score / factor_count if factor_count > 0 else 0
        
        # 输出综合评价
        f.write(f"稳健性综合评分: {final_score:.1f}/5.0\n\n")
        
        if final_score >= 4.5:
            f.write("综合评价: 策略展现出极高的稳健性, 对参数设置和指标选择不敏感, 在样本外数据中表现优异。\n"
                    "推荐: 该策略框架适合在实际环境中应用, 无需重大调整。\n")
        elif final_score >= 4.0:
            f.write("综合评价: 策略展现出高度稳健性, 对大多数测试因素表现稳定, 仅有少量敏感点。\n"
                    "推荐: 可以应用于实际环境, 但建议对识别出的敏感点进行优化。\n")
        elif final_score >= 3.5:
            f.write("综合评价: 策略展现出良好稳健性, 大部分测试条件下保持稳定, 但存在一些敏感区域。\n"
                    "推荐: 应用前建议优化识别出的敏感参数, 并增强风险管理机制。\n")
        elif final_score >= 3.0:
            f.write("综合评价: 策略稳健性一般, 对某些参数和条件较为敏感, 但整体框架有效。\n"
                    "推荐: 需要进一步优化参数设置和指标选择, 在应用前进行更全面的样本外测试。\n")
        else:
            f.write("综合评价: 策略稳健性较弱, 对多个测试因素表现敏感, 可能存在过拟合风险。\n"
                    "推荐: 需要重新评估策略设计, 考虑简化模型或采用更稳健的状态分类方法。\n")
            
    logger.info(f"稳健性测试报告已生成: {filename}")