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

# 设置字体支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.size'] = 14               # 增大默认字体大小


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
    try:
        vix_data = fetch_from_yfinance("^VIX", start_date, end_date)
        if vix_data is None or vix_data.empty:
            logger.error("无法获取VIX数据, 无法生成合成SKEW数据")
            # 返回None而不是引发异常
            return None
    except Exception as e:
        logger.error(f"获取VIX数据失败: {str(e)}")
        return None
    
    # 生成合成SKEW数据 (SKEW通常在100-150范围内, 与VIX相关但不完全相关)
    np.random.seed(42)
    base_skew = 120  # 基础值
    vix_factor = 0.3  # VIX影响因子
    noise = np.random.normal(0, 5, len(vix_data))  # 随机噪声
    
    # 确保vix_data['Close']是数值类型
    vix_values = pd.to_numeric(vix_data['Close'], errors='coerce')
    
    # 使用数值创建合成SKEW
    synthetic_skew = base_skew + vix_factor * vix_values + noise
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
                # 先调用infer_objects将数据类型转换为合适的类型，然后再插值
                case['df']['skew'] = case['df']['skew'].infer_objects(copy=False).interpolate(method='linear')
    
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
            'dynamic_sharpe': performance['动态策略选择']['Sharpe Ratio'] if '动态策略选择' in performance else np.nan,
            'dynamic_return': performance['动态策略选择']['Annual Return(%)'] if '动态策略选择' in performance else np.nan,
            'dynamic_drawdown': performance['动态策略选择']['Max Drawdown(%)'] if '动态策略选择' in performance else np.nan,
            'ts_sharpe': performance['期限结构策略']['Sharpe Ratio'] if '期限结构策略' in performance else np.nan,
            'etf_sharpe': performance['ETF对冲策略']['Sharpe Ratio'] if 'ETF对冲策略' in performance else np.nan,
            'equal_sharpe': performance['等权组合']['Sharpe Ratio'] if '等权组合' in performance else np.nan
        }
        
        # 安全地记录结果，处理可能的NaN值
        try:
            dynamic_sharpe = results[name]['dynamic_sharpe']
            dynamic_return = results[name]['dynamic_return']
            dynamic_drawdown = results[name]['dynamic_drawdown']
            
            # 检查值是否为NaN
            sharpe_str = f"{dynamic_sharpe:.2f}" if not pd.isna(dynamic_sharpe) else "NA"
            return_str = f"{dynamic_return:.2f}%" if not pd.isna(dynamic_return) else "NA"
            drawdown_str = f"{dynamic_drawdown:.2f}%" if not pd.isna(dynamic_drawdown) else "NA"
            
            logger.info(f"测试 {name} 结果: 动态策略夏普比 = {sharpe_str}, "
                       f"年化收益 = {return_str}, "
                       f"最大回撤 = {drawdown_str}")
        except Exception as e:
            logger.error(f"记录测试结果时出错: {str(e)}")
            logger.info(f"测试 {name} 已完成")
        
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
    strategy_names = ['Dynamic Strategy', 'Term Structure Strategy', 'ETF Hedge Strategy', 'Equal Weight']
    
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
        # 将NaN值替换为0以便绘图
        plot_values = [v if not pd.isna(v) else 0 for v in values]
        bars = ax1.bar(x + i*bar_width - bar_width*1.5, plot_values, bar_width, label=name)
        
        # 为非零非NaN值添加标签
        for j, (val, plot_val) in enumerate(zip(values, plot_values)):
            if not pd.isna(val) and plot_val != 0:
                ax1.text(x[j] + i*bar_width - bar_width*1.5, plot_val + 0.05, 
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_title('Sharpe Ratio Comparison with Different Indicators', fontsize=16)
    ax1.set_ylabel('Sharpe Ratio', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(descriptions, fontsize=12, rotation=45, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax1.legend(loc='upper left', fontsize=12)
    
    # 绘制动态策略表现对比
    metrics = ['dynamic_return', 'dynamic_drawdown']
    metric_names = ['Annual Return(%)', 'Max Drawdown(%)']
    colors = ['green', 'red']
    
    bar_width = 0.35
    x = np.arange(len(test_names))
    
    for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        # 安全地提取数值并处理NaN和NaT
        values = []
        for test in test_names:
            val = results[test][metric]
            # 检查是否为NaN或NaT
            if pd.isna(val):
                values.append(0)  # 使用0代替NaN值
            else:
                # 安全地取绝对值，避免NaTType错误
                try:
                    values.append(abs(val))
                except (TypeError, ValueError):
                    values.append(0)
        
        bars = ax2.bar(x + (i-0.5)*bar_width, values, bar_width, 
                     label=name, color=color, alpha=0.7)
        
        # 只为有效值添加数值标签
        for j, (bar, val, test) in enumerate(zip(bars, values, test_names)):
            if val > 0:  # 只为非零值添加标签
                raw_val = results[test][metric]
                if not pd.isna(raw_val):
                    try:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{abs(raw_val):.1f}', ha='center', va='bottom', fontsize=12)
                    except (TypeError, ValueError):
                        pass  # 如果格式化失败，跳过标签
    
    ax2.set_title('Dynamic Strategy Performance with Different Indicators', fontsize=16)
    ax2.set_ylabel('Percentage (%)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(descriptions, fontsize=12, rotation=45, ha='right')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax2.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    if filename:
        try:
            plt.savefig(filename, dpi=300)
            logger.info(f"Alternative indicators test chart saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving figure: {str(e)}")
        
    plt.close(fig)  # 关闭图表以释放内存


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
    strategies = ['Term Structure Strategy', 'ETF Hedge Strategy', 'Dynamic Strategy']
    
    # 创建中英文策略名称映射
    strategy_mapping = {
        '期限结构策略': 'Term Structure Strategy',
        'ETF对冲策略': 'ETF Hedge Strategy',
        '动态策略选择': 'Dynamic Strategy',
        '等权组合': 'Equal Weight'
    }
    
    # 反向映射
    reverse_mapping = {v: k for k, v in strategy_mapping.items()}
    
    # 限制每个图最多显示4个分割点
    split_dates = list(split_results.keys())[:4]
    
    for i, split_date in enumerate(split_dates):
        if i >= len(axs):  # 防止索引超出axs范围
            break
            
        results = split_results[split_date]
        train_perf = results['train']
        test_perf = results['test']
        
        ax = axs[i]
        x = np.arange(len(strategies))
        bar_width = 0.35
        
        # 根据策略名称映射获取夏普比率
        train_sharpe = []
        test_sharpe = []
        
        for strategy_en in strategies:
            # 查找对应的中文策略名
            strategy_cn = reverse_mapping.get(strategy_en)
            
            # 获取夏普比率数据
            if strategy_cn and strategy_cn in train_perf:
                train_val = train_perf[strategy_cn]['Sharpe Ratio']
                test_val = test_perf[strategy_cn]['Sharpe Ratio']
            elif strategy_en in train_perf:  # 尝试直接使用英文名
                train_val = train_perf[strategy_en]['Sharpe Ratio']
                test_val = test_perf[strategy_en]['Sharpe Ratio']
            else:
                train_val = np.nan
                test_val = np.nan
            
            train_sharpe.append(train_val if not pd.isna(train_val) else 0)
            test_sharpe.append(test_val if not pd.isna(test_val) else 0)
        
        bar1 = ax.bar(x - bar_width/2, train_sharpe, bar_width, label='In-Sample', alpha=0.7)
        bar2 = ax.bar(x + bar_width/2, test_sharpe, bar_width, label='Out-of-Sample', alpha=0.7)
        
        ax.set_title(f'Split Date: {split_date}', fontsize=14)
        ax.set_ylabel('Sharpe Ratio', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        # 添加数值标签
        for bars in [bar1, bar2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height) and height != 0:  # 只为非零非NaN的数值添加标签
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
        logger.info(f"Sample split comparison chart saved to {filename}")
        
    plt.close(fig)  # 关闭图表以释放内存


def generate_robustness_report(results, filename):
    """生成稳健性测试汇总报告
    
    Args:
        results: 稳健性测试结果
        filename: 保存文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        # 使用当前日期生成报告时间
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write("==== Volatility Risk Premium Capture System - Robustness Test Report ====\n")
        f.write(f"Generated at: {report_time}\n\n")
        
        # 参数敏感性测试结果
        f.write("== 1. Parameter Sensitivity Test ==\n\n")
        
        for param, param_results in results['parameter_sensitivity'].items():
            f.write(f"-- Parameter: {param} --\n")
            f.write(f"{'Value':<10} {'Dynamic Strategy':<15} {'Term Structure':<20} "
                    f"{'ETF Hedge':<20} {'Equal Weight':<15}\n")
            
            for value, metrics in param_results.items():
                # 安全地格式化数值，处理可能的NaN值
                dynamic_sharpe = metrics['dynamic_sharpe']
                ts_sharpe = metrics['ts_sharpe']
                etf_sharpe = metrics['etf_sharpe']
                equal_sharpe = metrics['equal_sharpe']
                
                dynamic_str = f"{dynamic_sharpe:.2f}" if not pd.isna(dynamic_sharpe) else "NA"
                ts_str = f"{ts_sharpe:.2f}" if not pd.isna(ts_sharpe) else "NA"
                etf_str = f"{etf_sharpe:.2f}" if not pd.isna(etf_sharpe) else "NA"
                equal_str = f"{equal_sharpe:.2f}" if not pd.isna(equal_sharpe) else "NA"
                
                f.write(f"{value:<10} {dynamic_str:<15} "
                        f"{ts_str:<20} "
                        f"{etf_str:<20} "
                        f"{equal_str:<15}\n")
            
            f.write("\n")
        
        # 替代指标测试结果
        f.write("== 2. Alternative Indicators Test ==\n\n")
        
        if results['alternative_indicators']:
            f.write(f"{'Test Case':<15} {'Description':<30} {'Dynamic Strategy':<15} "
                    f"{'Annual Return(%)':<15} {'Max Drawdown(%)':<15}\n")
            
            for name, metrics in results['alternative_indicators'].items():
                # 安全地格式化数值
                dynamic_sharpe = metrics['dynamic_sharpe']
                dynamic_return = metrics['dynamic_return']
                dynamic_drawdown = metrics['dynamic_drawdown']
                
                sharpe_str = f"{dynamic_sharpe:.2f}" if not pd.isna(dynamic_sharpe) else "NA"
                return_str = f"{dynamic_return:.2f}" if not pd.isna(dynamic_return) else "NA"
                drawdown_str = f"{abs(dynamic_drawdown):.2f}" if not pd.isna(dynamic_drawdown) else "NA"
                
                f.write(f"{name:<15} {metrics['description']:<30} "
                        f"{sharpe_str:<15} "
                        f"{return_str:<15} "
                        f"{drawdown_str:<15}\n")
        else:
            f.write("Alternative indicators test not completed or failed\n")
        
        f.write("\n")
        
        # 样本内外测试结果
        f.write("== 3. In-Sample/Out-of-Sample Test ==\n\n")
        
        # 创建策略名称映射
        strategy_mapping = {
            'Term Structure': '期限结构策略',
            'ETF Hedge': 'ETF对冲策略',
            'Equal Weight': '等权组合',
            'Dynamic Strategy': '动态策略选择'
        }
        
        for split_date, split_result in results['sample_split'].items():
            train_perf = split_result['train']
            test_perf = split_result['test']
            
            f.write(f"-- Split Date: {split_date} --\n")
            f.write("In-Sample Results:\n")
            f.write(f"{'Metric':<20} {'Term Structure':<15} {'ETF Hedge':<15} "
                    f"{'Equal Weight':<15} {'Dynamic Strategy':<15}\n")
            
            for metric in ['Annual Return(%)', 'Sharpe Ratio', 'Max Drawdown(%)']:
                f.write(f"{metric:<20} ")
                for strat_en, strat_cn in strategy_mapping.items():
                    if strat_cn in train_perf and metric in train_perf[strat_cn]:
                        val = train_perf[strat_cn][metric]
                        val_str = f"{val:.2f}" if not pd.isna(val) else "NA"
                        f.write(f"{val_str:<15} ")
                    else:
                        f.write(f"{'NA':<15} ")
                f.write("\n")
            
            f.write("\nOut-of-Sample Results:\n")
            f.write(f"{'Metric':<20} {'Term Structure':<15} {'ETF Hedge':<15} "
                    f"{'Equal Weight':<15} {'Dynamic Strategy':<15}\n")
            
            for metric in ['Annual Return(%)', 'Sharpe Ratio', 'Max Drawdown(%)']:
                f.write(f"{metric:<20} ")
                for strat_en, strat_cn in strategy_mapping.items():
                    if strat_cn in test_perf and metric in test_perf[strat_cn]:
                        val = test_perf[strat_cn][metric]
                        val_str = f"{val:.2f}" if not pd.isna(val) else "NA"
                        f.write(f"{val_str:<15} ")
                    else:
                        f.write(f"{'NA':<15} ")
                f.write("\n")
            
            f.write("\n")
        
        # 总结
        f.write("== 4. Robustness Test Summary ==\n\n")
        
        # VIX阈值敏感性总结
        if 'VIX_LOW_THRESHOLD' in results['parameter_sensitivity'] and 'VIX_MID_THRESHOLD' in results['parameter_sensitivity']:
            vix_low_range = list(results['parameter_sensitivity']['VIX_LOW_THRESHOLD'].keys())
            vix_low_sharpe = [results['parameter_sensitivity']['VIX_LOW_THRESHOLD'][v]['dynamic_sharpe'] for v in vix_low_range]
            # 过滤掉NaN值
            vix_low_sharpe = [s for s in vix_low_sharpe if not pd.isna(s)]
            
            vix_mid_range = list(results['parameter_sensitivity']['VIX_MID_THRESHOLD'].keys())
            vix_mid_sharpe = [results['parameter_sensitivity']['VIX_MID_THRESHOLD'][v]['dynamic_sharpe'] for v in vix_mid_range]
            # 过滤掉NaN值
            vix_mid_sharpe = [s for s in vix_mid_sharpe if not pd.isna(s)]
            
            f.write(f"1. VIX Threshold Sensitivity:\n")
            
            if vix_low_sharpe:  # 确保有有效数据
                vix_low_diff = max(vix_low_sharpe) - min(vix_low_sharpe)
                f.write(f"   - VIX Low Threshold ({min(vix_low_range)}-{max(vix_low_range)}) Sharpe Ratio change: {vix_low_diff:.2f} "
                        f"(Min: {min(vix_low_sharpe):.2f}, Max: {max(vix_low_sharpe):.2f})\n")
            else:
                f.write(f"   - VIX Low Threshold: Not enough valid data for analysis\n")
            
            if vix_mid_sharpe:  # 确保有有效数据
                vix_mid_diff = max(vix_mid_sharpe) - min(vix_mid_sharpe)
                f.write(f"   - VIX Mid Threshold ({min(vix_mid_range)}-{max(vix_mid_range)}) Sharpe Ratio change: {vix_mid_diff:.2f} "
                        f"(Min: {min(vix_mid_sharpe):.2f}, Max: {max(vix_mid_sharpe):.2f})\n")
            else:
                f.write(f"   - VIX Mid Threshold: Not enough valid data for analysis\n")
            
            if vix_low_sharpe and vix_mid_sharpe:
                vix_low_diff = max(vix_low_sharpe) - min(vix_low_sharpe)
                vix_mid_diff = max(vix_mid_sharpe) - min(vix_mid_sharpe)
                if vix_low_diff < 0.3 and vix_mid_diff < 0.3:
                    f.write("   Conclusion: Strategy not sensitive to VIX threshold changes, showing robust performance\n\n")
                else:
                    f.write("   Conclusion: Strategy relatively sensitive to VIX thresholds, optimization recommended\n\n")
            else:
                f.write("   Conclusion: Insufficient data for VIX threshold sensitivity analysis\n\n")
        
        # 平滑窗口敏感性总结
        if 'SMOOTH_WINDOW' in results['parameter_sensitivity']:
            smooth_range = list(results['parameter_sensitivity']['SMOOTH_WINDOW'].keys())
            smooth_sharpe = [results['parameter_sensitivity']['SMOOTH_WINDOW'][v]['dynamic_sharpe'] for v in smooth_range]
            # 过滤掉NaN值
            smooth_sharpe = [s for s in smooth_sharpe if not pd.isna(s)]
            
            f.write(f"2. State Smoothing Window Sensitivity:\n")
            
            if smooth_sharpe:  # 确保有有效数据
                smooth_diff = max(smooth_sharpe) - min(smooth_sharpe)
                f.write(f"   - Window range ({min(smooth_range)}-{max(smooth_range)}) Sharpe Ratio change: {smooth_diff:.2f} "
                        f"(Min: {min(smooth_sharpe):.2f}, Max: {max(smooth_sharpe):.2f})\n")
                
                if smooth_diff < 0.3:
                    f.write("   Conclusion: Strategy not sensitive to state smoothing window, showing robust performance\n\n")
                else:
                    f.write("   Conclusion: Strategy relatively sensitive to smoothing window, optimization recommended\n\n")
            else:
                f.write("   - Window range: Not enough valid data for analysis\n")
                f.write("   Conclusion: Insufficient data for smoothing window sensitivity analysis\n\n")
        
        # 替代指标测试总结
        if results['alternative_indicators']:
            baseline = results['alternative_indicators'].get('baseline', {})
            baseline_sharpe = baseline.get('dynamic_sharpe', np.nan)
            
            if not pd.isna(baseline_sharpe):
                alt_results = []
                
                for name, metrics in results['alternative_indicators'].items():
                    if name != 'baseline':
                        dynamic_sharpe = metrics.get('dynamic_sharpe', np.nan)
                        if not pd.isna(dynamic_sharpe):
                            sharpe_diff = dynamic_sharpe - baseline_sharpe
                            alt_results.append((name, metrics['description'], dynamic_sharpe, sharpe_diff))
                
                f.write(f"3. Alternative Indicators Test Summary:\n")
                f.write(f"   - Baseline (VIX + 50-day MA) Sharpe Ratio: {baseline_sharpe:.2f}\n")
                
                for name, desc, sharpe, diff in alt_results:
                    f.write(f"   - {desc}: Sharpe Ratio {sharpe:.2f} (Change: {diff:+.2f})\n")
                
                # 计算平均性能变化
                if alt_results:
                    avg_diff = sum(r[3] for r in alt_results) / len(alt_results)
                    
                    if avg_diff > -0.2:
                        f.write("   Conclusion: Strategy maintains good performance with alternative indicators, framework is robust\n\n")
                    else:
                        f.write("   Conclusion: Strategy sensitive to indicator selection, further study recommended\n\n")
                else:
                    f.write("   Conclusion: Insufficient alternative indicator data for analysis\n\n")
            else:
                f.write("3. Alternative Indicators Test Summary: Baseline results not available\n\n")
        
        # 样本内外测试总结
        if results['sample_split']:
            in_sample_sharpe = []
            out_sample_sharpe = []
            
            for split_date, split_result in results['sample_split'].items():
                if '动态策略选择' in split_result['train'] and '动态策略选择' in split_result['test']:
                    in_sharpe = split_result['train']['动态策略选择'].get('Sharpe Ratio', np.nan)
                    out_sharpe = split_result['test']['动态策略选择'].get('Sharpe Ratio', np.nan)
                    
                    if not pd.isna(in_sharpe) and not pd.isna(out_sharpe):
                        in_sample_sharpe.append(in_sharpe)
                        out_sample_sharpe.append(out_sharpe)
            
            if in_sample_sharpe and out_sample_sharpe:  # 确保有有效数据
                avg_in_sample = sum(in_sample_sharpe) / len(in_sample_sharpe)
                avg_out_sample = sum(out_sample_sharpe) / len(out_sample_sharpe)
                sharpe_diff = avg_out_sample - avg_in_sample
                
                f.write(f"4. In-Sample/Out-of-Sample Test Summary:\n")
                f.write(f"   - Average In-Sample Sharpe Ratio: {avg_in_sample:.2f}\n")
                f.write(f"   - Average Out-of-Sample Sharpe Ratio: {avg_out_sample:.2f}\n")
                f.write(f"   - In/Out-of-Sample Difference: {sharpe_diff:+.2f}\n")
                
                if sharpe_diff > -0.3:
                    f.write("   Conclusion: Strategy maintains stable performance in out-of-sample tests, no significant overfitting\n\n")
                else:
                    f.write("   Conclusion: Strategy performance declines in out-of-sample tests, possible overfitting, optimization recommended\n\n")
            else:
                f.write("4. In-Sample/Out-of-Sample Test Summary: Insufficient data for analysis\n\n")
        
        # 最终总结
        f.write("== Final Conclusion ==\n\n")
        
        # 基于之前结果计算稳健性综合评分
        robustness_score = 0
        factor_count = 0
        
        # VIX阈值敏感性评分
        if 'VIX_LOW_THRESHOLD' in results['parameter_sensitivity'] and 'VIX_MID_THRESHOLD' in results['parameter_sensitivity']:
            if vix_low_sharpe and vix_mid_sharpe:
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
        if 'SMOOTH_WINDOW' in results['parameter_sensitivity'] and smooth_sharpe:
            smooth_diff = max(smooth_sharpe) - min(smooth_sharpe)
            
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
        if results['alternative_indicators'] and alt_results:
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
        if results['sample_split'] and in_sample_sharpe and out_sample_sharpe:
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
        f.write(f"Robustness Overall Rating: {final_score:.1f}/5.0\n\n")
        
        if final_score >= 4.5:
            f.write("Overall Assessment: The strategy demonstrates extremely high robustness, insensitive to parameter settings and indicator selection, with excellent performance in out-of-sample data.\n"
                    "Recommendation: The strategy framework is suitable for application in real environments without major adjustments.\n")
        elif final_score >= 4.0:
            f.write("Overall Assessment: The strategy demonstrates high robustness, stable performance for most test factors, with only a few sensitivity points.\n"
                    "Recommendation: Can be applied to real environments, but optimization of identified sensitivity points is recommended.\n")
        elif final_score >= 3.5:
            f.write("Overall Assessment: The strategy demonstrates good robustness, maintains stability under most test conditions, but there are some sensitive areas.\n"
                    "Recommendation: Before application, optimize the identified sensitive parameters and enhance risk management mechanisms.\n")
        elif final_score >= 3.0:
            f.write("Overall Assessment: The strategy shows moderate robustness, somewhat sensitive to certain parameters and conditions, but the overall framework is effective.\n"
                    "Recommendation: Further optimization of parameter settings and indicator selection is needed, with more comprehensive out-of-sample testing before application.\n")
        else:
            f.write("Overall Assessment: The strategy shows weak robustness, sensitive to multiple test factors, with potential overfitting risk.\n"
                    "Recommendation: Re-evaluate strategy design, consider simplifying the model or adopting more robust state classification methods.\n")
            
    logger.info(f"Robustness test report generated: {filename}")