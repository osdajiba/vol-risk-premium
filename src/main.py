"""
波动率风险溢价捕捉系统 - 主程序入口

基于市场状态的动态策略选择框架，结合波动率期限结构交易策略和波动率ETF对冲策略
"""

import os
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import traceback
import sys

# 确保src目录在模块搜索路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入系统各模块
import config
from data_fetcher import fetch_market_data, fetch_data_from_file
from market_state import classify_market_states, analyze_market_states
from strategies import term_structure_strategy, etf_hedge_strategy
from dynamic_selection import dynamic_strategy_selection, analyze_state_transitions
from performance import (
    compare_strategies, analyze_by_market_state, analyze_covid_period,
    train_test_split_analysis
)
from visualization import (
    plot_strategy_performance, plot_market_states, plot_state_performance,
    plot_covid_analysis, plot_weight_transition, plot_train_test_comparison,
    plot_returns_distribution
)

# 配置日志
def setup_logging():
    """配置日志系统"""
    log_level = getattr(logging, config.LOG_LEVEL)
    log_format = config.LOG_FORMAT
    
    # 创建logger
    logger = logging.getLogger('vrp_system')
    logger.setLevel(log_level)
    
    # 清除已有的处理器
    if logger.handlers:
        logger.handlers = []
    
    # 确保日志目录存在
    log_dir = os.path.dirname(config.LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建控制台处理器
    if config.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(config.LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 创建文件处理器
    if hasattr(config, 'LOG_FILE') and config.LOG_FILE:
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(config.LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def run_backtest(start_date=config.START_DATE, end_date=config.END_DATE, 
                output_dir=config.RESULT_DIR, save_plots=True, use_cache=True, 
                force_download=False, local_file=None, logger=None):
    """运行策略回测系统
    
    Args:
        start_date: 回测起始日期
        end_date: 回测结束日期
        output_dir: 结果输出目录
        save_plots: 是否保存图表
        use_cache: 是否使用缓存数据
        force_download: 是否强制重新下载数据
        local_file: 本地数据文件路径（如有）
        logger: 日志记录器
        
    Returns:
        dict: 包含回测结果的字典
    """
    start_time = time.time()
    if logger:
        logger.info(f"开始运行波动率风险溢价捕捉系统回测...")
        logger.info(f"回测期间: {start_date} 至 {end_date}")
    else:
        print(f"开始运行波动率风险溢价捕捉系统回测...")
        print(f"回测期间: {start_date} 至 {end_date}")
    
    # 创建输出目录
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if logger:
            logger.info(f"创建输出目录: {output_dir}")
    
    try:
        # 步骤1: 获取市场数据
        if local_file:
            if logger:
                logger.info(f"从本地文件加载数据: {local_file}")
            df = fetch_data_from_file(local_file, start_date, end_date)
        else:
            if logger:
                logger.info(f"获取市场数据，设置: use_cache={use_cache}, force_download={force_download}")
            df = fetch_market_data(
                start_date=start_date, 
                end_date=end_date,
                use_cache=use_cache,
                force_download=force_download,
                alt_sources=config.USE_ALTERNATIVE_SOURCES
            )
        
        if df is None or df.empty:
            error_msg = "无法获取有效的市场数据，回测终止"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            return None
        
        if logger:
            logger.info(f"成功获取市场数据，共 {len(df)} 个交易日")
        
        # 步骤2: 市场状态分类
        if logger:
            logger.info("进行市场状态分类...")
        df = classify_market_states(df)
        
        # 检查DataFrame是否为空或是否成功添加了市场状态
        if df.empty or 'market_state_smooth' not in df.columns:
            error_msg = "市场状态分类失败，回测终止"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            return None
        
        market_state_analysis = analyze_market_states(df)
        
        # 步骤3: 实现单一策略
        if logger:
            logger.info("执行期限结构交易策略...")
        df = term_structure_strategy(df)
        
        if logger:
            logger.info("执行ETF对冲策略...")
        df = etf_hedge_strategy(df)
        
        # 步骤4: 动态策略选择
        if logger:
            logger.info("执行动态策略选择...")
        df = dynamic_strategy_selection(df)
        state_transitions = analyze_state_transitions(df)
        
        # 步骤5: 绩效分析
        # 定义策略列表
        strategies = [
            ('期限结构策略', 'ts_returns_net'),
            ('ETF对冲策略', 'etf_returns_net'),
            ('等权组合', 'equal_weight_returns'),
            ('动态策略选择', 'dynamic_returns')
        ]
        
        if logger:
            logger.info("进行策略绩效分析...")
        
        # 比较策略绩效
        performance_comparison = compare_strategies(df, strategies)
        print("\n==== 策略绩效对比 ====")
        print(performance_comparison[['期限结构策略', 'ETF对冲策略', '动态策略选择']].loc[
            ['Annual Return(%)', 'Sharpe Ratio', 'Max Drawdown(%)', 'Win Rate(%)']
        ])
        
        # 按市场状态分析策略绩效
        state_performance = analyze_by_market_state(df, strategies)
        print("\n==== 按市场状态分析 ====")
        print(state_performance[['Days', 'Percentage(%)', '期限结构策略 Monthly Return(%)', 
                            'ETF对冲策略 Monthly Return(%)', '动态策略选择 Monthly Return(%)']])
        
        # COVID-19案例分析
        covid_analysis = analyze_covid_period(
            df, strategies, 
            covid_start=config.COVID_START,
            covid_end=config.COVID_END,
            recovery_end=config.COVID_RECOVERY_END
        )
        print("\n==== COVID-19期间分析 ====")
        print(covid_analysis)
        
        # 样本内外测试
        train_performance, test_performance = train_test_split_analysis(
            df, strategies, split_date=config.TRAIN_TEST_SPLIT
        )
        print("\n==== 样本内外绩效对比 ====")
        print("样本内(2015-2019)绩效:")
        print(train_performance[['期限结构策略', 'ETF对冲策略', '动态策略选择']].loc[
            ['Annual Return(%)', 'Sharpe Ratio', 'Max Drawdown(%)']
        ])
        print("\n样本外(2020-2023)绩效:")
        print(test_performance[['期限结构策略', 'ETF对冲策略', '动态策略选择']].loc[
            ['Annual Return(%)', 'Sharpe Ratio', 'Max Drawdown(%)']
        ])
        
        # 步骤6: 可视化
        if save_plots:
            if logger:
                logger.info("生成策略分析图表...")
            
            # 策略累积收益曲线
            plot_strategy_performance(df, strategies, 
                                    filename=f"{output_dir}/strategy_performance.png")
            
            # 市场状态分类图
            plot_market_states(df, filename=f"{output_dir}/market_states.png")
            
            # 分市场状态策略表现
            plot_state_performance(state_performance, 
                                  filename=f"{output_dir}/state_performance.png")
            
            # COVID-19案例分析
            plot_covid_analysis(df, strategies, 
                               filename=f"{output_dir}/covid_analysis.png")
            
            # 策略权重动态转换
            plot_weight_transition(df, filename=f"{output_dir}/weight_transition.png")
            
            # 样本内外对比
            plot_train_test_comparison(train_performance, test_performance,
                                      filename=f"{output_dir}/train_test_comparison.png")
            
            # 收益分布分析
            plot_returns_distribution(df, strategies,
                                     filename=f"{output_dir}/returns_distribution.png")
            
            if logger:
                logger.info(f"图表已保存至 {output_dir} 目录")
            else:
                print(f"图表已保存至 {output_dir} 目录")
        
        # 整合结果
        results = {
            'df': df,
            'market_state_analysis': market_state_analysis,
            'performance_comparison': performance_comparison,
            'state_performance': state_performance,
            'covid_analysis': covid_analysis,
            'train_performance': train_performance,
            'test_performance': test_performance,
            'state_transitions': state_transitions
        }
        
        # 保存结果数据
        if save_plots:
            if logger:
                logger.info("保存回测结果数据...")
            df.to_csv(f"{output_dir}/backtest_data.csv")
            performance_comparison.to_csv(f"{output_dir}/performance_comparison.csv")
            state_performance.to_csv(f"{output_dir}/state_performance.csv")
        
        elapsed_time = time.time() - start_time
        if logger:
            logger.info(f"回测完成，耗时: {elapsed_time:.2f}秒")
        else:
            print(f"\n回测完成，耗时: {elapsed_time:.2f}秒")
        
        return results
    
    except Exception as e:
        error_msg = f"回测过程中发生错误: {str(e)}"
        if logger:
            logger.error(error_msg)
            logger.error(traceback.format_exc())
        else:
            print(error_msg)
            print(traceback.format_exc())
        return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='波动率风险溢价捕捉系统')
    parser.add_argument('--start_date', type=str, default=config.START_DATE,
                        help='回测起始日期，格式：YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=config.END_DATE,
                        help='回测结束日期，格式：YYYY-MM-DD')
    parser.add_argument('--output_dir', type=str, default=config.RESULT_DIR,
                        help='结果输出目录')
    parser.add_argument('--no_plots', action='store_true',
                        help='不保存图表')
    parser.add_argument('--no_cache', action='store_true',
                        help='不使用缓存数据')
    parser.add_argument('--force_download', action='store_true',
                        help='强制重新下载数据')
    parser.add_argument('--local_file', type=str, default=None,
                        help='使用本地数据文件路径')
    return parser.parse_args()


if __name__ == "__main__":
    # 设置日志
    logger = setup_logging()
    logger.info("启动波动率风险溢价捕捉系统")
    
    # 解析命令行参数
    args = parse_args()
    
    # 运行回测
    results = run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
        use_cache=not args.no_cache,
        force_download=args.force_download,
        local_file=args.local_file,
        logger=logger
    )