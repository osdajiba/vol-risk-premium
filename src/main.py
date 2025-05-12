"""
波动率风险溢价捕捉系统 - 主程序入口

基于市场状态的动态策略选择框架，结合波动率期限结构交易策略和波动率ETF对冲策略
"""

import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

# 导入系统各模块
import config
from data_fetcher import fetch_market_data
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


def run_backtest(start_date=config.START_DATE, end_date=config.END_DATE, 
                output_dir='results', save_plots=True):
    """运行策略回测系统
    
    Args:
        start_date: 回测起始日期
        end_date: 回测结束日期
        output_dir: 结果输出目录
        save_plots: 是否保存图表
        
    Returns:
        dict: 包含回测结果的字典
    """
    start_time = time.time()
    print(f"开始运行波动率风险溢价捕捉系统回测...")
    print(f"回测期间: {start_date} 至 {end_date}")
    
    # 创建输出目录
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 步骤1: 获取市场数据
    df = fetch_market_data(start_date, end_date)
    
    # 步骤2: 市场状态分类
    df = classify_market_states(df)
    market_state_analysis = analyze_market_states(df)
    
    # 步骤3: 实现单一策略
    df = term_structure_strategy(df)
    df = etf_hedge_strategy(df)
    
    # 步骤4: 动态策略选择
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
    covid_analysis = analyze_covid_period(df, strategies)
    print("\n==== COVID-19期间分析 ====")
    print(covid_analysis)
    
    # 样本内外测试
    train_performance, test_performance = train_test_split_analysis(df, strategies)
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
        df.to_csv(f"{output_dir}/backtest_data.csv")
        performance_comparison.to_csv(f"{output_dir}/performance_comparison.csv")
        state_performance.to_csv(f"{output_dir}/state_performance.csv")
    
    elapsed_time = time.time() - start_time
    print(f"\n回测完成，耗时: {elapsed_time:.2f}秒")
    
    return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='波动率风险溢价捕捉系统')
    parser.add_argument('--start_date', type=str, default=config.START_DATE,
                        help='回测起始日期，格式：YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=config.END_DATE,
                        help='回测结束日期，格式：YYYY-MM-DD')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='结果输出目录')
    parser.add_argument('--no_plots', action='store_true',
                        help='不保存图表')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 运行回测
    results = run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        save_plots=not args.no_plots
    )