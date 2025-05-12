"""
数据可视化模块 - 生成策略分析图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats
import config

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

def plot_strategy_performance(df, strategies, filename=None):
    """绘制策略累积收益曲线
    
    Args:
        df: 包含策略收益的DataFrame
        strategies: 策略列表，每个元素为(策略名称, 收益列名)的元组
        filename: 保存文件名，若为None则不保存
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 计算并绘制累积收益曲线
    for name, col in strategies:
        cum_returns = (1 + df[col]).cumprod()
        ax.plot(cum_returns.index, cum_returns, label=name, linewidth=2)
    
    # 格式化横坐标为日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    # 添加网格和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', fontsize=12)
    
    # 添加标题和标签
    ax.set_title('策略累积收益对比 (2015-2023)', fontsize=14)
    ax.set_ylabel('累积收益 (初始资金=1)', fontsize=12)
    ax.set_xlabel('日期', fontsize=12)
    
    # 标记COVID-19时期
    covid_start = pd.Timestamp(config.COVID_START)
    covid_end = pd.Timestamp(config.COVID_END)
    recovery_end = pd.Timestamp(config.COVID_RECOVERY_END)
    
    ax.axvspan(covid_start, covid_end, color='red', alpha=0.2, label='COVID-19危机期')
    ax.axvspan(covid_end, recovery_end, color='blue', alpha=0.1, label='恢复期')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    return fig

def plot_market_states(df, filename=None):
    """绘制市场状态分类图
    
    Args:
        df: 包含市场状态的DataFrame
        filename: 保存文件名，若为None则不保存
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 绘制VIX指数
    ax1.plot(df.index, df['vix'], 'b-', label='VIX指数', alpha=0.8)
    ax1.fill_between(df.index, 0, df['vix'], color='blue', alpha=0.1)
    
    # 添加VIX阈值线
    ax1.axhline(y=config.VIX_LOW_THRESHOLD, color='green', linestyle='--', alpha=0.7, 
               label=f'低波动阈值 ({config.VIX_LOW_THRESHOLD})')
    ax1.axhline(y=config.VIX_MID_THRESHOLD, color='red', linestyle='--', alpha=0.7, 
               label=f'高波动阈值 ({config.VIX_MID_THRESHOLD})')
    
    # 格式化横坐标为日期
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    # 添加标题和标签
    ax1.set_title('VIX指数与市场状态分类 (2015-2023)', fontsize=14)
    ax1.set_ylabel('VIX', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    
    # 绘制市场状态热图
    cmap = plt.cm.get_cmap('viridis', 6)
    cmap = plt.cm.colors.ListedColormap([cmap(i) for i in range(cmap.N)])
    
    ax2.scatter(df.index, np.ones(len(df)), c=df['market_state_smooth'], 
               cmap=cmap, marker='s', s=3, vmin=1, vmax=6)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 6))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, orientation='horizontal', pad=0.05)
    cbar.set_ticks(np.arange(1.5, 7))
    cbar.set_ticklabels(['状态1\n平静上涨', '状态2\n平静整理', '状态3\n轻微压力', 
                         '状态4\n明显压力', '状态5\n恐慌', '状态6\n恢复'])
    
    # 设置y轴不可见
    ax2.yaxis.set_visible(False)
    ax2.set_yticks([])
    
    # 去除边框
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    ax2.set_xlabel('日期', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    return fig

def plot_state_performance(state_performance_df, filename=None):
    """绘制分市场状态策略表现
    
    Args:
        state_performance_df: 按市场状态分析的绩效DataFrame
        filename: 保存文件名，若为None则不保存
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # 提取返回值列
    return_cols = [col for col in state_performance_df.columns if 'Return' in col]
    returns_df = state_performance_df[return_cols]
    
    # 绘制月均收益柱状图
    returns_df.plot(kind='bar', ax=ax1)
    ax1.set_title('各市场状态下的月均收益 (%)', fontsize=12)
    ax1.set_ylabel('月均收益 (%)', fontsize=12)
    ax1.set_xlabel('市场状态', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.legend(title='策略', loc='best')
    
    # 为柱状图添加数值标签
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f', fontsize=8)
    
    # 绘制市场状态分布饼图
    state_percent = state_performance_df['Percentage(%)'].values
    state_labels = [f"{idx} ({val:.1f}%)" for idx, val in zip(state_performance_df.index, state_percent)]
    
    ax2.pie(state_percent, labels=state_labels, autopct='', startangle=90, 
           colors=plt.cm.viridis(np.linspace(0, 1, len(state_percent))))
    ax2.set_title('市场状态分布 (2015-2023)', fontsize=12)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    return fig

def plot_covid_analysis(covid_df, strategies, filename=None):
    """绘制COVID-19期间策略表现分析图
    
    Args:
        covid_df: 包含COVID-19期间数据的DataFrame
        strategies: 策略列表
        filename: 保存文件名，若为None则不保存
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    covid_start = pd.Timestamp(config.COVID_START)
    covid_end = pd.Timestamp(config.COVID_END)
    recovery_end = pd.Timestamp(config.COVID_RECOVERY_END)
    
    # 提取COVID-19期间的数据
    mask = (covid_df.index >= covid_start) & (covid_df.index <= recovery_end)
    df_covid = covid_df[mask].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
    
    # 绘制VIX和SPX
    ax1_twin = ax1.twinx()
    
    # 绘制VIX
    vix_line, = ax1.plot(df_covid.index, df_covid['vix'], 'r-', label='VIX')
    ax1.set_ylabel('VIX', color='r', fontsize=12)
    ax1.tick_params(axis='y', colors='r')
    
    # 绘制SPX
    spx_line, = ax1_twin.plot(df_covid.index, df_covid['spx'], 'b-', label='S&P500')
    ax1_twin.set_ylabel('S&P500', color='b', fontsize=12)
    ax1_twin.tick_params(axis='y', colors='b')
    
    # 添加标记区域
    ax1.axvspan(covid_start, covid_end, color='red', alpha=0.2, label='危机期')
    ax1.axvspan(covid_end, recovery_end, color='green', alpha=0.2, label='恢复期')
    
    # 设置标题和格式
    ax1.set_title('COVID-19期间市场表现 (2020.02-2020.05)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 合并图例
    lines = [vix_line, spx_line]
    ax1.legend(lines, [line.get_label() for line in lines], loc='upper right')
    
    # 绘制策略累积收益
    norm_date = covid_start - pd.Timedelta(days=5)  # 将启始日提前几天，建立基准
    mask_extended = (covid_df.index >= norm_date) & (covid_df.index <= recovery_end)
    df_extended = covid_df[mask_extended].copy()
    
    # 计算累积收益并归一化
    for name, col in strategies:
        # 从1开始的累积收益
        df_extended[f'{name}_cum'] = (1 + df_extended[col]).cumprod()
        # 归一化到危机开始前
        start_val = df_extended[f'{name}_cum'].iloc[df_extended.index.get_loc(covid_start, method='ffill') - 1]
        df_extended[f'{name}_norm'] = df_extended[f'{name}_cum'] / start_val
        # 绘制线图
        ax2.plot(df_extended.index, df_extended[f'{name}_norm'], label=name, linewidth=2)
    
    # 添加标记区域
    ax2.axvspan(covid_start, covid_end, color='red', alpha=0.2)
    ax2.axvspan(covid_end, recovery_end, color='green', alpha=0.2)
    
    # 设置分割线
    ax2.axvline(x=covid_start, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=covid_end, color='black', linestyle='--', alpha=0.5)
    
    # 设置标题和格式
    ax2.set_title('COVID-19期间策略表现对比', fontsize=14)
    ax2.set_ylabel('归一化累积收益', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right')
    
    # 格式化日期
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    return fig

def plot_weight_transition(df, filename=None):
    """绘制策略权重动态转换图
    
    Args:
        df: 包含策略权重的DataFrame
        filename: 保存文件名，若为None则不保存
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 3]})
    
    # 绘制市场状态
    ax1.scatter(df.index, np.ones(len(df)), c=df['market_state_smooth'], 
               cmap='viridis', marker='s', s=10, vmin=1, vmax=6)
    
    # 设置y轴不可见
    ax1.yaxis.set_visible(False)
    ax1.set_yticks([])
    
    # 去除边框
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    ax1.set_title('市场状态与策略权重动态转换', fontsize=14)
    
    # 添加颜色条
    cmap = plt.cm.get_cmap('viridis', 6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 6))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.1)
    cbar.set_ticks(np.arange(1.5, 7))
    cbar.set_ticklabels(['状态1\n平静上涨', '状态2\n平静整理', '状态3\n轻微压力', 
                         '状态4\n明显压力', '状态5\n恐慌', '状态6\n恢复'])
    
    # 绘制策略权重堆叠区域图
    ax2.fill_between(df.index, 0, df['ts_weight'], label='期限结构策略', alpha=0.7, color='blue')
    ax2.fill_between(df.index, df['ts_weight'], 1, label='ETF对冲策略', alpha=0.7, color='green')
    
    # 标记COVID-19时期
    covid_start = pd.Timestamp(config.COVID_START)
    covid_end = pd.Timestamp(config.COVID_END)
    recovery_end = pd.Timestamp(config.COVID_RECOVERY_END)
    
    ax2.axvspan(covid_start, covid_end, color='red', alpha=0.2, label='COVID-19危机期')
    ax2.axvspan(covid_end, recovery_end, color='blue', alpha=0.1, label='恢复期')
    
    # 设置坐标轴
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('策略权重', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    # 格式化日期
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    return fig

def plot_train_test_comparison(train_df, test_df, filename=None):
    """绘制样本内外性能对比图
    
    Args:
        train_df: 训练集绩效DataFrame
        test_df: 测试集绩效DataFrame
        filename: 保存文件名，若为None则不保存
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    # 选择要展示的指标
    metrics = ['Annual Return(%)', 'Annual Volatility(%)', 'Sharpe Ratio', 'Max Drawdown(%)', 'Win Rate(%)']
    
    # 准备绘图数据
    train_data = train_df.loc[metrics]
    test_data = test_df.loc[metrics]
    
    fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 15))
    
    for i, metric in enumerate(metrics):
        # 提取对应指标的数据
        train_values = train_data.loc[metric]
        test_values = test_data.loc[metric]
        
        # 如果是最大回撤，需要取负值使得更小的回撤显示为更好的表现
        if 'Drawdown' in metric:
            train_values = -train_values
            test_values = -test_values
            metric = metric + ' (负值更好)'
        
        # 创建柱状图
        x = np.arange(len(train_values))
        width = 0.35
        
        axs[i].bar(x - width/2, train_values, width, label='样本内 (2015-2019)', alpha=0.7)
        axs[i].bar(x + width/2, test_values, width, label='样本外 (2020-2023)', alpha=0.7)
        
        # 设置图表格式
        axs[i].set_title(f'{metric}对比', fontsize=12)
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(train_values.index)
        axs[i].grid(True, linestyle='--', alpha=0.5, axis='y')
        
        # 只在第一个子图显示图例
        if i == 0:
            axs[i].legend(loc='upper right')
        
        # 添加数值标签
        for j, v in enumerate(train_values):
            axs[i].text(j - width/2, v + (max(train_values.max(), test_values.max()) * 0.02), 
                      f'{v:.2f}' if 'Ratio' in metric else f'{v:.1f}', 
                      ha='center', va='bottom', fontsize=9)
            
        for j, v in enumerate(test_values):
            axs[i].text(j + width/2, v + (max(train_values.max(), test_values.max()) * 0.02), 
                      f'{v:.2f}' if 'Ratio' in metric else f'{v:.1f}', 
                      ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    return fig

def plot_returns_distribution(df, strategies, filename=None):
    """绘制策略收益分布图
    
    Args:
        df: 包含策略收益的DataFrame
        strategies: 策略列表
        filename: 保存文件名，若为None则不保存
        
    Returns:
        matplotlib.figure.Figure: 图形对象
    """
    fig, axs = plt.subplots(len(strategies), 1, figsize=(12, 4 * len(strategies)))
    
    # 如果只有一个策略，确保axs是列表
    if len(strategies) == 1:
        axs = [axs]
    
    for i, (name, col) in enumerate(strategies):
        # 提取日收益率
        returns = df[col].dropna() * 100  # 转换为百分比
        
        # 绘制直方图和核密度估计
        sns.histplot(returns, bins=50, kde=True, ax=axs[i], color='skyblue')
        
        # 计算收益统计信息
        mean = returns.mean()
        median = returns.median()
        std = returns.std()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        # 添加统计信息文本
        stats_text = (f'均值: {mean:.2f}%   中位数: {median:.2f}%   标准差: {std:.2f}%\n'
                     f'偏度: {skew:.2f}   峰度: {kurt:.2f}')
        axs[i].text(0.02, 0.95, stats_text, transform=axs[i].transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 添加垂直线标记均值和中位数
        axs[i].axvline(mean, color='red', linestyle='--', label=f'均值: {mean:.2f}%')
        axs[i].axvline(median, color='green', linestyle='-.', label=f'中位数: {median:.2f}%')
        axs[i].axvline(0, color='black', linestyle='-')
        
        # 设置坐标轴和标题
        axs[i].set_title(f'{name}日收益分布', fontsize=12)
        axs[i].set_xlabel('日收益率 (%)', fontsize=10)
        axs[i].set_ylabel('频数', fontsize=10)
        axs[i].legend()
        axs[i].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    return fig