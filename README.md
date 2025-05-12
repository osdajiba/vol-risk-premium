# 波动率风险溢价捕捉系统

## 项目概述
波动率风险溢价捕捉系统是一个基于Python开发的量化交易策略框架，通过动态组合期限结构交易策略和ETF对冲策略，实现在不同市场环境下捕捉波动率风险溢价的目标。

## 功能特性
- **双策略协同**: 期限结构策略+ETF对冲策略动态组合
- **市场状态识别**: 自动识别6种市场状态并优化策略权重
- **全面分析**: 提供详细的绩效指标和市场状态分析
- **可视化展示**: 生成直观的策略表现图表
- **样本内外测试**: 支持训练集/测试集分离验证
- **参数化配置**: 所有关键参数均可配置调整

## 快速开始

### 系统要求
- Python 3.8+
- 依赖包: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `yfinance`

### 运行系统
```bash
./run.sh    # 自动安装项目依赖
```
默认运行2015-2023年的回测，结果保存在`result`目录。

## 使用说明

### 基本命令
| 命令 | 说明 |
|------|------|
| `python main.py` | 默认回测(2015-2023) |
| `python main.py --start 2020-01-01 --end 2022-12-31` | 自定义回测周期 |
| `python main.py --production` | 生产模式(启用所有优化) |
| `python main.py --no_plots` | 只运行回测不生成图表 |

### 数据选项
| 选项 | 说明 |
|------|------|
| `--force_download` | 强制重新下载数据 |
| `--no_cache` | 不使用任何缓存数据 |
| `--local_file path/to/data.csv` | 使用本地数据文件 |

## 输出结果
系统运行完成后会在`result`目录生成以下文件:

### 数据文件
- `backtest_data.csv` - 完整回测数据
- `performance_comparison.csv` - 策略绩效对比
- `state_performance.csv` - 分市场状态绩效

### 图表文件
- `strategy_performance.png` - 策略累积收益
- `market_states.png` - 市场状态分布
- `state_performance.png` - 分状态表现
- `covid_analysis.png` - COVID-19期间分析
- `weight_transition.png` - 策略权重变化
- `train_test_comparison.png` - 样本内外对比
- `returns_distribution.png` - 收益分布

## 配置调整
主要参数可在`src/config.py`中修改:

```python
# 市场状态阈值
VIX_THRESHOLDS = [15, 20, 25, 30]

# 策略参数
TS_STRATEGY = {
    'target_vol': 0.15,  # 目标波动率
    'max_leverage': 3.0  # 最大杠杆
}

# 交易成本
FUTURES_COST = 0.0005  # 期货交易成本
ETF_COST = 0.0010      # ETF交易成本
```

## 典型输出示例
```
==== 策略绩效对比 ====
                     期限结构策略    ETF对冲策略     动态策略选择
Annual Return(%)        37.88        -1.63        52.40
Sharpe Ratio            0.77         0.25         1.18
Max Drawdown(%)       -49.24       -82.60       -31.07
Win Rate(%)            49.11        56.67        56.43

==== COVID-19期间分析 ====
                         Crisis Period      Recovery Period
期限结构策略 Return(%)           118.20            19.40
ETF对冲策略 Return(%)          -76.25            39.38
动态策略选择 Return(%)          -13.98            26.29
```

## 技术支持
遇到问题时请提供:
1. `log/backtest.log`日志文件
2. 使用的参数配置
3. 问题重现步骤

更多详细信息请参考项目文档或联系我们的技术支持团队：2354889815@qq.com

## 贡献说明
我们欢迎社区贡献！如果您有任何改进建议或发现了bug，请提交issue或pull request。

## 版权声明
© 2025 Jacky。保留所有权利。
本项目采用MIT许可证授权，详情请见项目根目录下的`LICENSE`文件。

## 联系作者
- 项目所有者: Jacky
- 邮箱: 2354889815@qq.com
- 项目主页: git@github.com:osdajiba/vol-risk-premium.git