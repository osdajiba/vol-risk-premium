"""
波动率风险溢价捕捉系统配置文件
"""

# 回测参数
START_DATE = '2015-01-01'
END_DATE = '2023-12-31'

# 数据API参数
API_KEY = ""  # 如需使用付费API，可在此处填写

# 数据获取参数
DATA_CACHE_DIR = 'data'             # 数据缓存目录
CACHE_MAX_AGE_DAYS = 1              # 缓存最大有效期(天)
DATA_MAX_RETRIES = 5                # 数据下载最大重试次数
DATA_MIN_DELAY = 2                  # 数据下载最小延迟(秒)
DATA_MAX_DELAY = 10                 # 数据下载最大延迟(秒)
USE_ALTERNATIVE_SOURCES = True      # 是否使用备用数据源
SYNTHETIC_DATA_ENABLED = False      # 禁用合成数据(仅使用真实数据)

# 结果输出目录
RESULT_DIR = 'result'               # 结果保存目录
LOG_DIR = 'log'                     # 日志保存目录

# 交易成本参数
FUTURES_COST = 0.0005  # 期货交易成本0.05%
ETF_COST = 0.001       # ETF交易成本0.1%
SLIPPAGE = 0.0005      # 滑点0.05%
SHORT_COST = 0.015/252 # 做空成本1.5%年化
MAX_LEVERAGE = 2.0     # 最大杠杆2倍

# 策略参数
TARGET_VOL = 0.15      # 目标波动率15%

# 市场状态分类参数
VIX_LOW_THRESHOLD = 15
VIX_MID_THRESHOLD = 25
TS_LOW_THRESHOLD = 0.97
TS_HIGH_THRESHOLD = 1.03
TREND_STRENGTH = 0.02  # 趋势强度：当价格偏离MA超过2%时确认趋势
SMOOTH_WINDOW = 3      # 市场状态平滑窗口大小

# 动态策略选择权重
STATE_WEIGHTS = {
    1: (0.2, 0.8),  # 状态1：20% TS策略，80% ETF策略
    2: (0.3, 0.7),  # 状态2：30% TS策略，70% ETF策略
    3: (0.5, 0.5),  # 状态3：50% TS策略，50% ETF策略
    4: (0.7, 0.3),  # 状态4：70% TS策略，30% ETF策略
    5: (0.9, 0.1),  # 状态5：90% TS策略，10% ETF策略
    6: (0.6, 0.0)   # 状态6：60% TS策略，0% ETF策略
}

# 权重调整速度
MAX_DAILY_WEIGHT_CHANGE = 0.1  # 每日最多调整10%

# COVID-19研究期间
COVID_START = '2020-02-19'
COVID_END = '2020-03-23'
COVID_RECOVERY_END = '2020-05-31'

# 样本外测试分割点
TRAIN_TEST_SPLIT = '2020-01-01'

# 风险管理参数
VIX_SPIKE_THRESHOLD_1 = 10  # VIX日涨幅超10%减仓50%
VIX_SPIKE_THRESHOLD_2 = 20  # VIX日涨幅超20%全部平仓

# 日志配置
LOG_LEVEL = 'INFO'  # 可选：DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'log/backtest.log'  # 日志文件路径
LOG_TO_CONSOLE = True  # 是否同时输出到控制台

# 检查文件路径配置
DEFAULT_DATA_FILES = {
    'vix': 'data/vix_data.csv',      # 本地VIX数据文件
    'spx': 'data/spx_data.csv',      # 本地SPX数据文件
    'vxx': 'data/vxx_data.csv'       # 本地VXX数据文件
}

# 默认合并数据文件
DEFAULT_COMBINED_DATA = 'data/combined_market_data.csv'  # 默认合并数据文件