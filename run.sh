#!/bin/bash
# 波动率风险溢价捕捉系统运行脚本

set -e  # Exit on error

echo "=================================================="
echo "               波动率风险溢价捕捉系统"               
echo "=================================================="

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查Python环境
python3 --version 2>/dev/null || { echo "错误: 请安装Python 3"; exit 1; }

# 检查必要的Python包
echo "检查必要的Python包..."
python3 -c "import pandas, numpy, matplotlib, seaborn, scipy, yfinance" 2>/dev/null || {
    echo "检测到缺少必要的Python包，是否安装? (y/n)"
    read answer
    if [ "$answer" != "${answer#[Yy]}" ]; then
        echo "安装必要的Python包..."
        pip install pandas numpy matplotlib seaborn scipy yfinance
    else
        echo "请手动安装必要的包: pandas, numpy, matplotlib, seaborn, scipy, yfinance"
        exit 1
    fi
}

# 检查更新visualizaiton.py文件以修复字体问题
if [ -f "src/visualization.py" ]; then
    echo "检查visualization.py文件是否需要更新以解决字体问题..."
    if grep -q "SimHei" "src/visualization.py"; then
        echo "发现字体问题，正在更新visualization.py文件..."
        if [ -f "visualization.py.fixed" ]; then
            cp "visualization.py.fixed" "src/visualization.py"
            echo "已更新visualization.py文件以解决字体问题"
        else
            echo "警告: 未找到修复后的visualization.py文件，图形显示可能会有中文字体问题"
        fi
    fi
fi

# 运行主程序
echo "开始运行波动率风险溢价捕捉系统..."
python3 src/main.py "$@"

status=$?
if [ $status -eq 0 ]; then
    echo "程序成功执行，请查看 result 目录下的结果。"
else
    echo "程序执行出错，请检查 log/backtest.log 日志文件获取更多信息。"
fi

echo "=========================运行结束========================="