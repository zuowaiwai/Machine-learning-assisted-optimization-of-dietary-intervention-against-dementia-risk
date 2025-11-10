# Quick Start Guide

## 运行分析

最简单的方式运行完整分析：

```bash
# 进入src目录
cd src

# 运行简化版分析（推荐）
python3 simple_analysis.py
```

分析会生成：
- 模拟数据（10,000名参与者）
- MODERN和MIND饮食评分
- 统计分析结果
- 3个可视化图表
- 详细的分析报告

## 输出文件位置

所有结果保存在：

```
results/
├── ANALYSIS_REPORT.txt          # 主要分析报告
├── figures/
│   ├── diet_score_distributions.png      # 饮食评分分布
│   ├── incidence_by_tertiles.png         # 按三分位数的痴呆发病率
│   └── odds_ratio_comparison.png         # MODERN vs MIND对比
└── tables/                      # 统计表格

data/
├── simulated/ukb_simulated.csv          # 模拟数据
└── processed/data_with_scores.csv       # 带评分的数据
```

## MODERN饮食评分系统

MODERN饮食由7个组成部分（总分0-7）：

### 充足摄入 (Adequacy)
1. **橄榄油**: >0份/天 = 1分

### 适度摄入 (Moderation)
2. **绿叶蔬菜**: 0-1.5份/天 = 1分
3. **浆果和柑橘**: 0-2份/天 = 1分
4. **土豆**: 0-0.75份/天 = 1分
5. **鸡蛋**: 0-1份/天 = 1分
6. **家禽**: 0-0.5份/天 = 1分

### 限制摄入 (Restriction)
7. **含糖饮料**: 0份/天 = 1分

## 关键发现

基于模拟数据的分析显示：

- **MODERN评分越高 → 痴呆风险越低**
- **MIND评分越高 → 痴呆风险越低**
- 两种饮食模式都显示保护作用
- 在三分位数分析中观察到梯度效应

## 自定义分析

修改参数：

```python
# 在simple_analysis.py中
generator = DataGenerator(n_participants=5000)  # 改变样本量
```

## 使用实际数据

如果你有UK Biobank或类似数据：

```python
from diet_scores import MODERNScore, MINDScore
import pandas as pd

# 加载你的数据
data = pd.read_csv('your_data.csv')

# 计算评分
modern_calc = MODERNScore()
data_with_scores = modern_calc.calculate_dataframe(data)
```

## 论文引用

如果使用此代码，请引用原始论文：

```
Chen, S.J., Chen, H., You, J. et al.
Machine learning-assisted optimization of dietary intervention
against dementia risk. Nat Hum Behav (2025).
https://doi.org/10.1038/s41562-025-02255-w
```

## 注意事项

⚠️ 这是一个教育性MVP实现
⚠️ 使用模拟数据进行演示
⚠️ 实际应用需要真实数据和更全面的验证
⚠️ 临床应用前请咨询医疗专业人员
