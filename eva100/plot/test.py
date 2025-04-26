import seaborn as sns
import matplotlib.pyplot as plt

# 示例数据
import pandas as pd
import numpy as np

# 创建示例数据
np.random.seed(0)
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) + np.cos(x)
df = pd.DataFrame({'x': x, 'y1': y1, 'y2': y2, 'y3': y3})

# 使用 palette="Greens" 来绘制多条线
sns.lineplot(x='x', y='y1', data=df, palette='Greens', label='Line 1')
sns.lineplot(x='x', y='y2', data=df, palette='Greens', label='Line 2')
sns.lineplot(x='x', y='y3', data=df, palette='Greens', label='Line 3')
plt.savefig('/home/shijinliang/module/Libra/eva100/plot/kernel_spmm/4090_fp16_128/4090_spmm_fp16_result_1281.png', dpi=800)


