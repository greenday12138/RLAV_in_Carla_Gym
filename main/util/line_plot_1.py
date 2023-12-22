import seaborn as sns
import matplotlib.pyplot as plt

# 输入数据
x = [0, 2, 4, 6, 8, 10]
# y1 = [65.00, 64.57, 63.76, 62.49, 62.41, 62.53]
# y2 = [66.35, 66.03, 66.91, 66.31, 67.57, 67.68]
# y3 = [65.21, 65.34, 65.41, 65.75, 65.87, 66.92]
# y4 = [68.73, 69.02, 69.31, 69.25, 69.54, 69.61]
# y5 = [66.87, 66.98, 67.02, 67.07, 68.00, 68.31]

y1 = [0.53, 0.76, 0.91, 1.01, 1.03, 1.02]
y2 = [-0.42, -0.52, -0.62, -0.38, -0.15, -0.32]
y3 = [-0.51, -0.47, -0.55, -0.49, -0.45, -0.48]
y4 = [-0.21, -0.25, -0.24, -0.29, -0.27, -0.26]
y5 = [-0.35, -0.34, -0.37, -0.32, -0.35, -0.36]

# 设置颜色代码
color1 = "#038355" # 孔雀绿
color2 = "#ffc34e" # 向日黄
color3 = 'darkviolet'

# 设置字体
font={'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
plt.rc('font', **font)

# 绘图
#sns.set_style("whitegrid") # 设置背景样式
sns.lineplot(x=x, y=y4, color='tan', linewidth=2.0, marker="P", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='Autopilot')
sns.lineplot(x=x, y=y5, color='cadetblue', linewidth=2.0, marker="X", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='Transfuser')
sns.lineplot(x=x, y=y2, color='darkgoldenrod', linewidth=2.0, marker="s", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='LINE')
sns.lineplot(x=x, y=y3, color='darkviolet', linewidth=2.0, marker="d", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='MA2C')
sns.lineplot(x=x, y=y1, color='green', linewidth=2.0, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='MODUS')

# 添加标题和标签
#plt.title("Title", fontweight='bold', fontsize=14)
plt.xlabel("CAV number", font)
#plt.ylabel("Avg-DT(s)", font)
plt.ylabel("Avg-SOC", font)

# 添加图例
plt.legend(loc='lower left', frameon=True, fontsize=15)

# 设置刻度字体和范围
plt.xticks(fontsize=14)
plt.yticks(fontsize=12)
#plt.ylim(50, 72)
plt.ylim(-1.5, 1.5)

# 设置坐标轴样式
for spine in plt.gca().spines.values():
    spine.set_edgecolor("#CCCCCC")
    spine.set_linewidth(1.5)

plt.savefig('lineplot.png', dpi=300, bbox_inches='tight')
# 显示图像
plt.show()