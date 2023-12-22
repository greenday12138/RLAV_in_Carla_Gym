# 导入第三方模块
import numpy as np
import matplotlib.pyplot as plt

# 设置中文 雅黑
plt.rcParams['font.sans-serif'] = ['SimHei']

# 构造数据
values = [3.2, 2.1, 3.5, 2.8]
values2 = [4, 4.1, 4.5, 4]
feature = ['Safety','Efficiency','Comfort','Social Impact']

N = len(values)

# 设置雷达图的角度，用于平分切开一个圆面
angles=np.linspace(0, 2*np.pi, N, endpoint=False)

# 将雷达图中的折线图封闭
values=np.concatenate((values,[values[0]]))
values2=np.concatenate((values2,[values2[0]]))
angles=np.concatenate((angles,[angles[0]]))
feature=np.concatenate((feature,[feature[0]]))

# 绘图
fig=plt.figure(figsize=(20,8),dpi=80)
ax = fig.add_subplot(111, polar=True)

# 绘制折线图
ax.plot(angles, values, 'o-', linewidth=2, label = '活动前')

# 填充颜色
ax.fill(angles, values, alpha=0.25)

# 绘制第二条折线图
ax.plot(angles, values2, 'o-', linewidth=2, label = '活动后')
ax.fill(angles, values2, alpha=0.25)

# 添加每个特征的标签
ax.set_thetagrids(angles*180/np.pi, feature)

# 设置雷达图的范围
ax.set_ylim(0, 2)

# 添加标题
# plt.title('活动前后员工状态表现')

# 添加网格线
ax.grid(True)

# 设置图例
plt.legend(loc = 'best')

plt.savefig("radar.pdf")

# 显示图形
plt.show()