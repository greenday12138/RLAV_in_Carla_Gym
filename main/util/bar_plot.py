import matplotlib.pyplot as plt
import numpy as np

# 数据

perturbation = ['0', '5', '10', '15']
MODUS = [4, 10, 13, 15]
transfuser = [7, 12, 15.1, 16.5]
LINE = [6, 11, 14, 15.9]
MA2C = [8.5, 14, 16.3, 17.7]
autopilot = [4.5, 14.5, 17, 18]

font = {'family': "Times New Roman", 'weight': "normal", 'size': 18}
legend_font = {'family': "Times New Roman", 'weight': "normal", 'size': 15}

barWidth = 0.13

r1 = np.arange(len(LINE))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
plt.figure(figsize=(6.3, 5))
# 创建条形图

plt.bar(r1, autopilot, width=barWidth, hatch='/',  color='tan', label='Autopilot')
plt.bar(r2, LINE, width=barWidth, color='darkgoldenrod', hatch='x', label='LINE')
plt.bar(r3, MA2C, width=barWidth,  hatch='*', color='darkviolet', label='MA2C')
plt.bar(r4, transfuser, width=barWidth,  hatch='+', color='cadetblue', label='TransFuser')
plt.bar(r5, MODUS, width=barWidth,  hatch='\\',  color='green', label='MODUS')
plt.figure(1)
# 添加标签，标题和图例
plt.xlabel('Sensor Data Perturbation (%)',fontdict=font)
plt.ylabel('PT-RT (%)',fontdict=font)
plt.ylim((0, 30))
# plt.title('PRT(%) under different weather conditions')
plt.xticks([r + 2*barWidth for r in range(len(LINE))], perturbation,fontproperties='Times New Roman', size=16)
plt.yticks(fontproperties='Times New Roman', size=16)
plt.legend(loc = 2, ncol=2,frameon=False, prop=legend_font,handletextpad=0.8, borderpad=0.3)#图例

# 显示图形
# plt.tight_layout()
plt.savefig('weather.png',dpi=600, bbox_inches='tight')
plt.show()