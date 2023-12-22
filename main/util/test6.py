import matplotlib.pyplot as plt
import numpy as np

# 数据

weather_conditions = ['Normal', 'Rain', 'Fog', 'Night']
acc_lc = [4.5, 16.5, 22, 9.4]
transfuser = [7, 12, 15.1, 5]
drl_mf = [8.5, 14.5, 17, 6]
head = [5.8, 13, 15.7, 5.7]
think_twice = [6, 11, 14, 4.5]
auto_nohis = [2, 10, 12, 3.5]
auto = [1.3, 7, 10, 3]

font = {'family': "Times New Roman", 'weight': "normal", 'size': 20}
legend_font = {'family': "Times New Roman", 'weight': "normal", 'size': 16}

barWidth = 0.13
r1 = np.arange(len(acc_lc))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]
plt.figure(figsize=(6.3, 5))
# 创建条形图
plt.bar(r1, acc_lc, width=barWidth, hatch='/',  color='tan', label='ACC-LC')
plt.bar(r2, transfuser, width=barWidth, hatch='.', color='y', label='TransFuser')
plt.bar(r3, drl_mf, width=barWidth, color='darkgoldenrod', hatch='x', label='DRL-MF')
plt.bar(r4, head, width=barWidth,  hatch='+', color='cadetblue', label='HEAD')
plt.bar(r5, think_twice, width=barWidth,  hatch='*', color='darkviolet', label='ThinkTwice')
plt.bar(r6, auto_nohis, width=barWidth,  hatch='\\',  color='green', label='AUTO-NoHIS')
plt.bar(r7, auto, width=barWidth, color='gray', label='AUTO')

# 添加标签，标题和图例
plt.xlabel('Weather Conditions',fontdict=font)
plt.ylabel('PT-RT (%)',fontdict=font)
plt.ylim((0, 30))
# plt.title('PRT(%) under different weather conditions')
plt.xticks([r + 3*barWidth for r in range(len(acc_lc))], weather_conditions,fontproperties='Times New Roman', size=16)
plt.yticks(fontproperties='Times New Roman', size=16)
plt.legend(loc = 2, ncol=2,frameon=False, prop=legend_font,handletextpad=0.8, borderpad=0.3)#图例

# 显示图形
plt.tight_layout()
plt.savefig('weather.png',dpi=600, bbox_inches='tight')
plt.show()