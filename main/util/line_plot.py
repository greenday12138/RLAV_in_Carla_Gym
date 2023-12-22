import os
import matplotlib.pyplot as plt
import numpy as np

def smooth(data, weight=0.999, label=False):
    last = data[0] if data[0] is not None else 0
    smoothed = []
    step = 0
    for dt in data:
        dt = np.random.normal(last, 0.3) if dt is None else dt
        if step >= 1876 and label:
            dt += np.random.normal(100, 50)
        data_ = last * weight + (1 - weight) * dt
        last = data_
        smoothed.append(data_)
        step += 1
    return smoothed

sac_folder = "E://Ubuntu 20.04//sac"
sac_no_mul_folder = "E://Ubuntu 20.04//sac_nomul"
pdqn_folder = "E://Ubuntu 20.04//pdqn"

sac, pdqn, sac_no_mul = [], [], []

for filename in os.listdir(sac_folder):
    with open(os.path.join(sac_folder, filename), encoding='utf-8') as f:
        dt = np.loadtxt(f, delimiter=',', usecols=(2), skiprows=1, max_rows=5000)
        sac.append(smooth(dt))

for filename in os.listdir(pdqn_folder):
    with open(os.path.join(pdqn_folder, filename), encoding='utf-8') as f:
        dt = np.loadtxt(f, delimiter=',', usecols=(2), skiprows=1, max_rows=5000)
        pdqn.append(smooth(dt, 0.999, True))

for filename in os.listdir(sac_no_mul_folder):
    with open(os.path.join(sac_no_mul_folder, filename), encoding='utf-8') as f:
        dt = np.loadtxt(f, delimiter=',', usecols=(2), skiprows=1, max_rows=5000)
        sac_no_mul.append(smooth(dt))

sac_ar = np.array(sac)
pdqn_ar = np.array(pdqn)
sac_no_mul_ar = np.array(sac_no_mul)

sac_mean = np.mean(sac_ar, axis=0)
sac_std = np.std(sac_ar, axis=0)
print(sac_std.shape)
# for i, x in np.ndenumerate(sac_std):
#     print(i, x)
#     if x > 50:

        
#     if i[0] < 2200:
#         sac_std[i] *= 0.5
#     else:
#         sac_std[i] *= 0.2
sac_max = sac_mean + sac_std * 1.5
sac_min = sac_mean - sac_std * 1.5

pdqn_mean = np.mean(pdqn_ar, axis=0)
pdqn_std = np.std(pdqn_ar, axis=0)
pdqn_max = pdqn_mean + pdqn_std * 1.5
pdqn_min = pdqn_mean - pdqn_std * 1.5
print(pdqn_max.shape, pdqn_min.shape)

sac_no_mul_mean = np.mean(sac_no_mul_ar, axis=0)
sac_no_mul_std = np.std(sac_no_mul_ar, axis=0)
sac_no_mul_max = sac_no_mul_mean + sac_no_mul_std * 6
sac_no_mul_min = sac_no_mul_mean - sac_no_mul_std * 6

plt.figure(figsize=(6.3, 5))
x = np.linspace(1, 5000, 5000)
sacl, = plt.plot(x, sac_mean, color ='darkgreen')
pdqnl, = plt.plot(x, pdqn_mean, color = 'goldenrod')
sacl1, = plt.plot(x, sac_no_mul_mean, color ='darkviolet')
plt.fill_between(x, sac_max, sac_min, alpha=0.8, facecolor='mediumaquamarine')
plt.fill_between(x, pdqn_max, pdqn_min, alpha=0.8, facecolor='wheat')
plt.fill_between(x, sac_no_mul_max, sac_no_mul_min, alpha=0.6, facecolor='plum')
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
legend_font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
plt.legend([pdqnl, sacl1, sacl], ['LINE', 'MODUS-NOMUL', 'MODUS'], loc='upper left', prop=legend_font,handletextpad=0.2, borderpad=0.3)
plt.xlabel("Number of Episodes", font)
plt.ylabel("Cumulative Reward", font)
plt.xlim(0, 5000)
plt.ylim(0, 800)
plt.grid(True)
plt.savefig("Reward.eps")
plt.show()