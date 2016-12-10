import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt('results/_300s_1sig_entropy/y_train_stats.csv')

center = .5
plt.hist(x, label='SuperPixel Skin Ratio')
#plt.bar(center, x, align='center')

plt.title('Ratio of Skin Pixels in SuperPixels')
# plt.plot(false_positive_rate, true_positive_rate, '-o', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='upper right', fontsize='small')
plt.ylabel('Pixel Count')
plt.xlabel('SuperPixel Skin Ratio')
plt.grid(which='both')

plt.show()

