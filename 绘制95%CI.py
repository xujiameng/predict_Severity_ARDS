# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:40:47 2022

@author: 佳盟
"""

# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# import scipy.stats as st
matplotlib.rcParams.update({'font.size': 12})

# generate dataset
# data_points = 50
# sample_points = 10000
# Mu = (np.linspace(-5, 5, num=data_points)) ** 2
# Sigma = np.ones(data_points) * 8
# data = np.random.normal(loc=Mu, scale=Sigma, size=(100, data_points))

# # predicted expect and calculate confidence interval
# predicted_expect = np.mean(data, 0)
# low_CI_bound, high_CI_bound = st.t.interval(0.95, data_points - 1,
#                                             loc=np.mean(data, 0),
#                                             scale=st.sem(data))


predicted_expect=np.array([0.8572,0.8278,0.77794,0.7438,0.64113])
low_CI_bound=np.array([0.84990,0.821094,0.76539,0.7325,0.6273])
high_CI_bound=np.array([0.8644,0.83390,0.7888,0.7580, 0.6616])

# plot confidence interval
x = np.linspace(0, 5 - 1, num=5)

plt.plot(predicted_expect, linewidth=3., label='AUC of early warning methods of mild ARDS')
# plt.plot(Mu, color='r', label='grand truth')
plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.5)
                # (label='95% CI with Continuous Non-invasive parameters') #confidence interval




predicted_expect=np.array([0.8077,0.7765,0.7121,0.6908,0.60645])
low_CI_bound=np.array([0.80016,0.7703,0.6968,0.675647,0.5911])
high_CI_bound=np.array([0.81476,0.7826,0.7277,0.70628,0.6191])

plt.plot(predicted_expect, linewidth=3., label='AUC of early warning methods of moderate ARDS')
plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.5)
                # (label='95% CI with Continuous Non-invasive parameters') #confidence interval



predicted_expect=np.array([0.8844,0.86908,0.7843,0.74629,0.7109])
low_CI_bound=np.array([0.87272,0.8588,0.7671,0.7286,0.6889])
high_CI_bound=np.array([0.8951,0.8766,0.8029,0.7658,0.7349])

plt.plot(predicted_expect, linewidth=3., label='AUC of early warning methods of severe ARDS')
plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.5)
                # (label='95% CI with Continuous Non-invasive parameters') #confidence interval



predicted_expect=np.array([0.84147,0.81327,0.75383,0.72394,0.638059])
plt.plot(predicted_expect, linewidth=3., label='AUC of early warning methods of ARDS symptoms')


plt.legend(fontsize=22,loc='lower left')
# plt.title('95% Confidence interval',fontsize=28)


# ax=plt.gca()
# ax.xaxis.set_major_locator(plt.MultipleLocator(1))
values = ['2 hours', '4 hours', '8 hours', '12 hours', '24 hours'] 
plt.xticks(x,values,fontsize=28)
plt.xlabel("Lead time",fontsize=28,fontweight='bold')
plt.yticks(fontsize=28)
plt.ylabel("AUC",fontsize=28,fontweight='bold')
plt.ylim(0.5,0.95)
plt.show()
