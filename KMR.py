import numpy as np
import matplotlib.pyplot as plt
import random
from random import uniform
from __future__ import division
from mc_tools import mc_compute_stationary, mc_sample_path
from discrete_rv import DiscreteRV

profit = np.array([[[4, 4], [0, 3]], [[3, 0], [2, 2]]]) # 利得
profit_A = np.array([[profit[0, 0, 0], profit[0, 1, 0]], [profit[1, 0, 0], profit[1, 1, 0]]]) #プレーヤーAの利得(4,0,3,1)
N = 1000 # プレーヤー数
times = 100000 # 試行回数
epsilon = 0.1 # 突然変異確率
x_0 = int(uniform(0, N))  # x_t　ｔ期における行動１をとる人数

p = np.zeros((N+1, N+1)) #(n+1)×(n+1)の行列　まずは0でうめる←ほとんどのマスが０だから
#1人増加or減少、変化なしのマスを埋めていく


for i in range(1, N+1): #行動１→行動０への変更
    m_i = i / N  #行動 1 をとっているプレイヤーとマッチする確率,行動１の人が選ばれる確率
    ratio = np.array([1-m_i, m_i])
    exp = np.dot(profit_A, ratio)
    if exp[0] > exp[1]: #行動０が最適反応のとき
        p[i][i-1] = m_i*((1.0-epsilon) + epsilon*0.5)     # p[i][i-1] x_0が１減少するマスの確率
    else:
        p[i][i-1] = m_i*(epsilon*0.5)
for i in range(N): #行動０→行動１への変更
    m_i = i / N
    ratio = np.array([1-m_i, m_i])
    exp = np.dot(profit_A, ratio)
    if exp[0] < exp[1]:
        p[i][i+1] = (1-m_i)*((1-epsilon) + epsilon*0.5)
    else:
        p[i][i+1] = (1-m_i)*(epsilon*0.5)
for i in range(1, N): #行動を変えない
    p[i][i] = 1 - p[i][i-1] - p[i][i+1]

#上からもれたマス
p[0][0] = 1 - p[0][1]
p[N][N] = 1 - p[N][N-1]
    
    
    
    
X = mc_sample_path(p, x_0, times)   #マルコフ連鎖
plt.plot(X)
plt.show()