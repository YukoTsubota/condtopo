# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 18:15:05 2025

@author: cooyo
"""

import numpy as np
from scipy.special import comb
from scipy.optimize import nnls

# 蛍光標識効率
p = 0.86

# 観測された分子相当数の分布 (n = 1 〜 5)
P_obs = np.array([0.52930649, 0.24871641, 0.09736305, 0.04916116, 0.07545289])


# nの範囲（観測された分子数）
n_max = len(P_obs)
n_values = np.arange(1, n_max + 1)

# 仮定する実分子数の最大値（5〜8程度を試す。ここでは8まで）
m_max = 8
m_values = np.arange(1, m_max + 1)

# 畳み込み行列 A を構築：A[n-1, m-1] = P(n観測 | m実在)
A = np.zeros((n_max, m_max))
for i, n in enumerate(n_values):
    for j, m in enumerate(m_values):
        if m >= n:
            A[i, j] = comb(m, n) * (p ** n) * ((1 - p) ** (m - n))

# 非負最小二乗法で P_true を推定
P_true, _ = nnls(A, P_obs)

# 正規化（検出されたものに限っているので、合計が1になるはず）
P_true /= P_true.sum()

P_true_results = dict(zip(m_values, P_true))
P_true_results

import numpy as np
from scipy.special import comb
from scipy.optimize import nnls

# 蛍光標識効率
p = 0.876

# 観測された分子相当数の分布
P_obs = np.array([0.52930649, 0.24871641, 0.09736305, 0.04916116, 0.07545289])

# 仮定：元の観測数（例えば10000カウントから割合を得たとする）
n_counts = 212
obs_counts = (P_obs * n_counts).astype(int)

# 観測範囲（1〜5分子）と仮定する実分子数（1〜8分子）
n_max = len(P_obs)
n_values = np.arange(1, n_max + 1)
m_max = 8
m_values = np.arange(1, m_max + 1)

# 畳み込み行列 A を構築：A[n-1, m-1] = P(n観測 | m実在)
A = np.zeros((n_max, m_max))
for i, n in enumerate(n_values):
    for j, m in enumerate(m_values):
        if m >= n:
            A[i, j] = comb(m, n) * (p ** n) * ((1 - p) ** (m - n))

# ブートストラップ
n_bootstrap = 212
P_true_boot = []

for _ in range(n_bootstrap):
    # 観測データを再サンプリング（多項分布）
    resampled_counts = np.random.multinomial(n_counts, P_obs)
    resampled_probs = resampled_counts / n_counts

    # NNLS 解
    P_estimate, _ = nnls(A, resampled_probs)
    P_estimate /= P_estimate.sum()
    P_true_boot.append(P_estimate)

# 結果の配列化
P_true_boot = np.array(P_true_boot)
mean = np.mean(P_true_boot, axis=0)
std = np.std(P_true_boot, axis=0)

# 結果の表示
for m, mu, sigma in zip(m_values, mean, std):
    print(f"m={m}: {mu:.4f} ± {sigma:.4f}")

