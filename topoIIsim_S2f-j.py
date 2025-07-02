# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 16:50:58 2025

@author: cooyo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# パラメータ
labeling_efficiency = 0.625
mean_intensity = 30.73443503
var_intensity = 67.67236495
n_samples = 10000  # 総サンプル数

# n=2とn=6の両方の分子数について処理
molecule_cases = [2, 4, 6, 8, 14,]

ratio_n2 = 0
ratio_n4 = 1
ratio_n6 = 0
ratio_n8 = 0
ratio_n10 =0
case_sample_counts = [int(n_samples * ratio_n2), int(n_samples * ratio_n4), int(n_samples * ratio_n6), int(n_samples * ratio_n8), int(n_samples * ratio_n10)]


all_intensities = []

for n_molecules, case_samples in zip(molecule_cases, case_sample_counts):
    # 各ラベル数の確率（0～n_molecules）
    label_probs = binom.pmf(range(n_molecules + 1), n_molecules, labeling_efficiency)

    # サンプル数をラベル数の確率に比例させて分配
    sample_counts = (label_probs * case_samples).astype(int)

    for k, count in enumerate(sample_counts):
        if count == 0:
            continue
        mean = k * mean_intensity
        std = np.sqrt(k * var_intensity) if k > 0 else 1e-3  # k=0 のときは微小なstdで固定
        samples = np.random.normal(loc=mean, scale=std, size=count)
        all_intensities.extend(samples)

# ヒストグラムを描画
plt.figure(figsize=(4, 3.2))
bin_edges = np.arange(0, 240 + 1, 5)

plt.hist(all_intensities, bins=bin_edges, density=True, alpha=0.7, color='skyblue')
plt.xticks(np.arange(0, 240, 30))
plt.xlim(0, 210)
plt.yticks(np.arange(0, 0.035, 0.005))
plt.ylim(0, 0.035)
plt.xlabel("Fluorescence Intensity")
plt.ylabel('Relative frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig('Simulated_topo_Histogram_0_100.png', dpi=600)
plt.show()

