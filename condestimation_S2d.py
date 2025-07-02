
import numpy as np
from scipy.special import comb
from scipy.optimize import nnls

p = 0.876

P_obs = np.array([0.52930649, 0.24871641, 0.09736305, 0.04916116, 0.07545289])

n_counts = 212
obs_counts = (P_obs * n_counts).astype(int)

n_max = len(P_obs)
n_values = np.arange(1, n_max + 1)
m_max = 8
m_values = np.arange(1, m_max + 1)

A = np.zeros((n_max, m_max))
for i, n in enumerate(n_values):
    for j, m in enumerate(m_values):
        if m >= n:
            A[i, j] = comb(m, n) * (p ** n) * ((1 - p) ** (m - n))

n_bootstrap = 212
P_true_boot = []

for _ in range(n_bootstrap):
    resampled_counts = np.random.multinomial(n_counts, P_obs)
    resampled_probs = resampled_counts / n_counts

    P_estimate, _ = nnls(A, resampled_probs)
    P_estimate /= P_estimate.sum()
    P_true_boot.append(P_estimate)

P_true_boot = np.array(P_true_boot)
mean = np.mean(P_true_boot, axis=0)
std = np.std(P_true_boot, axis=0)

# 結果の表示
for m, mu, sigma in zip(m_values, mean, std):
    print(f"m={m}: {mu:.4f} ± {sigma:.4f}")

