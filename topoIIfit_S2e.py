
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# データ
data_single = np.array([
    14.3433848, 29.381408, 19.1272884, 14.17404, 37.4404736, 34.497905, 39.24704, 34.7496, 28.3521608,
    27.4302728, 38.035413, 44.1244445, 38.9588772, 17.11681685, 31.013843, 41.8823438, 37.944044,
    47.7893888, 32.4597402, 19.486912, 34.4271625, 48.2040708, 30.0095642, 44.5471044, 37.1771754,
    22.428231, 21.7349616, 27.636724, 30.7033328, 29.1105915, 31.0502192, 33.236736, 30.4409236,
    27.4940116, 42.2913854, 42.54732, 16.7825928, 33.615132, 22.2942912, 47.920444, 24.3373554,
    46.9708512, 50.4266231, 32.4591176, 18.2702085, 30.1858495, 32.0927258, 24.834254, 31.4237902,
    26.8711368, 20.592759, 29.990288, 21.8419201, 38.6528624, 27.37672, 37.211832, 23.1087168,
    37.4272558, 32.525064, 35.951154, 32.0033664, 29.0818944, 22.355153, 27.7217632, 30.4513724,
    35.199714, 26.6641573, 31.2167068, 29.541564, 35.5809152, 23.5937292, 20.6611325, 32.071818,
    23.687681, 18.3657936, 25.610439, 30.9751916, 26.253398, 28.1096296, 25.851528
])
data_exp = np.array([71.82024,37.071152,24.54816,68.49075,66.8044,101.593065,26.297013,46.193459,39.5955,54.091856,67.193771,14.887223,56.699552,31.048199,19.29249,22.27349,28.791932,72.1404,23.195536,59.243964,60.82386,5.73955199999997,30.43601,82.282375,50.80104,34.53624,44.086224,29.053416,33.5328,-0.8805,32.353335,47.255625,54.802384,57.892104,21.973455,59.391288,51.24016,21.8155,25.025571,217.531635,21.087865,51.501632,41.47926,58.705375,28.68411,0.389864999999975,70.66696,41.751916,57.545488,60.551156,64.181936,60.78801,31.588545,35.467441,66.54676,67.832955,70.745412,80.623323,102.585798,62.82159,180.612432,5.01443599999999,50.133916,46.4765,33.220295,18.183984,57.158192,22.65009,2.709672,80.6985,-2.76057599999998,40.91568,17.86974,36.471638,28.118865,61.425,21.88325,30.456291,30.695392,20.902246,40.412736,60.198,7.533045,25.644608,47.812128,50.057955,2.20051199999998,-6.770896,27.7775,72.035865,63.837135,46.609684,5.80650000000001,40.328288,54.80975,73.732065,42.360552
])

mean = np.mean(data_single)
var = np.var(data_single)

n_components = 5
fixed_means = np.array([(i+1) * mean for i in range(n_components)])
fixed_vars = np.array([(i+1) * var for i in range(n_components)])

from scipy.optimize import minimize

x_smooth = np.linspace(min(data_exp), max(data_exp), 1000)

hist_y, hist_x = np.histogram(data_exp, bins=100, density=True)
hist_xc = 0.5 * (hist_x[:-1] + hist_x[1:])  # ビンの中心

pdf_matrix = np.array([
    norm.pdf(x_smooth, loc=mu, scale=np.sqrt(v)) for mu, v in zip(fixed_means, fixed_vars)
]).T  # shape = (len(hist_xc), n_components)

def loss(weights):
    model = pdf_matrix @ weights
    model_resampled = np.interp(hist_xc, x_smooth, model)
    return np.sum((hist_y - model_resampled) ** 2)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1)] * n_components
initial_weights = np.ones(n_components) / n_components

result = minimize(loss, initial_weights, constraints=constraints, bounds=bounds)
fitted_weights = result.x
model_pdf = pdf_matrix @ fitted_weights

x = np.linspace(min(data_exp), max(data_exp), 1000)
plt.figure(figsize=(4, 3.2))
bin_edges = np.arange(0, 240+ 1, 5)  # 0から30まで5刻み
plt.hist(data_exp, bins=bin_edges, density=True, alpha=0.2, color = '#FF00FF', label='Experimental Data')
for i, (mu, v, w) in enumerate(zip(fixed_means, fixed_vars, fitted_weights)):
    plt.plot(
        x_smooth, 
        w * norm.pdf(x_smooth, loc=mu, scale=np.sqrt(v)), 
        '--', 
        label=f'{i+1}-Molecule (w={w:.2f})')
plt.xticks(np.arange(0, 240, 30))
plt.xlim(0, 210)
plt.yticks(np.arange(0, 0.035, 0.005))
plt.ylim(0, 0.035)
plt.plot(x_smooth, model_pdf, 'k-', label='Total Fit')
plt.legend(handlelength=1.5, fontsize=9.5)
plt.xlabel('Topoisomerase IIα intensity on lump (a.u.)')
plt.ylabel('Relative frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
