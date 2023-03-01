# %%
import numpy as np
import matplotlib.pyplot as plt

# volatility
sigma = 0.25

# maturity
T = 3 # years

# short rate
r = 0.04

# strike price
K = 100

# initial stock price
S0 = 100

# sampling rate
N = np.array([6, 36, 150, 750])

# number of simulations
N_sims = 1000

# geometric brownian motion
def gbm(S0, r, sigma, T, N):
    dt = T / N
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(1, N + 1):
        S[i] = S[i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
    return S

# plot the stock price for all values of N
# plt.figure(figsize=(10, 6))
# for n in N:
#     S = gbm(S0, r, sigma, T, n)
#     plt.plot(np.linspace(0, T, n + 1), S, label='N = {}'.format(n))
# plt.xlabel('time')
# plt.ylabel('stock price')
# plt.legend()
# plt.show()

def f(r, T, K, S0, sigma, N):
    S = gbm(S0, r, sigma, T, N)
    return np.exp(-r * T) * (0.5*(S[0]+S[-1]+2*np.sum(S[1:-1]))*(T/N)/T - K)

def monte_carlo(f, r, T, K, S0, sigma, N, N_sims):
    res = np.zeros(len(N))
    for i, N_i in enumerate(N):
        for _ in range(N_sims):
            res[i] += f(r, T, K, S0, sigma, N_i)
    return res/N_sims

# %%
# get estimate of continuous path with large N
limit = monte_carlo(f, r, T, K, S0, sigma, [10000], N_sims)[0]
# %%
estimates = monte_carlo(f, r, T, K, S0, sigma, N, N_sims)
error = np.abs(estimates - limit)
# %%
# assume error follows N^{-\alpha}, estimate \alpha with linreg
from scipy.stats import linregress
alpha, intercept = linregress(np.log(N), np.log(error))[0:2]
dx = np.linspace(6, max(N), 1000) 
curve = np.exp(intercept) * dx ** alpha
print("alpha = {}".format(alpha))
# %%
# plot error
plt.figure(dpi=150)
plt.plot(N, error, label='observed error')
plt.plot(dx, curve, label=rf"${round(np.exp(intercept),3)}N^{({round(alpha, 3)})}$")
plt.xlabel('N')
plt.legend()
plt.ylabel('error')
# %%
