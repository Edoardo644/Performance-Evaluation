'''
Random Number Generation
1. Using the procedure that generates random numbers uniformly distributed in the range
    [0,1) of your chosen programming language, generate N = 10000 samples of the following 
    distributions:
        a. An Exponential distribution of rate lamda = 0.25
        b. A Pareto distribution with parameters a = 2.5, m = 3
        c. An Erlang distribution with k = 8, and lamba = 0.8
        d. A Hypo-Exponential distribution with rates lamba 1 = 0.25, lambda 2 = 0.4
        e. A Hyper-Exponential distribution with rates lambda 1 = 1, lambda 2 = 0.05, p1 = 0.75
2. For each distribution, compare a plot the Empirical distribution obtained from the samples,
    with the corresponding real distribution (using its formula).
    Please choose a range on the x-axis to make the evolution of the distribution
    clearly visible, and show only one distribution per figure.
'''

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
import scipy.optimize as opt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")

#Alternative way to generate samples : Linear Congruential Generator
"""
    m = 2*31 - 1
a = 1103515245
c = 12345
x0 = 123456  # Seed

def lcg(m, a, c, seed, N):
    numbers = []
    x = seed
    for _ in range(N):
        x = (a x + c) % m
        numbers.append(x / m)
    return np.array(numbers)
    
    
u1 = lcg(m, a, c, x0, N)
u2 = lcg(m, a, c, x0 + 1, N)
    """

#Generation of random samples and plotting
N = 10000
yp = np.r_[1:N+1] / N
t = np.r_[0:80:1000j]

#u = np.random.rand(N)
#2 pseudo-random variables
u1 = np.random.uniform(0, 1, N)
u2 = np.random.uniform(0, 1, N)

# Exponential distribution

lambdaExp = 0.25
XExp = - np.log(1 - u1) / lambdaExp
XExp.sort(0)
FExpReal = 1 - np.exp(-lambdaExp * t)

plt.plot(XExp, yp, ".", label="Empirical")
plt.plot(t, FExpReal, label="Real")
plt.legend()
plt.xlabel("Time")
plt.ylabel("CDF")
plt.grid(True)
plt.title("Comparison of Empirical and Real Exponential CDFs")
plt.show()


# Pareto distribution

aPareto = 2.5
mPareto = 3
XPareto = mPareto / (u1 ** (1 / aPareto))
XPareto.sort(0)
FParetoReal = np.where(t >= mPareto, 1 - (mPareto / t) ** aPareto, 0)

plt.plot(XPareto, yp, ".", label="Empirical")
plt.plot(t, FParetoReal, label="Real")
plt.legend()
plt.xlabel("Time")
plt.ylabel("CDF")
plt.grid(True)
plt.title("Comparison of Empirical and Real Pareto CDFs")
plt.show()


# Erlang distribution

kErlang = 8
lambdaErlang = 0.8
XErlang = np.sum(-np.log(np.random.rand(N, kErlang)) / lambdaErlang, axis=1)
XErlang.sort(0)
FErlangReal = 1 - np.array([np.sum(np.fromiter((np.power(lambdaErlang * t, i) * np.exp(-lambdaErlang * t) / math.factorial(i) for i in range(0, kErlang)), dtype=float)) for t in t])

plt.plot(XErlang, yp, ".", label="Empirical")
plt.plot(t, FErlangReal, label="Real")
plt.legend()
plt.xlabel("Time")
plt.ylabel("CDF")
plt.grid(True)
plt.title("Comparison of Empirical and Real Erlang CDFs")
plt.show()


# Hypo-Exponential distribution

lambda1Hypo = 0.25
lambda2Hypo = 0.4
XHypo = -(np.log(u1) / lambda1Hypo) - (np.log(u2) / lambda2Hypo)
XHypo.sort(0)
FHypoReal = 1 - 1 / (lambda2Hypo - lambda1Hypo) * (lambda2Hypo * np.exp(-lambda1Hypo * t) - lambda1Hypo * np.exp(-lambda2Hypo * t))

plt.plot(XHypo, yp, ".", label="Empirical")
plt.plot(t, FHypoReal, label="Real")
plt.legend()
plt.xlabel("Time")
plt.ylabel("CDF")
plt.grid(True)
plt.title("Comparison of Empirical and Real Hypo-Exponential CDFs")
plt.show()


# Hyper-Exponential distribution

lambda1Hyper = 1
lambda2Hyper = 0.05
p1Hyper = 0.75
XHyper = -np.log(u1) / np.where(u2 < p1Hyper, lambda1Hyper, lambda2Hyper)
XHyper.sort(0)
FHyperReal = 1 - p1Hyper * np.exp(-lambda1Hyper * t) - (1 - p1Hyper) * np.exp(-lambda2Hyper * t)

plt.plot(XHyper, yp, ".", label="Empirical")
plt.plot(t, FHyperReal, label="Real")
plt.legend()
plt.xlabel("Time")
plt.ylabel("CDF")
plt.grid(True)
plt.title("Comparison of Empirical and Real Hyper-Exponential CDFs")
plt.show()

