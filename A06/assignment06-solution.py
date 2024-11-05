import numpy as np
import matplotlib.pyplot as plt

def hyper_exponential(lambda_1, lambda_2, prob_1, number_of_samples):
    u = np.random.rand(number_of_samples)
    Xhyper = np.where(np.random.rand(number_of_samples) < prob_1,
                    -np.log(1 - u) / lambda_1,
                    -np.log(1 - u) / lambda_2)
    return Xhyper

def erlang(lambda_erlang, k_erlang, number_of_samples):
    Xerlang = np.zeros(number_of_samples)
    for i in range(number_of_samples):
        Xerlang[i] = np.sum(-np.log(1 - np.random.rand(int(k_erlang))) / lambda_erlang)
    return Xerlang

def uniform(a, b, number_of_samples):
    u = np.random.rand(number_of_samples)
    Xuniform = a + (b - a) * u
    return Xuniform

def weibull(lambda_weibull, k_weibull, number_of_samples):
    u = np.random.rand(number_of_samples)
    Xweibull = lambda_weibull * (-np.log(1 - u)) ** (1 / k_weibull)
    return Xweibull

# This dictionary will return the corresponding d_gamma given a confidence level
gamma_dict = {0.99: 2.576, 0.98: 2.326, 0.95: 1.96, 0.9: 1.645}

required_confidence_interval = 0.95
d_gamma = gamma_dict[required_confidence_interval]

M = 5000
K0 = 50
DeltaK = 10
MaxK = 20000

maxRelErr = 0.02


# SCENARIO 1

U_1 = 0
U_2 = 0

X_1 = 0
X_2 = 0

N_1 = 0
N_2 = 0

R_1 = 0
R_2 = 0

K = K0
Krange = K

while K < MaxK:
    for k in range(0, Krange):

        Xarr = hyper_exponential(lambda_1 = 0.025, lambda_2 = 0.1, prob_1 = 0.35, number_of_samples = M)
        Xsrv = weibull(lambda_weibull = 2.5, k_weibull = 0.333, number_of_samples = M)
        
        A = np.cumsum(Xarr)

        C = np.zeros(M)
        C[0] = A[0] + Xsrv[0]

        for i in range(1, M):
            #A[i] = A[i-1] + Xarr[i]
            C[i] = max(A[i], C[i-1]) + Xsrv[i]
            
        # Total time
        T = C[-1] - A[0]

        # Busy time
        B = np.sum(Xsrv)

        # Utilization of a single batch
        U_k = B / T

        U_1 += U_k
        U_2 += U_k*U_k

        # Throughput of the current batch 
        X_k = M / T

        X_1 += X_k
        X_2 += X_k * X_k

        
        W_k = np.sum(C - A)

        # Average number of jobs in the system
        N_k = W_k / T
        N_1 += N_k
        N_2 += N_k * N_k

        # Response time of the current batch
        R_k = np.sum(W_k) / M
        R_1 += R_k
        R_2 += R_k * R_k

    mean_U     = U_1 / K
    mean_X     = X_1 / K
    mean_N     = N_1 / K
    mean_R     = R_1 / K

    var_U = (U_2/K) - (mean_U ** 2)
    var_X = (X_2/K) - (mean_X ** 2)
    var_N = N_2/K - mean_N ** 2
    var_R = R_2/K - mean_R ** 2


    delta_U = d_gamma * np.sqrt(var_U / K)
    Ul_U = mean_U - delta_U
    Uu_U = mean_U + delta_U
    RelErr_U = 2 * (Uu_U - Ul_U) / (Uu_U + Ul_U)


    delta_X = d_gamma * np.sqrt(var_X / K)
    Ul_X = mean_X - delta_X
    Uu_X = mean_X + delta_X
    RelErr_X = 2 * (Uu_X - Ul_X) / (Uu_X + Ul_X)

    delta_N = d_gamma * np.sqrt(var_N / K)
    Ul_N = mean_N - delta_N
    Uu_N = mean_N + delta_N
    RelErr_N = 2 * (Uu_N - Ul_N) / (Uu_N + Ul_N)

    delta_R = d_gamma * np.sqrt(var_R / K)
    Ul_R = mean_R - delta_R
    Uu_R = mean_R + delta_R
    RelErr_R = 2 * (Uu_R - Ul_R) / (Uu_R + Ul_R)

    
    if RelErr_U < maxRelErr and RelErr_X < maxRelErr and RelErr_N < maxRelErr and RelErr_R < maxRelErr:
        break
        
    K = K + DeltaK
    Krange = DeltaK

print("*"*100)
print(f"Utilization:\tmin = {Ul_U}\tmax = {Uu_U}\t relative error = {RelErr_U}")
print(f"Throughput:\tmin = {Ul_X}\tmax = {Uu_X}\t relative error = {RelErr_X}")
print(f"Average number of jobs in the system:\tmin = {Ul_N}\tmax = {Uu_N}\t relative error = {RelErr_N}")
print(f"Response time:\tmin = {Ul_R}\tmax = {Uu_R}\t relative error = {RelErr_R}")
print(f"Number of batches = {K}")

#########################################################################################################################
# SCENARIO 2

U_1 = 0
U_2 = 0

X_1 = 0
X_2 = 0

N_1 = 0
N_2 = 0

R_1 = 0
R_2 = 0

K = K0
Krange = K

while K < MaxK:
    for k in range(0, Krange):

        Xarr = erlang(lambda_erlang = 1.25, k_erlang = 8, number_of_samples = M)
        Xsrv = uniform(a = 1, b = 10, number_of_samples = M)

        A = np.cumsum(Xarr)

        C = np.zeros(M)
        C[0] = A[0] + Xsrv[0]
        for i in range(1, M):
            C[i] = max(A[i], C[i-1]) + Xsrv[i]
            
        # Total time
        T = C[-1] - A[0]

        # Busy time
        B = np.sum(Xsrv)

        # Utilization of a single batch
        U_k = B / T

        U_1 += U_k
        U_2 += U_k*U_k

        # Throughput of the current batch 
        X_k = M / T

        X_1 += X_k
        X_2 += X_k * X_k
        
        W_k = np.sum(C - A)

        # Average number of jobs in the system
        N_k = W_k / T
        N_1 += N_k
        N_2 += N_k * N_k

        # Response time of the current batch
        R_k = np.sum(W_k) / M
        R_1 += R_k
        R_2 += R_k * R_k

    mean_U     = U_1 / K
    mean_X     = X_1 / K
    mean_N     = N_1 / K
    mean_R     = R_1 / K

    var_U = (U_2/K) - (mean_U ** 2)
    var_X = (X_2/K) - (mean_X ** 2)
    var_N = N_2/K - mean_N ** 2
    var_R = R_2/K - mean_R ** 2


    delta_U = d_gamma * np.sqrt(var_U / K)
    Ul_U = mean_U - delta_U
    Uu_U = mean_U + delta_U
    RelErr_U = 2 * (Uu_U - Ul_U) / (Uu_U + Ul_U)

    delta_X = d_gamma * np.sqrt(var_X / K)
    Ul_X = mean_X - delta_X
    Uu_X = mean_X + delta_X
    RelErr_X = 2 * (Uu_X - Ul_X) / (Uu_X + Ul_X)

    delta_N = d_gamma * np.sqrt(var_N / K)
    Ul_N = mean_N - delta_N
    Uu_N = mean_N + delta_N
    RelErr_N = 2 * (Uu_N - Ul_N) / (Uu_N + Ul_N)

    delta_R = d_gamma * np.sqrt(var_R / K)
    Ul_R = mean_R - delta_R
    Uu_R = mean_R + delta_R
    RelErr_R = 2 * (Uu_R - Ul_R) / (Uu_R + Ul_R)

    
    if RelErr_U < maxRelErr and RelErr_X < maxRelErr and RelErr_N < maxRelErr and RelErr_R < maxRelErr:
        break
        
    K = K + DeltaK
    Krange = DeltaK


print("*"*100)
print(f"Utilization:\tmin = {Ul_U}\tmax = {Uu_U}\t relative error = {RelErr_U}")
print(f"Throughput:\tmin = {Ul_X}\tmax = {Uu_X}\t relative error = {RelErr_X}")
print(f"Average number of jobs in the system:\tmin = {Ul_N}\tmax = {Uu_N}\t relative error = {RelErr_N}")
print(f"Response time:\tmin = {Ul_R}\tmax = {Uu_R}\t relative error = {RelErr_R}")
print(f"Number of batches = {K}")
print("*"*100)