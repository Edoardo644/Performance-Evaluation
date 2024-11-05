import numpy as np

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

M = 5000 # number of jobs
maxRelErr = 0.02 # maximum relative error for stopping criterion

deltaK = 10 # increment for K in each iteration
K0 = 50 # initial value of K
K = K0
Krange = K
maxK = 20000 # maximum allowed K value before stopping

gamma_dict = {0.99: 2.576, 0.98: 2.326, 0.95: 1.96, 0.9: 1.645}

required_confidence_interval = 0.95
d_gamma = gamma_dict[required_confidence_interval]



# Parameters for hyper-exponential distribution
p1Hyper = 0.35
lambda1Hyper = 0.025
lambda2Hyper = 0.1

# Parameters for Weibull distribution
kWeibull = 0.333
lambdaWeibull = 2.5

#Parameters for Erlang distribution
kErlang = 8
lambdaErlang = 1.25

#Parameters for uniform distribution
a = 1
b = 10



                        ############# SCENARIO 1 ################  
#Utilization
U_1 = 0
U_2 = 0

#Throughput
X_1 = 0
X_2 = 0

#Avg number of jobs
N_1 = 0
N_2 = 0

#Avg response time
R_1 = 0
R_2 = 0



while K < maxK:
    
    # Reset accumulators at the start of each K iteration
    #U = Uv = X = Xv = L = Lv = W = Wv = 0
    
    for k in range(0, Krange):
        
        # Generate random variables for hyper-exponential and Weibull distributions
        XHyper = hyper_exponential(lambda1Hyper, lambda2Hyper, p1Hyper, M)
        XWeibull = weibull(lambdaWeibull, kWeibull,M)
        
        A = np.cumsum(XHyper)

        C = np.zeros(M)
        C[0] = A[0] + XWeibull[0]

        for i in range(1, M):
            C[i] = max(A[i], C[i-1]) + XWeibull[i]
        
        
        #Calculate system metrics 
        T = C[-1] - A[0] #total simulation time
        
        B = np.sum(XWeibull) #total busy time
        
        Uk = B / T #utilization
        U_1 += Uk
        U_2 += Uk*Uk
        
        Xk = M / T #throughput of current batch
        X_1 += Xk
        X_2 += Xk * Xk
        
        Wk = np.sum(C - A)

        # Average number of jobs in the system
        Nk = Wk / T
        N_1 += Nk
        N_2 += Nk * Nk
        
        # Response time of the current batch
        Rk = np.sum(Wk) / M
        R_1 += Rk
        R_2 += Rk * Rk
    
    #Calculate averages, variances, and confidence intervals for system metrics
    EU = U_1 / K #mean value
    EUv = U_2 / K #mean square value
    VarU = (EUv) - (EU ** 2) #variance
    SigmaU = np.sqrt(VarU) #standard deviation
    deltaU = d_gamma * np.sqrt(VarU / K)
    #bring values into the interval
    Ul_U = EU - deltaU
    Uu_U = EU + deltaU
    #real error
    RelErrU = 2 * (Uu_U - Ul_U) / (Uu_U + Ul_U)

    #The comments can be repeated fo throughput, Jobs, Repsponse time
    EX = X_1 / K
    EXv = X_2 / K
    VarX = (EXv) - (EX ** 2)  # clamp to zero if small negative due to precision
    SigmaX = np.sqrt(VarX)
    deltaX = d_gamma * np.sqrt(VarX / K)
    Ul_X = EX - deltaX
    Uu_X = EX + deltaX
    RelErrX = 2 * (Uu_X - Ul_X) / (Uu_X + Ul_X)

    EL = N_1 / K
    ELv = N_2 / K
    VarL = (ELv) - (EL ** 2) # clamp to zero if small negative due to precision
    SigmaL = np.sqrt(VarL)
    deltaL = d_gamma * np.sqrt(VarL / K)
    Ul_L = EL - deltaL
    Uu_L = EL + deltaL
    RelErrL = 2 * (Uu_L - Ul_L) / (Uu_L + Ul_L)

    ER = R_1 / K
    ERv = R_2 / K
    VarR = (ERv) - (ER ** 2)  # clamp to zero if small negative due to precision
    SigmaR = np.sqrt(VarR)
    deltaR = d_gamma * np.sqrt(VarR / K)
    Ul_R = ER - deltaR
    Uu_R = ER + deltaR
    RelErrW = 2 * (Uu_R - Ul_R) / (Uu_R + Ul_R)



    # Check if all relative errors are within the maximum allowed threshold
    if (RelErrU < maxRelErr) and (RelErrL < maxRelErr) and (RelErrW < maxRelErr) and (RelErrX < maxRelErr):
        break

    # increment K and repeat the process 
    K = K + deltaK
    Krange = deltaK
    
    

print("#" * 50)
print("Scenario 1")
print("#" * 50)   
 
print("Solution obtained in K = ", K, " iterations")
print("Utilizatiomn = [", Ul_U, ", ", Uu_U, "], Relative error: ", RelErrU)
print("Throughput = [", Ul_X, ", ", Uu_X, "], Relative error: ", RelErrX)
print("Average #jobs = [", Ul_L, ", ", Uu_L, "], Relative error: ", RelErrL)
print("Average Resp. time = [", Ul_R, ", ", Uu_R, "], Relative error: ", RelErrW)


print("#" * 50)
print("Scenario 2")
print("#" * 50)


#################### Scenario 2 ####################

#Utilization
U_1 = 0
U_2 = 0

#Throughput
X_1 = 0
X_2 = 0

#Avg number of jobs
N_1 = 0
N_2 = 0

#Avg response time
R_1 = 0
R_2 = 0

K = K0
Krange = K


while K < maxK:
    
    # Reset accumulators at the start of each K iteration
    #U = Uv = X = Xv = L = Lv = W = Wv = 0
    
    for k in range(0, Krange):
        
        # Generate random variables for hyper-exponential and Weibull distributions
        XErlang = erlang(lambdaErlang, kErlang, M)
        XUniform = uniform(a, b,M)
        
        A = np.cumsum(XErlang)

        C = np.zeros(M)
        C[0] = A[0] + XUniform[0]

        for i in range(1, M):
            C[i] = max(A[i], C[i-1]) + XUniform[i]
        
        
        #Calculate system metrics 
        T = C[-1] - A[0] #total simulation time
        
        B = np.sum(XUniform) #total busy time
        
        Uk = B / T #utilization
        U_1 += Uk
        U_2 += Uk*Uk
        
        Xk = M / T #throughput of current batch
        X_1 += Xk
        X_2 += Xk * Xk
        
        Wk = np.sum(C - A)

        # Average number of jobs in the system
        Nk = Wk / T
        N_1 += Nk
        N_2 += Nk * Nk
        
        # Response time of the current batch
        Rk = np.sum(Wk) / M
        R_1 += Rk
        R_2 += Rk * Rk
    
    #Calculate averages, variances, and confidence intervals for system metrics
    EU = U_1 / K #mean value
    EUv = U_2 / K #mean square value
    VarU = (EUv) - (EU ** 2) #variance
    SigmaU = np.sqrt(VarU) #standard deviation
    deltaU = d_gamma * np.sqrt(VarU / K)
    #bring values into the interval
    Ul_U = EU - deltaU
    Uu_U = EU + deltaU
    #real error
    RelErrU = 2 * (Uu_U - Ul_U) / (Uu_U + Ul_U)

    #The comments can be repeated fo throughput, Jobs, Repsponse time
    EX = X_1 / K
    EXv = X_2 / K
    VarX = (EXv) - (EX ** 2)  
    SigmaX = np.sqrt(VarX)
    deltaX = d_gamma * np.sqrt(VarX / K)
    Ul_X = EX - deltaX
    Uu_X = EX + deltaX
    RelErrX = 2 * (Uu_X - Ul_X) / (Uu_X + Ul_X)

    EL = N_1 / K
    ELv = N_2 / K
    VarL = (ELv) - (EL ** 2) 
    SigmaL = np.sqrt(VarL)
    deltaL = d_gamma * np.sqrt(VarL / K)
    Ul_L = EL - deltaL
    Uu_L = EL + deltaL
    RelErrL = 2 * (Uu_L - Ul_L) / (Uu_L + Ul_L)

    ER = R_1 / K
    ERv = R_2 / K
    VarR = (ERv) - (ER ** 2)  
    SigmaR = np.sqrt(VarR)
    deltaR = d_gamma * np.sqrt(VarR / K)
    Ul_R = ER - deltaR
    Uu_R = ER + deltaR
    RelErrW = 2 * (Uu_R - Ul_R) / (Uu_R + Ul_R)


    # Check if all relative errors are within the maximum allowed threshold
    if (RelErrU < maxRelErr) and (RelErrL < maxRelErr) and (RelErrW < maxRelErr) and (RelErrX < maxRelErr):
        break

    # increment K and repeat the process 
    K = K + deltaK
    Krange = deltaK



print("#" * 50)
print("Scenario 2")
print("#" * 50)
print("Solution obtained in K = ", K, " iterations")
print("Utilizatiomn = [", Ul_U, ", ", Uu_U, "], Relative error: ", RelErrU)
print("Throughput = [", Ul_X, ", ", Uu_X, "], Relative error: ", RelErrX)
print("Average #jobs = [", Ul_L, ", ", Uu_L, "], Relative error: ", RelErrL)
print("Average Resp. time = [", Ul_R, ", ", Uu_R, "], Relative error: ", RelErrW)