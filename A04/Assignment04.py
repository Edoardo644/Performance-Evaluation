import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
import scipy.optimize as opt


# Load CSV files
path_trace_1 = r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance Evaluation\A04\Trace1.csv'
path_trace_2 = r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance Evaluation\A04\Trace2.csv'

# Read  CSV files
trace1_data = pd.read_csv(path_trace_1, header=None)
trace2_data = pd.read_csv(path_trace_2, header=None)

# Convert into arrays (lists)
trace_1 = trace1_data[0].tolist()
trace_2 = trace2_data[0].tolist()

# Get the number of rows in each file
num_rows_trace1 = len(trace_1)
num_rows_trace2 = len(trace_2)

# ----------------- Trace 1 -----------------
print("-------- VALUES FOR TRACE 1 --------")

# Calculate the first three moments (mean, 2nd moment, 3rd moment) for Trace 1
EX1_trace1 = np.sum(trace_1) / num_rows_trace1
print(f"Mean value for Trace 1 = {EX1_trace1}")
EX2_trace1 = np.sum(np.square(trace_1)) / num_rows_trace1
print(f"2nd moment for Trace 1 = {EX2_trace1}")
EX3_trace1 = np.sum(np.power(trace_1, 3)) / num_rows_trace1
print(f"3rd moment for Trace 1 = {EX3_trace1}")

# Calculate variance, standard deviation, and coefficient of variation for Trace 1
variance_trace1 = np.sum(np.power(trace_1 - EX1_trace1, 2)) / num_rows_trace1
sigma_trace1 = np.sqrt(variance_trace1)
cv_trace1 = sigma_trace1 / EX1_trace1

# Fit the Uniform distribution to Trace 1 
a_trace1 = EX1_trace1 - 0.5 * np.sqrt(12 * (EX2_trace1 - np.power(EX1_trace1, 2)))
b_trace1 = EX1_trace1 + 0.5 * np.sqrt(12 * (EX2_trace1 - np.power(EX1_trace1, 2)))
print(f"Uniform distribution for Trace 1: a = {a_trace1}, b = {b_trace1}")

# Fit the Exponential distribution to Trace 1 
lambda_exp_trace1 = 1 / EX1_trace1
print(f"Exponential distribution for Trace 1: lambda = {lambda_exp_trace1}")

# Fit the Erlang distribution to Trace 1 using method of moments
k_erlang_trace1 = np.round(np.power(EX1_trace1, 2) / (EX2_trace1 - np.power(EX1_trace1, 2)))
lambda_erlang_trace1 = k_erlang_trace1 / EX1_trace1
print(f"Erlang distribution for Trace 1: k = {k_erlang_trace1}, lambda = {lambda_erlang_trace1}")

# Fit the Weibull distribution using the method of moments for Trace 1
def fit_weibull(M1, M2):
    def fun(x):
        l1 = x[0]
        k = x[1]

        M1d = gamma(1 + 1 / k) * l1
        M2d = gamma(1 + 2 / k) * np.power(l1, 2)

        return (M1d / M1 - 1) ** 2 + (M2d / M2 - 1) ** 2

    initial_guess = np.array([0.001, 0.001])

    bounds = ((0.001, 100.0), (0.1, 100.0))
    sx = opt.minimize(fun, initial_guess, bounds=bounds)

    l1d = sx.x[0]
    k = sx.x[1]

    return l1d, k

lambda_weibull_trace1, k_weibull_trace1 = fit_weibull(EX1_trace1, EX2_trace1)
print(f"Weibull distribution for Trace 1: lambda = {lambda_weibull_trace1}, k = {k_weibull_trace1}")
moment1_weibull_trace1 = gamma(1 + 1 / k_weibull_trace1) * lambda_weibull_trace1
moment2_weibull_trace1 = gamma(1 + 2 / k_weibull_trace1) * np.power(lambda_weibull_trace1, 2)
print(f"1st moment for Weibull distribution = {moment1_weibull_trace1}")
print(f"2nd moment for Weibull distribution = {moment2_weibull_trace1}")

# Fit the Pareto distribution using the method of moments for Trace 1
def fit_pareto(M1, M2):
    def fun(x):
        m = x[0]
        alpha = x[1]

        if alpha <= 2:
            return 1e10

        M1d = m * alpha / (alpha - 1)
        M2d = m ** 2 * alpha / (alpha - 2)
        
        return (M1d / M1 - 1) ** 2 + (M2d / M2 - 1) ** 2
    
    initial_guess = np.array([1.0, 2.0])

    bounds = ((1.001, 100.0), (2.001, 100.0))
    sx = opt.minimize(fun, initial_guess, bounds=bounds)
    
    m = sx.x[0]
    alpha = sx.x[1]

    return m, alpha

m_pareto_trace1, alpha_pareto_trace1 = fit_pareto(EX1_trace1, EX2_trace1)
print(f"Pareto distribution for Trace 1: m = {m_pareto_trace1}, alpha = {alpha_pareto_trace1}")
moment1_pareto_trace1 = m_pareto_trace1 * alpha_pareto_trace1 / (alpha_pareto_trace1 - 1)
moment2_pareto_trace1 = m_pareto_trace1 ** 2 * alpha_pareto_trace1 / (alpha_pareto_trace1 - 2)
print(f"1st moment for Pareto distribution = {moment1_pareto_trace1}")
print(f"2nd moment for Pareto distribution = {moment2_pareto_trace1}")

# Fit the 2 stage Hyperexponential distribution using maximum likelihood estimation for Trace 1
srt_trace1 = np.sort(trace_1)
def fit_hyperexponential(M1, M2, srt_trace):
    def fun(x):
        l1 = x[0]
        l2 = x[1]
        p1 = x[2]
        p2 = 1 - p1

        return -np.sum(np.log(p1 * l1 * np.exp(-l1 * srt_trace) + p2 * l2 * np.exp(-l2 * srt_trace)))
    
    sx = opt.minimize(fun, np.array([0.8/M1,1.2/M1,0.4]), bounds=((0.001, 100.0), (0.001, 100.0), (0.001, 0.999)), constraints=[{'type': 'ineq', 'fun': lambda x:  x[1] - x[0] - 0.001}])

    l1d = sx.x[0]
    l2d = sx.x[1]
    p1d = sx.x[2]
    p2d = 1 - p1d

    return l1d, l2d, p1d, p2d

l1_hyper_trace1, l2_hyper_trace1, p1_hyper_trace1, p2_hyper_trace1 = fit_hyperexponential(EX1_trace1, EX2_trace1, srt_trace1)
print(f"Hyperexponential distribution for Trace 1: l1 = {l1_hyper_trace1}, l2 = {l2_hyper_trace1}, p1 = {p1_hyper_trace1}, p2 = {p2_hyper_trace1}")
moment1_hyper_trace1 = p1_hyper_trace1 / l1_hyper_trace1 + p2_hyper_trace1 / l2_hyper_trace1
moment2_hyper_trace1 = 2 * (p1_hyper_trace1 / l1_hyper_trace1 ** 2 + p2_hyper_trace1 / l2_hyper_trace1 ** 2)
moment3_hyper_trace1 = 6 * (p1_hyper_trace1 / l1_hyper_trace1 ** 3 + p2_hyper_trace1 / l2_hyper_trace1 ** 3)
print(f"1st moment for Hyperexponential distribution = {moment1_hyper_trace1}")
print(f"2nd moment for Hyperexponential distribution = {moment2_hyper_trace1}")
print(f"3rd moment for Hyperexponential distribution = {moment3_hyper_trace1}")

# Fit the 2 stage Hypoexponential distribution using maximum likelihood estimation for Trace 1
def fit_hypoexponential(M1, M2, srt_trace):
    def fun(x):
        l1 = x[0]
        l2 = x[1]

        if l1 == l2:
            return -1000000-l1-l2
        else:
            return -np.sum(np.log(l1 * l2 / (l1 - l2) * (np.exp(-l2 * srt_trace) - np.exp(-l1 * srt_trace))))
        
    sx = opt.minimize(fun, np.array([1/(0.7*M1),1/(0.3*M1)]), bounds=((0.001, 100.0), (0.001, 100.0)), constraints=[{'type': 'ineq', 'fun': lambda x:  x[1] - x[0] - 0.001}])

    l1d = sx.x[0]
    l2d = sx.x[1]

    return l1d, l2d

l1_hypo_trace1, l2_hypo_trace1 = fit_hypoexponential(EX1_trace1, EX2_trace1, srt_trace1)
print(f"Hypoexponential distribution for Trace 1: l1 = {l1_hypo_trace1}, l2 = {l2_hypo_trace1}")
moment1_hypo_trace1 = 1 / l1_hypo_trace1 + 1 / l2_hypo_trace1
moment2_hypo_trace1 = 2 * (1 / l1_hypo_trace1 ** 2 + 1 / (l1_hypo_trace1 * l2_hypo_trace1) + 1 / l2_hypo_trace1 ** 2)
print(f"1st moment for Hypoexponential distribution = {moment1_hypo_trace1}")
print(f"2nd moment for Hypoexponential distribution = {moment2_hypo_trace1}")


# Plotting the distributions
t = np.linspace(0, 30, 1000)  # Adjust t to range from 0 to 30
probV = np.r_[1.:num_rows_trace1 + 1] / num_rows_trace1

FUniform_trace1 = np.where(t < a_trace1, 0, np.where(t > b_trace1, 1, (t - a_trace1) / (b_trace1 - a_trace1)))
FExponential_trace1 = 1 - np.exp(-lambda_exp_trace1 * t)
FErlang_trace1 = 1 - np.array([np.sum([np.power(lambda_erlang_trace1 * t_val, i) * np.exp(-lambda_erlang_trace1 * t_val) / math.factorial(i) for i in range(int(k_erlang_trace1))]) for t_val in t])
FWeibull_trace1 = 1 - np.exp(-np.power(t / lambda_weibull_trace1, k_weibull_trace1))
FPareto_trace1 = np.where(t >= m_pareto_trace1, 1 - np.power(m_pareto_trace1 / t, alpha_pareto_trace1), 0)
FHyper_trace1 = 1 - p1_hyper_trace1 * np.exp(-t * l1_hyper_trace1) - p2_hyper_trace1 * np.exp(-t * l2_hyper_trace1)
FHypo_trace1 = 1 - 1 / (l2_hypo_trace1 - l1_hypo_trace1) * (l2_hypo_trace1 * np.exp(-l1_hypo_trace1 * t) - l1_hypo_trace1 * np.exp(-l2_hypo_trace1 * t))

'''
plt.plot(srt_trace1, probV, ".")
plt.plot(t, FUniform_trace1_trace1)
plt.plot(t, FExponential_trace1)
plt.plot(t, FErlang_trace1)
plt.plot(t, FWeibull_trace1)
plt.plot(t, FPareto_trace1)
plt.plot(t, FHyper_trace1)
plt.plot(t, FHypo_trace1)
plt.xlabel('t')
plt.ylabel('CDF')
plt.legend(["Empirical", "Uniform", "Exponential", "Erlang", "Weibull", "Pareto", "Hyperexponential", "Hypoexponential"])
plt.grid(True)
plt.show()
'''

# Plot empirical CDF and fitted CDFs for Trace 1

# Create subplots
fig, axs = plt.subplots(2, 4, figsize=(15, 10))

# Plot each distribution along with the empirical data in a separate subplot
axs[0, 0].plot(srt_trace1, probV, ".", label="Empirical Data")
axs[0, 0].plot(t, FUniform_trace1, label="Uniform")
axs[0, 0].grid(True)
axs[0, 0].set_title("Uniform Distribution")
axs[0, 0].legend()

axs[0, 1].plot(srt_trace1, probV, ".", label="Empirical Data")
axs[0, 1].plot(t, FExponential_trace1, label="Exponential")
axs[0, 1].grid(True)
axs[0, 1].set_title("Exponential Distribution")
axs[0, 1].legend()

axs[0, 2].plot(srt_trace1, probV, ".", label="Empirical Data")
axs[0, 2].plot(t, FErlang_trace1, label="Erlang")
axs[0, 2].grid(True)
axs[0, 2].set_title("Erlang Distribution")
axs[0, 2].legend()

axs[0, 3].plot(srt_trace1, probV, ".", label="Empirical Data")
axs[0, 3].plot(t, FWeibull_trace1, label="Weibull")
axs[0, 3].grid(True)
axs[0, 3].set_title("Weibull Distribution")
axs[0, 3].legend()

axs[1, 0].plot(srt_trace1, probV, ".", label="Empirical Data")
axs[1, 0].plot(t, FPareto_trace1, label="Pareto")
axs[1, 0].grid(True)
axs[1, 0].set_title("Pareto Distribution")
axs[1, 0].legend()

axs[1, 1].plot(srt_trace1, probV, ".", label="Empirical Data")
axs[1, 1].plot(t, FHyper_trace1, label="Hyper-exponential")
axs[1, 1].grid(True)
axs[1, 1].set_title("Hyper-exponential Distribution")
axs[1, 1].legend()

axs[1, 2].plot(srt_trace1, probV, ".", label="Empirical Data")
axs[1, 2].plot(t, FHypo_trace1, label="Hypo-exponential")
axs[1, 2].grid(True)
axs[1, 2].set_title("Hypo-exponential Distribution")
axs[1, 2].legend()

# Hide the last subplot (bottom-right) if not used
axs[1, 3].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()


#Comments are repeated for trace 2, no need to actually rewrite them

# ----------------- Trace 2 -----------------
print("-------- VALUES FOR TRACE 2 --------")

EX1_trace2 = np.sum(trace_2) / num_rows_trace2
print(f"Mean value for Trace 2 = {EX1_trace2}")
EX2_trace2 = np.sum(np.square(trace_2)) / num_rows_trace2
print(f"2nd moment for Trace 2 = {EX2_trace2}")
EX3_trace2 = np.sum(np.power(trace_2, 3)) / num_rows_trace2
print(f"3rd moment for Trace 2 = {EX3_trace2}")

CX1_trace2 = np.sum(np.power(trace_2 - EX1_trace2, 1)) / num_rows_trace2
sigma_trace2 = np.sqrt(CX1_trace2)

# Uniform distribution
a_trace2 = EX1_trace2 - 0.5 * np.sqrt(12 * (EX2_trace2 - np.power(EX1_trace2, 2)))
b_trace2 = EX1_trace2 + 0.5 * np.sqrt(12 * (EX2_trace2 - np.power(EX1_trace2, 2)))
print(f"Uniform distribution for Trace 2: a = {a_trace2}, b = {b_trace2}")

# Exponential distribution
lambda_exp_trace2 = 1 / EX1_trace2
print(f"Exponential distribution for Trace 2: lambda = {lambda_exp_trace2}")

# Erlang distribution
k_erlang_trace2 = np.round(np.power(EX1_trace2, 2) / (EX2_trace2 - np.power(EX1_trace2, 2)))
lambda_erlang_trace2 = k_erlang_trace2 / EX1_trace2
print(f"Erlang distribution for Trace 2: k = {k_erlang_trace2}, lambda = {lambda_erlang_trace2}")

# Weibull distribution method of moments
lambda_weibull_trace2, k_weibull_trace2 = fit_weibull(EX1_trace2, EX2_trace2)
print(f"Weibull distribution for Trace 2: lambda = {lambda_weibull_trace2}, k = {k_weibull_trace2}")
moment1_weibull_trace2 = gamma(1 + 1 / k_weibull_trace2) * lambda_weibull_trace2
moment2_weibull_trace2 = gamma(1 + 2 / k_weibull_trace2) * np.power(lambda_weibull_trace2, 2)
print(f"1st moment for Weibull distribution = {moment1_weibull_trace2}")
print(f"2nd moment for Weibull distribution = {moment2_weibull_trace2}")

# Pareto distribution method of moments
m_pareto_trace2, alpha_pareto_trace2 = fit_pareto(EX1_trace2, EX2_trace2)
print(f"Pareto distribution for Trace 2: m = {m_pareto_trace2}, alpha = {alpha_pareto_trace2}")
moment1_pareto_trace2 = m_pareto_trace2 * alpha_pareto_trace2 / (alpha_pareto_trace2 - 1)
moment2_pareto_trace2 = m_pareto_trace2 ** 2 * alpha_pareto_trace2 / (alpha_pareto_trace2 - 2)
print(f"1st moment for Pareto distribution = {moment1_pareto_trace2}")
print(f"2nd moment for Pareto distribution = {moment2_pareto_trace2}")

# Two stage hyperexponential distribution maximum likelihood method
srt_trace2 = np.sort(trace_2)
l1_hyper_trace2, l2_hyper_trace2, p1_hyper_trace2, p2_hyper_trace2 = fit_hyperexponential(EX1_trace2, EX2_trace2, srt_trace2)
print(f"Hyperexponential distribution for Trace 2: l1 = {l1_hyper_trace2}, l2 = {l2_hyper_trace2}, p1 = {p1_hyper_trace2}, p2 = {p2_hyper_trace2}")
moment1_hyper_trace2 = p1_hyper_trace2 / l1_hyper_trace2 + p2_hyper_trace2 / l2_hyper_trace2
moment2_hyper_trace2 = 2 * (p1_hyper_trace2 / l1_hyper_trace2 ** 2 + p2_hyper_trace2 / l2_hyper_trace2 ** 2)
moment3_hyper_trace2 = 6 * (p1_hyper_trace2 / l1_hyper_trace2 ** 3 + p2_hyper_trace2 / l2_hyper_trace2 ** 3)
print(f"1st moment for Hyperexponential distribution = {moment1_hyper_trace2}")
print(f"2nd moment for Hyperexponential distribution = {moment2_hyper_trace2}")
print(f"3rd moment for Hyperexponential distribution = {moment3_hyper_trace2}")

# Two stage hypoexponential distribution maximum likelihood method
l1_hypo_trace2, l2_hypo_trace2 = fit_hypoexponential(EX1_trace2, EX2_trace2, srt_trace2)
print(f"Hypoexponential distribution for Trace 2: l1 = {l1_hypo_trace2}, l2 = {l2_hypo_trace2}")
moment1_hypo_trace2 = 1 / l1_hypo_trace2 + 1 / l2_hypo_trace2
moment2_hypo_trace2 = 2 * (1 / l1_hypo_trace2 ** 2 + 1 / (l1_hypo_trace2 * l2_hypo_trace2) + 1 / l2_hypo_trace2 ** 2)
print(f"1st moment for Hypoexponential distribution = {moment1_hypo_trace2}")
print(f"2nd moment for Hypoexponential distribution = {moment2_hypo_trace2}")

# Plotting the distributions
t = np.linspace(0, 80, 1000)  # Adjust t to range from 0 to 80
probV = np.r_[1.:num_rows_trace2 + 1] / num_rows_trace2

FUniform_trace2 = np.where(t < a_trace2, 0, np.where(t > b_trace2, 1, (t - a_trace2) / (b_trace2 - a_trace2)))
FExponential_trace2 = 1 - np.exp(-lambda_exp_trace2 * t)
FErlang_trace2 = 1 - np.array([np.sum([np.power(lambda_erlang_trace2 * t_val, i) * np.exp(-lambda_erlang_trace2 * t_val) / math.factorial(i) for i in range(int(k_erlang_trace2))]) for t_val in t])
FWeibull_trace2 = 1 - np.exp(-np.power(t / lambda_weibull_trace2, k_weibull_trace2))
FPareto_trace2 = np.where(t >= m_pareto_trace2, 1 - np.power(m_pareto_trace2 / t, alpha_pareto_trace2), 0)
FHyper_trace2 = 1 - p1_hyper_trace2 * np.exp(-t * l1_hyper_trace2) - p2_hyper_trace2 * np.exp(-t * l2_hyper_trace2)
FHypo_trace2 = 1 - 1 / (l2_hypo_trace2 - l1_hypo_trace2) * (l2_hypo_trace2 * np.exp(-l1_hypo_trace2 * t) - l1_hypo_trace2 * np.exp(-l2_hypo_trace2 * t))

#IT GIVES SOME UNDEFINED VARIABLES FOR SOME REASON
#Save all calculated values for Trace 1 and Trace 2 into a text file to make them easier to read
"""def save_trace_values_to_file():
   
    with open("trace_values.txt", "w") as file:
        # Writing values for Trace 1
        file.write("-------- VALUES FOR TRACE 1 --------\n")
        file.write(f"Mean value for Trace 1 = {EX1_trace1}\n")
        file.write(f"2nd moment for Trace 1 = {EX2_trace1}\n")
        file.write(f"3rd moment for Trace 1 = {EX3_trace1}\n")
        file.write(f"Variance for Trace 1 = {variance_trace1}\n")
        file.write(f"Standard deviation for Trace 1 = {sigma_trace1}\n")
        file.write(f"Coefficient of variation for Trace 1 = {cv_trace1}\n")
        file.write(f"Uniform distribution (a, b) = ({a_trace1}, {b_trace1})\n")
        file.write(f"Exponential distribution lambda = {lambda_exp_trace1}\n")
        file.write(f"Erlang distribution (k, lambda) = ({k_erlang_trace1}, {lambda_erlang_trace1})\n")
        file.write(f"Weibull distribution (lambda, k) = ({lambda_weibull_trace1}, {k_weibull_trace1})\n")
        file.write(f"Pareto distribution (m, alpha) = ({m_pareto_trace1}, {alpha_pareto_trace1})\n")
        file.write(f"Hyperexponential distribution (l1, l2, p1, p2) = ({l1_hyper_trace1}, {l2_hyper_trace1}, {p1_hyper_trace1}, {1 - p1_hyper_trace1})\n")
        file.write(f"Hypoexponential distribution (l1, l2) = ({l1_hypo_trace1}, {l2_hypo_trace1})\n")
        
        # Add a separator for clarity between Trace 1 and Trace 2
        file.write("\n-------- VALUES FOR TRACE 2 --------\n")
        
        # Repeat the same for Trace 2, assuming you've calculated the values
        file.write(f"Mean value for Trace 2 = {EX1_trace2}\n")
        file.write(f"2nd moment for Trace 2 = {EX2_trace2}\n")
        file.write(f"3rd moment for Trace 2 = {EX3_trace2}\n")
        file.write(f"Variance for Trace 2 = {variance_trace2}\n")
        file.write(f"Standard deviation for Trace 2 = {sigma_trace2}\n")
        file.write(f"Coefficient of variation for Trace 2 = {cv_trace2}\n")
        file.write(f"Uniform distribution (a, b) = ({a_trace2}, {b_trace2})\n")
        file.write(f"Exponential distribution lambda = {lambda_exp_trace2}\n")
        file.write(f"Erlang distribution (k, lambda) = ({k_erlang_trace2}, {lambda_erlang_trace2})\n")
        file.write(f"Weibull distribution (lambda, k) = ({lambda_weibull_trace2}, {k_weibull_trace2})\n")
        file.write(f"Pareto distribution (m, alpha) = ({m_pareto_trace2}, {alpha_pareto_trace2})\n")
        file.write(f"Hyperexponential distribution (l1, l2, p1, p2) = ({l1_hyper_trace2}, {l2_hyper_trace2}, {p1_hyper_trace2}, {1 - p1_hyper_trace2})\n")
        file.write(f"Hypoexponential distribution (l1, l2) = ({l1_hypo_trace2}, {l2_hypo_trace2})\n")
    
    print("Values for Trace 1 and Trace 2 have been saved to 'trace_values.txt'.")

# Call the function at the end of the script
save_trace_values_to_file()"""



# Filter empirical data to only include points where x <= 80
filtered_indices = srt_trace2 <= 80
srt_trace2_filtered = srt_trace2[filtered_indices]
probV_filtered = probV[filtered_indices]

# Create subplots
fig, axs = plt.subplots(2, 4, figsize=(15, 10))

# Plot each distribution along with the empirical data in a separate subplot
axs[0, 0].plot(srt_trace2_filtered, probV_filtered, ".", label="Empirical Data")
axs[0, 0].plot(t, FUniform_trace2, label="Uniform")
axs[0, 0].grid(True)
axs[0, 0].set_title("Uniform Distribution")
axs[0, 0].legend()

axs[0, 1].plot(srt_trace2_filtered, probV_filtered, ".", label="Empirical Data")
axs[0, 1].plot(t, FExponential_trace2, label="Exponential")
axs[0, 1].grid(True)
axs[0, 1].set_title("Exponential Distribution")
axs[0, 1].legend()

axs[0, 2].plot(srt_trace2_filtered, probV_filtered, ".", label="Empirical Data")
axs[0, 2].plot(t, FErlang_trace2, label="Erlang")
axs[0, 2].grid(True)
axs[0, 2].set_title("Erlang Distribution")
axs[0, 2].legend()

axs[0, 3].plot(srt_trace2_filtered, probV_filtered, ".", label="Empirical Data")
axs[0, 3].plot(t, FWeibull_trace2, label="Weibull")
axs[0, 3].grid(True)
axs[0, 3].set_title("Weibull Distribution")
axs[0, 3].legend()

axs[1, 0].plot(srt_trace2_filtered, probV_filtered, ".", label="Empirical Data")
axs[1, 0].plot(t, FPareto_trace2, label="Pareto")
axs[1, 0].grid(True)
axs[1, 0].set_title("Pareto Distribution")
axs[1, 0].legend()

axs[1, 1].plot(srt_trace2_filtered, probV_filtered, ".", label="Empirical Data")
axs[1, 1].plot(t, FHyper_trace2, label="Hyper-exponential")
axs[1, 1].grid(True)
axs[1, 1].set_title("Hyper-exponential Distribution")
axs[1, 1].legend()

axs[1, 2].plot(srt_trace2_filtered, probV_filtered, ".", label="Empirical Data")
axs[1, 2].plot(t, FHypo_trace2, label="Hypo-exponential")
axs[1, 2].grid(True)
axs[1, 2].set_title("Hypo-exponential Distribution")
axs[1, 2].legend()

# Hide the last subplot (bottom-right) if not used
axs[1, 3].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()