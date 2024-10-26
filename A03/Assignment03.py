import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
path_trace_1 = r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance Evaluation\A03\Trace1.csv'
path_trace_2 = r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance Evaluation\A03\Trace2.csv'
path_trace_3 = r'C:\Users\User\Desktop\OneDrive - Universita degli Studi Roma Tre\Desktop\Programming\Performance Evaluation\A03\Trace3.csv'

# Read  CSV files
trace1_data = pd.read_csv(path_trace_1, header=None)
trace2_data = pd.read_csv(path_trace_2, header=None)
trace3_data = pd.read_csv(path_trace_3, header=None)

# Convert into arrays (lists)
trace_1 = trace1_data[0].tolist()
trace_2 = trace2_data[0].tolist()
trace_3 = trace3_data[0].tolist()

# Get the number of rows in each file
num_rows_trace1 = len(trace_1)
num_rows_trace2 = len(trace_2)
num_rows_trace3 = len(trace_3)

# ----------------- Trace 1 -----------------
print("-------- VALUES FOR TRACE 1 --------")
EX1_trace1 = np.sum(trace_1) / num_rows_trace1
print(f"Mean value for Trace 1 = {EX1_trace1}")
EX2_trace1 = np.sum(np.square(trace_1)) / num_rows_trace1
print(f"2nd moment for Trace 1 = {EX2_trace1}")
EX3_trace1 = np.sum(np.power(trace_1, 3)) / num_rows_trace1
print(f"3rd moment for Trace 1 = {EX3_trace1}")
EX4_trace1 = np.sum(np.power(trace_1, 4)) / num_rows_trace1
print(f"4th moment for Trace 1 = {EX4_trace1}")

CX2_trace1 = np.sum(np.power(trace_1 - EX1_trace1, 2)) / num_rows_trace1
print(f"Variance for Trace 1 = {CX2_trace1}")
sigma = np.sqrt(CX2_trace1)
CX3_trace1 = np.sum(np.power(trace_1 - EX1_trace1, 3)) / num_rows_trace1
print(f"3rd central moment for Trace 1 = {CX3_trace1}")
CX4_trace1 = np.sum(np.power(trace_1 - EX1_trace1, 4)) / num_rows_trace1
print(f"4th central moment for Trace 1 = {CX4_trace1}")

SX3_trace1 = np.sum(np.power((trace_1 - EX1_trace1) / sigma, 3)) / num_rows_trace1
print(f"Skewness for Trace 1 = {SX3_trace1}")
SX4_trace1 = np.sum(np.power((trace_1 - EX1_trace1) / sigma, 4)) / num_rows_trace1
print(f"4th standardized moment for Trace 1 = {SX4_trace1}")

print(f"Standard deviation for Trace 1 = {sigma}")
cv_trace1 = sigma / EX1_trace1
print(f"Coefficent of variation for Trace 1 = {cv_trace1}")
Excess_kurtosis_trace1 = SX4_trace1 - 3
print(f"Excess kurtosis for Trace 1 = {Excess_kurtosis_trace1}")

sorted_trace_1 = np.sort(trace_1)

i50p_trace1 = (num_rows_trace1 - 1) * 50 / 100
i50pint_trace1 = np.floor(i50p_trace1)
p50_trace1 = sorted_trace_1[int(i50pint_trace1)] + (i50p_trace1 - i50pint_trace1) * (sorted_trace_1[int(i50pint_trace1) + 1] - sorted_trace_1[int(i50pint_trace1)])
print(f"Median for Trace 1 = {p50_trace1}")
i25p_trace1 = (num_rows_trace1 - 1) * 25 / 100
i25pint_trace1 = np.floor(i25p_trace1)
p25_trace1 = sorted_trace_1[int(i25pint_trace1)] + (i25p_trace1 - i25pint_trace1) * (sorted_trace_1[int(i25pint_trace1) + 1] - sorted_trace_1[int(i25pint_trace1)])
print(f"1st quartile for Trace 1 = {p25_trace1}")
i75p_trace1 = (num_rows_trace1 - 1) * 75 / 100
i75pint_trace1 = np.floor(i75p_trace1)
p75_trace1 = sorted_trace_1[int(i75pint_trace1)] + (i75p_trace1 - i75pint_trace1) * (sorted_trace_1[int(i75pint_trace1) + 1] - sorted_trace_1[int(i75pint_trace1)])
print(f"3rd quartile for Trace 1 = {p75_trace1}")
i05p_trace1 = (num_rows_trace1 - 1) * 5 / 100
i05pint_trace1 = np.floor(i05p_trace1)
p05_trace1 = sorted_trace_1[int(i05pint_trace1)] + (i05p_trace1 - i05pint_trace1) * (sorted_trace_1[int(i05pint_trace1) + 1] - sorted_trace_1[int(i05pint_trace1)])
print(f"5th percentile for Trace 1 = {p05_trace1}")
i90p_trace1 = (num_rows_trace1 - 1) * 90 / 100
i90pint_trace1 = np.floor(i90p_trace1)
p90_trace1 = sorted_trace_1[int(i90pint_trace1)] + (i90p_trace1 - i90pint_trace1) * (sorted_trace_1[int(i90pint_trace1) + 1] - sorted_trace_1[int(i90pint_trace1)])
print(f"95th percentile for Trace 1 = {p90_trace1}")

# ----------------- Trace 2 -----------------
print("-------- VALUES FOR TRACE 2 --------")
EX1_trace2 = np.sum(trace_2) / num_rows_trace2
print(f"Mean value for Trace 2 = {EX1_trace2}")
EX2_trace2 = np.sum(np.square(trace_2)) / num_rows_trace2
print(f"2nd moment for Trace 2 = {EX2_trace2}")
EX3_trace2 = np.sum(np.power(trace_2, 3)) / num_rows_trace2
print(f"3rd moment for Trace 2 = {EX3_trace2}")
EX4_trace2 = np.sum(np.power(trace_2, 4)) / num_rows_trace2
print(f"4th moment for Trace 2 = {EX4_trace2}")

CX2_trace2 = np.sum(np.power(trace_2 - EX1_trace2, 2)) / num_rows_trace2
print(f"Variance for Trace 2 = {CX2_trace2}")
sigma = np.sqrt(CX2_trace2)
CX3_trace2 = np.sum(np.power(trace_2 - EX1_trace2, 3)) / num_rows_trace2
print(f"3rd central moment for Trace 2 = {CX3_trace2}")
CX4_trace2 = np.sum(np.power(trace_2 - EX1_trace2, 4)) / num_rows_trace2
print(f"4th central moment for Trace 2 = {CX4_trace2}")

SX3_trace2 = np.sum(np.power((trace_2 - EX1_trace2) / sigma, 3)) / num_rows_trace2
print(f"Skewness for Trace 2 = {SX3_trace2}")
SX4_trace2 = np.sum(np.power((trace_2 - EX1_trace2) / sigma, 4)) / num_rows_trace2
print(f"4th standardized moment for Trace 2 = {SX4_trace2}")

print(f"Standard deviation for Trace 2 = {sigma}")
cv_trace2 = sigma / EX1_trace2
print(f"Coefficent of variation for Trace 2 = {cv_trace2}")
Excess_kurtosis_trace2 = SX4_trace2 - 3
print(f"Excess kurtosis for Trace 2 = {Excess_kurtosis_trace2}")

sorted_trace_2 = np.sort(trace_2)

i50p_trace2 = (num_rows_trace2 - 1) * 50 / 100
i50pint_trace2 = np.floor(i50p_trace2)
p50_trace2 = sorted_trace_2[int(i50pint_trace2)] + (i50p_trace2 - i50pint_trace2) * (sorted_trace_2[int(i50pint_trace2) + 1] - sorted_trace_2[int(i50pint_trace2)])
print(f"Median for Trace 2 = {p50_trace2}")
i25p_trace2 = (num_rows_trace2 - 1) * 25 / 100
i25pint_trace2 = np.floor(i25p_trace2)
p25_trace2 = sorted_trace_2[int(i25pint_trace2)] + (i25p_trace2 - i25pint_trace2) * (sorted_trace_2[int(i25pint_trace2) + 1] - sorted_trace_2[int(i25pint_trace2)])
print(f"1st quartile for Trace 2 = {p25_trace2}")
i75p_trace2 = (num_rows_trace2 - 1) * 75 / 100
i75pint_trace2 = np.floor(i75p_trace2)
p75_trace2 = sorted_trace_2[int(i75pint_trace2)] + (i75p_trace2 - i75pint_trace2) * (sorted_trace_2[int(i75pint_trace2) + 1] - sorted_trace_2[int(i75pint_trace2)])
print(f"3rd quartile for Trace 2 = {p75_trace2}")
i05p_trace2 = (num_rows_trace2 - 1) * 5 / 100
i05pint_trace2 = np.floor(i05p_trace2)
p05_trace2 = sorted_trace_2[int(i05pint_trace2)] + (i05p_trace2 - i05pint_trace2) * (sorted_trace_2[int(i05pint_trace2) + 1] - sorted_trace_2[int(i05pint_trace2)])
print(f"5th percentile for Trace 2 = {p05_trace2}")
i90p_trace2 = (num_rows_trace2 - 1) * 90 / 100
i90pint_trace2 = np.floor(i90p_trace2)
p90_trace2 = sorted_trace_2[int(i90pint_trace2)] + (i90p_trace2 - i90pint_trace2) * (sorted_trace_2[int(i90pint_trace2) + 1] - sorted_trace_2[int(i90pint_trace2)])
print(f"95th percentile for Trace 2 = {p90_trace2}")

# ----------------- Trace 3 -----------------
print("-------- VALUES FOR TRACE 3 --------")
EX1_trace3 = np.sum(trace_3) / num_rows_trace3
print(f"Mean value for Trace 3 = {EX1_trace3}")
EX2_trace3 = np.sum(np.square(trace_3)) / num_rows_trace3
print(f"2nd moment for Trace 3 = {EX2_trace3}")
EX3_trace3 = np.sum(np.power(trace_3, 3)) / num_rows_trace3
print(f"3rd moment for Trace 3 = {EX3_trace3}")
EX4_trace3 = np.sum(np.power(trace_3, 4)) / num_rows_trace3
print(f"4th moment for Trace 3 = {EX4_trace3}")

CX2_trace3 = np.sum(np.power(trace_3 - EX1_trace3, 2)) / num_rows_trace3
print(f"Variance for Trace 3 = {CX2_trace3}")
sigma = np.sqrt(CX2_trace3)
CX3_trace3 = np.sum(np.power(trace_3 - EX1_trace3, 3)) / num_rows_trace3
print(f"3rd central moment for Trace 3 = {CX3_trace3}")
CX4_trace3 = np.sum(np.power(trace_3 - EX1_trace3, 4)) / num_rows_trace3
print(f"4th central moment for Trace 3 = {CX4_trace3}")

SX3_trace3 = np.sum(np.power((trace_3 - EX1_trace3) / sigma, 3)) / num_rows_trace3
print(f"Skewness for Trace 3 = {SX3_trace3}")
SX4_trace3 = np.sum(np.power((trace_3 - EX1_trace3) / sigma, 4)) / num_rows_trace3
print(f"4th standardized moment for Trace 3 = {SX4_trace3}")

print(f"Standard deviation for Trace 3 = {sigma}")
cv_trace3 = sigma / EX1_trace3
print(f"Coefficent of variation for Trace 3 = {cv_trace3}")
Excess_kurtosis_trace3 = SX4_trace3 - 3
print(f"Excess kurtosis for Trace 3 = {Excess_kurtosis_trace3}")

sorted_trace_3 = np.sort(trace_3)

i50p_trace3 = (num_rows_trace3 - 1) * 50 / 100
i50pint_trace3 = np.floor(i50p_trace3)
p50_trace3 = sorted_trace_3[int(i50pint_trace3)] + (i50p_trace3 - i50pint_trace3) * (sorted_trace_3[int(i50pint_trace3) + 1] - sorted_trace_3[int(i50pint_trace3)])
print(f"Median for Trace 3 = {p50_trace3}")
i25p_trace3 = (num_rows_trace3 - 1) * 25 / 100
i25pint_trace3 = np.floor(i25p_trace3)
p25_trace3 = sorted_trace_3[int(i25pint_trace3)] + (i25p_trace3 - i25pint_trace3) * (sorted_trace_3[int(i25pint_trace3) + 1] - sorted_trace_3[int(i25pint_trace3)])
print(f"1st quartile for Trace 3 = {p25_trace3}")
i75p_trace3 = (num_rows_trace3 - 1) * 75 / 100
i75pint_trace3 = np.floor(i75p_trace3)
p75_trace3 = sorted_trace_3[int(i75pint_trace3)] + (i75p_trace3 - i75pint_trace3) * (sorted_trace_3[int(i75pint_trace3) + 1] - sorted_trace_3[int(i75pint_trace3)])
print(f"3rd quartile for Trace 3 = {p75_trace3}")
i05p_trace3 = (num_rows_trace3 - 1) * 5 / 100
i05pint_trace3 = np.floor(i05p_trace3)
p05_trace3 = sorted_trace_3[int(i05pint_trace3)] + (i05p_trace3 - i05pint_trace3) * (sorted_trace_3[int(i05pint_trace3) + 1] - sorted_trace_3[int(i05pint_trace3)])
print(f"5th percentile for Trace 3 = {p05_trace3}")
i90p_trace3 = (num_rows_trace3 - 1) * 90 / 100
i90pint_trace3 = np.floor(i90p_trace3)
p90_trace3 = sorted_trace_3[int(i90pint_trace3)] + (i90p_trace3 - i90pint_trace3) * (sorted_trace_3[int(i90pint_trace3) + 1] - sorted_trace_3[int(i90pint_trace3)])
print(f"95th percentile for Trace 3 = {p90_trace3}")


# Draw figure with the Pearson Correlation Coefficient for lags m=1 to m=100

def pearson(trace, range):
    mean = np.sum(trace) / len(trace)
    variance = 1 / len(trace) * np.sum((trace - mean)**2)

    return np.sum((trace[range:] - mean) * (trace[:-range] - mean)) / (len(trace)-range) / variance

range = np.arange(1, 101, 1)
pearson_trace1 = [pearson(trace_1, i) for i in range]
pearson_trace2 = [pearson(trace_2, i) for i in range]
pearson_trace3 = [pearson(trace_3, i) for i in range]

plt.plot(range, pearson_trace1)
plt.xlabel('Lag')
plt.ylabel('Pearson Correlation Coefficient')
plt.grid(True)
plt.show()

plt.plot(range, pearson_trace2)
plt.xlabel('Lag')
plt.ylabel('Pearson Correlation Coefficient')
plt.grid(True)
plt.show()

plt.plot(range, pearson_trace3)
plt.xlabel('Lag')
plt.ylabel('Pearson Correlation Coefficient')
plt.grid(True)
plt.show()


# Draw the approximated CDF of the corresponding distribution

yp1 = np.r_[1:num_rows_trace1+1] / num_rows_trace1
yp2 = np.r_[1:num_rows_trace2+1] / num_rows_trace2
yp3 = np.r_[1:num_rows_trace3+1] / num_rows_trace3

plt.plot(sorted_trace_1, yp1)
plt.xlabel('Values')
plt.ylabel('CDF')
plt.title('CDF of Trace 1')
plt.grid(True)
plt.show()

plt.plot(sorted_trace_2, yp2)
plt.xlim([0, 30])
plt.xlabel('Values')
plt.ylabel('CDF')
plt.title('CDF of Trace 2')
plt.grid(True)
plt.show()

plt.plot(sorted_trace_3, yp3)
plt.xlabel('Values')
plt.ylabel('CDF')
plt.title('CDF of Trace 3')
plt.grid(True)
plt.show()