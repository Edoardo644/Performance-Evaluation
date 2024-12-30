import numpy as np
from scipy import linalg

lambdaInA = np.array([1.5 / 60 , 0])
lambdaInB = np.array([2.5 / 60, 0])
lambdaInC = np.array([2.0 / 60, 0])

#lamInA = lamInA / 60   # Convert in seconds
#lamInB = lamInB / 60
#lamInC = lamInC / 60

lamA = np.sum(lambdaInA)
lamB = np.sum(lambdaInB)
lamC = np.sum(lambdaInC)

SA = np.array([8, 10])
SB = np.array([3, 2])
SC = np.array([4, 7])

#SA = SA * 60           # Convert in seconds
#SB = SB * 60
#SC = SC * 60

PA = np.array([
    [0.0, 1],
    [0.10, 0.0]
])
PB = np.array([
    [0.0, 1],
    [0.08, 0.0]
])
PC = np.array([
    [0.0, 1],
    [0.12, 0.0]
])

I = np.eye(2)

vA = linalg.solve((I - PA).T, lambdaInA / lamA)
vB = linalg.solve((I - PB).T, lambdaInB / lamB)
vC = linalg.solve((I - PC).T, lambdaInC / lamC)

print("vA:                      ", vA)
print("vB:                      ", vB)
print("vC:                      ", vC)

DA = vA * SA
DB = vB * SB
DC = vC * SC

U1A = lamA * DA[0]
U1B = lamB * DB[0]
U1C = lamC * DC[0]
U2A = lamA * DA[1]
U2B = lamB * DB[1]
U2C = lamC * DC[1]

U1 = U1A + U1B + U1C
U2 = U2A + U2B + U2C

print("U1:                      ", U1)
print("U2:                      ", U2)

R1A = DA[0] / (1 - U1)
R1B = DB[0] / (1 - U1)
R1C = DC[0] / (1 - U1)
R2A = DA[1] / (1 - U2)
R2B = DB[1] / (1 - U2)
R2C = DC[1] / (1 - U2)

RA = R1A + R2A
RB = R1B + R2B
RC = R1C + R2C

N1A = lamA * R1A
N1B = lamB * R1B
N1C = lamC * R1C
N2A = lamA * R2A
N2B = lamB * R2B
N2C = lamC * R2C

NA = N1A + N2A
NB = N1B + N2B
NC = N1C + N2C

print("NA:                      ", NA)
print("NB:                      ", NB)
print("NC:                      ", NC)

print("RA:                      ", RA)
print("RB:                      ", RB)
print("RC:                      ", RC)

X = lamA + lamB + lamC

R1 = lamA / X * R1A + lamB / X * R1B + lamC / X * R1C
R2 = lamA / X * R2A + lamB / X * R2B + lamC / X * R2C
R = R1 + R2

print("System Response Timne:   ", R)

N = NA + NB + NC

print("# of Jobs in the system: ", N)