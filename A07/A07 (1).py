import numpy as np
from collections import deque

def ticket_price():
    ticket_index = np.random.choice([0, 1, 2], p=[0.9, 0.06, 0.04])
    ticket_prices = [2.5, 4, 6]
    return ticket_prices[ticket_index]

tMax = 1000000
totalIncome = 0

s = 1
t = 0.0

sxt = deque()
sxt.append([t, s])

tsGUI = 0
tsCash = 0
tsElectronic = 0
tsPrint = 0

RTT = deque()
curRTT = 0.0

p1HyperExp = 0.8
lambda1HyperExp = 0.4
lambda2HyperExp = 0.1

lambdaExp = 0.4

kErlang = 4
lambdaErlang = 2

p1HyperErlang = 0.95
lambda1HyperErlang = 10
lambda2HyperErlang = 0.1
k1HyperErlang = 2
k2HyperErlang = 1

while t < tMax:
    # ------------- GUI state -------------
    if s == 1:
        p1 = np.random.rand()
        if p1 < p1HyperExp:
            dt = -np.log(1 - np.random.rand()) / lambda1HyperExp
            tsGUI = tsGUI + dt
            curRTT = curRTT + dt
        else:
            dt = -np.log(1 - np.random.rand()) / lambda2HyperExp
            tsGUI = tsGUI + dt
            curRTT = curRTT + dt
            RTT.append(curRTT)
            curRTT = 0
        p = np.random.rand()
        if p < 0.2:
            ns = 1                  # to GUI state
        else:
            p = np.random.rand()
            if p < 0.35:
                ns = 2              # to cash state
            else:
                ns = 3              # to electronic state

    # ------------- Cash state -------------
    elif s == 2:
        dt = -np.log(1 - np.random.rand()) / lambdaExp
        tsCash = tsCash + dt
        curRTT = curRTT + dt
        ns = 4                      # to print state
        price = ticket_price()
        totalIncome += price

    # ------------- Electronic state -------------
    elif s == 3:
        dt = np.sum(-np.log(1 - np.random.rand(kErlang)) / lambdaErlang)
        tsElectronic = tsElectronic + dt
        curRTT = curRTT + dt
        ns = 4                      # to print state

    # ------------- Print state -------------
    elif s == 4:
        p1 = np.random.rand()
        if p1 < p1HyperErlang:
            dt = np.sum(-np.log(1 - np.random.rand(k1HyperErlang)) / lambda1HyperErlang)
            tsPrint = tsPrint + dt
            curRTT = curRTT + dt
            RTT.append(curRTT)
            curRTT = 0
            ns = 1                  # to GUI state
        else:
            dt = np.sum(-np.log(1 - np.random.rand(k2HyperErlang)) / lambda2HyperErlang)
            tsPrint = tsPrint + dt
            curRTT = curRTT + dt
            RTT.append(curRTT)
            curRTT = 0
            ns = 1                  # to GUI state
        
    
    t = t + dt
    s = ns
    sxt.append([t, s])

incomePerMinute = totalIncome / tMax
incomePerHour = incomePerMinute * 60

print("Prob. GUI: ", tsGUI / t)
print("Prob. Cash: ", tsCash / t)
print("Prob. Electronic: ", tsElectronic / t)
print("Prob. Print: ", tsPrint / t)
print("Average time between two executions of the same task: ", np.mean(list(RTT)))
print("Average income in 20 hours: {:.2f}".format(incomePerHour * 20))