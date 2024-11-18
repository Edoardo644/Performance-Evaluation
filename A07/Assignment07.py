import numpy as np
from collections import deque

def ticket_price():
    ticket_index = np.random.choice([0, 1, 2], p=[0.9, 0.06, 0.04])
    ticket_prices = [2.5, 4, 6]
    return ticket_prices[ticket_index]

# Initialize simulation parameters
tMax = 1000000
totalIncome = 0
t = 0.0
s = 1
cycles = 0

# Data structures to store results
sxt = deque([[t, s]])
RTT = deque()
curRTT = 0.0

# Time tracking for each state
tsGUI = 0
tsCash = 0
tsCard = 0
tsPrint = 0

# Parameters for distributions
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

#Payment with cash
def handle_cash_state():
    global tsCash, curRTT, t, s, totalIncome
    dt = -np.log(1 - np.random.rand()) / lambdaExp
    tsCash += dt
    curRTT += dt
    totalIncome += ticket_price()
    s = 4 # go to print state
    t += dt
    
#Electronic payment with card
def handle_card_state():
    global tsCard, curRTT, t, s
    dt = np.sum(-np.log(1 - np.random.rand(kErlang)) / lambdaErlang)
    tsCard += dt
    curRTT += dt
    s = 4 # go to print state
    t += dt
    

# GUI starting state
def handle_gui_state():
    global tsGUI, curRTT, t, s
    p1 = np.random.rand()
    if p1 < p1HyperExp:
        dt = -np.log(1 - np.random.rand()) / lambda1HyperExp
        tsGUI += dt
        curRTT += dt
    else:
        dt = -np.log(1 - np.random.rand()) / lambda2HyperExp
        tsGUI += dt
        curRTT += dt
        RTT.append(curRTT)
        curRTT = 0
    p2 = np.random.rand()
    if p2 < 0.2:
        s = 1 #return to GUI state
    else:
        p3 = np.random.rand()
        if p3 < 0.65:
            s = 3 # 
        else:
            s = 2
    t += dt
"""
def handle_payment_state():
    global s, t
    p = np.random.rand()
    if p < 0.35:  # Cash payment
        t += use_cash()
    else:  # Card payment
        t += use_card()
    s = 3  # Transition to print state
"""

#Print final state
def handle_print_state():
    global tsPrint, curRTT, t, s, cycles
    p1 = np.random.rand()
    if p1 < p1HyperErlang:
        dt = np.sum(-np.log(1 - np.random.rand(k1HyperErlang)) / lambda1HyperErlang)
    else:
        dt = np.sum(-np.log(1 - np.random.rand(k2HyperErlang)) / lambda2HyperErlang)
    tsPrint += dt
    curRTT += dt
    RTT.append(curRTT)
    curRTT = 0
    s = 1  # Transition back to GUI state
    cycles += 1
    t += dt
    
# Dictionary mapping state to right function
state_handlers = {
    1: handle_gui_state,
    2: handle_cash_state,
    3: handle_card_state,
    4: handle_print_state
}

# Main simulation loop
while t < tMax:
    state_handlers[s]()
    sxt.append([t, s])

incomePerMinute = totalIncome / tMax
incomePerHour = incomePerMinute * 60

print("Prob. GUI: ", tsGUI / t)
print("Prob. Cash: ", tsCash / t)
print("Prob. Card: ", tsCard / t)
print("Prob. Print: ", tsPrint / t)
print("Average transaction duration: ", np.mean(list(RTT)))
print("Average income in 20 hours: {:.2f}".format(incomePerHour * 20))
print("Number of cycles:", cycles)