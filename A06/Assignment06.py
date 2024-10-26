import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
import scipy.optimize as opt
import warnings


#For each scenario, using batches of M = 5000 jobs, compute the 95% confidence interval, with a 2% 
#relative error, of the following performance indices:


M = 5000
confidence_interval = 0.95
relative_error = 0.2