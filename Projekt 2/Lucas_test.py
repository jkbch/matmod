import matplotlib as plt
import matplotlib.pyplot as py
import numpy as np
import math

mu1 = 175.5
mu2 = 162.9
x = np.linspace(-2,2,1000)
f1 = np.zeros((1,1000))
f2 = np.zeros((1,1000))
def f1(x):
     return 1 / (6.7*math.sqrt(2*math.pi))*np.exp(-1/2*1/(6.7**2)*(x-mu1)**2)
def f2(x): 
    return 1 / (6.7*math.sqrt(2*math.pi))*np.exp(-1/2*1/(6.7**2)*(x-mu2)**2)


py.plot(x,f1(x),color='green',linewidth=2)
py.plot(x,f2(x),color = 'red', linewidth = 2)
py.show()
