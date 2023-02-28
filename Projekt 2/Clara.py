import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits import mplot3d

mu1 = 175.5
mu2 = 162.9
x = np.linspace(130,200,1000)
f1 = np.zeros((1,1000))
f2 = np.zeros((1,1000))
def f1(x):
     return 1 / (6.7*math.sqrt(2*math.pi))*np.exp(-1/2*1/(6.7**2)*(x-mu1)**2)
def f2(x): 
    return 1 / (6.7*math.sqrt(2*math.pi))*np.exp(-1/2*1/(6.7**2)*(x-mu2)**2)


#py.plot(x,f1(x),color='green',linewidth=2)
#py.plot(x,f2(x),color = 'red', linewidth = 2)
#py.show()

# Solution til exercise 1.3
i=0
while(f1(x[i])<f2(x[i]) ):
    i=i+1


print(x[i])

#Exercise 1.5-1.7

def f3(x1,x2):
   return 1/(2*math.pi)*1/(2*3)*np.exp(-1/2*(1/4*x1**2+1/9*(x2-1)**2))

#mu is a vector containing the average of x1 and x2. sd is a 2x2 matrix whith sd1^2, p*sd1*sd2;p*sd1*sd2 ,sd2^2. 
def binorm(x1,x2,mu,sd):          #Bivariat normalfordeling
    p=sd[1]/(math.sqrt(sd[0]*sd[3]))
    return 1/(2*math.pi)*1/(sd[0]*sd[3])*1/(math.sqrt(1-p**2))*np.exp(-1/2*1/(1-p**2)*(1/sd[0]*(x1-mu[0])**2+1/sd[3]*(x2-mu[1])**2-2*p**2*1/sd[1]*(x1-mu[0])*(x2-mu[1])))

x1=np.linspace(-5,7,100)
x2=np.linspace(-5,7,100)
p=2/3

X, Y = np.meshgrid(x1, x2)

Z1=f3(X, Y)
#Z2=binorm(X,Y,p)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z1, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

plt.show()
