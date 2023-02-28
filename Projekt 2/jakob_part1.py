# %%
# Exercise 1

import matplotlib.pyplot as plt
import numpy as np


def f1(x, mu):
    return 1 / (6.7 * np.sqrt(2 * np.pi)) * np.exp(-1/2 * 1/6.7**2 * (x - mu)**2)

# %%
# Exercise 2


mu = 175.5
x = np.linspace(-10, 10, 100) + mu

plt.plot(x, f1(x, mu))
plt.show()

# %%
# Exercise 3

mu1 = 175.5
mu2 = 162.9

x = np.linspace(-20, 20, 100) + (mu1 + mu2) / 2

plt.plot(x, f(x, mu1))
plt.plot(x, f(x, mu2))
plt.show()

# %%
# Exercise 5


def f2(x1, x2):
    return 1 / (2 * np.pi) * 1 / (2 * 3) * np.exp(- 1/2 * (1/4 * x1**2 + 1/9 * (x2 - 1)**2))


x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)

X, Y = np.meshgrid(x1, x2)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f2(X, Y))


# %%
# Exercise 6
def g(x1, x2, p):
    return 1 / (2 * np.pi) * 1 / 6 * 1 / (np.sqrt(1 - p**2)) * np.exp(-1/2 * 1/(1 - p**2) * (1/4 * x1**2 - 2 * p * 1/6 * x1 * (x2 - 1) + 1/9 * (x2 - 1)**2))


x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)

X, Y = np.meshgrid(x1, x2)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, g(X, Y, 2/3))
