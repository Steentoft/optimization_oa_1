import numpy as np
import autograd.numpy as an
from autograd import grad

x_start = np.array([0.0,0.0])
x_end = np.array ([7.0, 10.0])

n_points = 75

x_init_line = np.linspace(x_start,x_end,n_points)


def f_L(x):
    sum = 0.0
    for i in range(len(x)-1):
        sum += abs(x[i+1]-x[i])**2
    return an.sum(sum)


def gradient_f_L(x):
    return grad(f_L)(x)


def f_S(x):
    sum = 0.0
    for i in range(len(x)-1):
        sum += abs(x[i+1]-2*x[i]+x[i-1])**2
    return an.sum(sum)

def gradient_f_S(x):
    return grad(f_S)(x)

    
def f_O(x):
    sum = 0.0
    for i in range(len(x)):
        sum += penalty_2(x[i])
    return an.sum(sum)

def gradient_f_O(x):
    return grad(f_O)(x)

print(gradient_f_O(x_init_line))

def penalty_2(x, obstacles, alpha=1):
    penalty = 0.0
    for i in range(len(obstacles)):
        penalty += an.exp(-alpha*(circular_obstacle(x, obstacles[i])**2-obstacles[i][1]**2))
    return penalty

def circular_obstacle(x, obstacle):
    return abs(x-obstacle[0])

