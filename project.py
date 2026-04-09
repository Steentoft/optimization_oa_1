import math

import numpy as np
import matplotlib.pyplot as plt

import autograd.numpy as an
from autograd import grad


########## Task 1 ##########
x_start = np.array([0.0,0.0])
x_end = np.array ([7.0, 10.0])

# (Point, Radius, Color)
obstacles = [
    (np.array([2,4]), 1.3, 'blue'),
    (np.array([5,7]), 1.0,'orange')
]

fig, ax = plt.subplots(figsize=(6, 6))

n_points = 75

x_init_line = np.linspace(x_start,x_end,n_points)

for j in range(len(obstacles)):
    ax.add_patch(plt.Circle(obstacles[j][0],obstacles[j][1],color=obstacles[j][2]))


########## Task 2 ##########


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
        sum += penalty_2(x[i],obstacles)
    return an.sum(sum)

def gradient_f_O(x):
    return grad(f_O)(x)

def f_Ld(x):
    longest_distance = an.linalg.norm(x[1]-x[0])
    for i in range(len(x)-1):
        point_dist = an.linalg.norm(x[i+1]-x[i])
        if longest_distance < point_dist:
            longest_distance = point_dist
    return longest_distance

def gradient_f_Ld(x):
    return grad(f_Ld)(x)

def penalty_2(x, obstacles, alpha=1):
    penalty = 0.0
    for i in range(len(obstacles)):
        penalty += an.exp(-alpha*(circular_obstacle(x, obstacles[i])**2-obstacles[i][1]**2))
    return penalty

def circular_obstacle(x, obstacle):
    return abs(x-obstacle[0])

def objective_function(x, lam=1, u=1, epsilon=1):
    # Objective Value
    objective_value = np.sum(f_L(x)+lam*f_S(x)+u*f_O(x)+f_Ld(x)*epsilon)

    # Gradient
    gradient = gradient_f_L(x) + gradient_f_S(x) + gradient_f_O(x) + gradient_f_Ld(x)

    return objective_value, gradient


epochs = 20
best_line = x_init_line
best_objective_value, best_gradient_array = objective_function(best_line)

ax.plot(best_line[:, 0], best_line[:, 1], marker='.', label="Initial Path")

# NOT WORKING -> needing gradient fixes | Momentum
velocity = np.zeros_like(best_line[1:-1])

def momentum_step(x,mom_v,lr=0.5,mom_decay=0.1):
    mom_v = mom_decay * mom_v - lr * np.linalg.norm(np.gradient(x[:-1]))
    x[1:-1] = x[1:-1] - mom_v[1:-1]
    return x, mom_v 


for e in range(epochs):
    new_line = np.copy(best_line)

    # for n in range(1, len(new_line - 1)):
    #     # new_line[n] = new_line[n] + 0.01*(penalty_2(new_line[n]) * np.gradient(new_line[n]))
    #     # new_line[n] = new_line[n] + np.array([2,0.3])
    
    new_objective_value, new_gradient_array = objective_function(new_line)    


    new_line = momentum_step(new_line,new_gradient_array)[0]

    print(new_line)
    ax.plot(new_line[:, 0], new_line[:, 1], marker='.', label=f"New Path no {e}")

    if new_objective_value < best_objective_value:
        best_line = new_line



ax.plot(best_line[:, 0], best_line[:, 1], marker='.', label="Best Path")

ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)
ax.legend()

plt.show()



# Adding something to penalize big spacing between points. Potentially recreating the line
# points so they are equally spread?

# Path length doesn't work, but maybe a longest distance between two points should be a objective value as well.

# For case 2, think of doing homemade optimizers.