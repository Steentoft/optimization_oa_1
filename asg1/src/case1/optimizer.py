import math

import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import reshape
from scipy.optimize import minimize

import autograd.numpy as an
from autograd import grad

########## Task 1 ##########
x_start = np.array([0.0, 0.0])
x_end = np.array([7.0, 10.0])

# (Point, Radius, Color)
obstacles = [
    (np.array([2, 4]), 1.3, 'blue'),
    (np.array([5, 7]), 1.0, 'orange')
]

fig, ax = plt.subplots(figsize=(6, 6))

n_points = 20

x_init_line = np.linspace(x_start, x_end, n_points)

for j in range(len(obstacles)):
    ax.add_patch(plt.Circle(obstacles[j][0], obstacles[j][1], color=obstacles[j][2]))


########## Task 2 ##########


### Path Length

def f_L(x):
    # For loop version... No bueno
     #sum = 0.0
     #for i in range(1, len(x)-2):
     #    sum += abs(x[i + 1] - x[i]) ** 2
         #sum += abs(x[i]-x[i+2]+x[i+1]-x[i+3])**2
     #return an.sum(sum)

    pts = x.reshape((-1, 2))
    differences = pts[1:] - pts[:-1]
    return an.sum(differences ** 2)


def gradient_f_L(x):
    return grad(f_L)(x)


### Smoothness
def f_S(x):
    # For loop version
    #sum = 0.0
    #for i in range(1,len(x)-1):
    #    sum += abs(x[i+1]-2*x[i]+x[i-1])**2
    #return an.sum(sum)

    x = x.reshape((-1, 2))
    differences = x[2:] - 2 * x[1:-1] + x[:-2]
    return an.sum(differences ** 2)


def gradient_f_S(x):
    return grad(f_S)(x)


### Obstacle Avoidance
def f_O(x):
    x = x.reshape((-1, 2))
    # For loop version, should also be vectorized.
    sum = 0.0
    for i in range(1, len(x)-1):
        sum += penalty_2(x[i], obstacles)
    return an.sum(sum)


def gradient_f_O(x):
    return grad(f_O)(x)

### Penalties

def penalty_1(x, obstacles):
    penalty = 0.0
    for i in range(len(obstacles)):
        if an.linalg.norm(circular_obstacle(x, [obstacles[i]]) > obstacles[i][1]):
            penalty += 1/(circular_obstacle(x,[obstacles[i]])-obstacles[i][1])**2
        else:
            penalty += 999999
    return penalty

def penalty_2(x, obstacles, alpha=1):
    penalty = 0.0
    for i in range(len(obstacles)):
        penalty += an.exp(-alpha * (circular_obstacle(x, obstacles[i]) ** 2 - obstacles[i][1] ** 2))
    return penalty


def circular_obstacle(x, obstacle):
    return an.linalg.norm(obstacle[0] - x)


def objective_function(x, lam=3, u=15):
    flat_x = x_init_line.copy()
    flat_x[1:-1] = x.reshape((-1, 2))

    x = flat_x.flatten()

    # Objective Value
    objective_value = an.sum(f_L(x) + lam * f_S(x) + u * f_O(x))

    # Gradient
    gradient = gradient_f_L(x) + lam * gradient_f_S(x) + u * gradient_f_O(x)

    gradient_full = gradient.reshape((-1, 2))

    gradient_interior = gradient_full[1:-1].flatten()

    return objective_value, gradient_interior

inner_x_init = x_init_line[1:-1].flatten()

res = minimize(objective_function, inner_x_init, method='CG', tol=0.001, jac=True, options={'maxiter': 300})

minimize_line = np.reshape(res.x, (-1, 2))

rebuilt_x = x_init_line.copy()
print(rebuilt_x[1:-1].shape)
print(minimize_line.shape)

rebuilt_x[1:-1] = minimize_line.reshape((-1, 2))

ax.plot(rebuilt_x[:, 0], rebuilt_x[:, 1], marker='.', label="Optimizer")

ax.plot(x_init_line[:, 0], x_init_line[:, 1], marker='.', label="Initial Path")


ax.set_xlim(-0.5, 11)
ax.set_ylim(-0.5, 11)
ax.legend()

plt.show()

