import math

import numpy as np
import matplotlib.pyplot as plt

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

n_points = 50

x_init_line = np.linspace(x_start, x_end, n_points)

for j in range(len(obstacles)):
    ax.add_patch(plt.Circle(obstacles[j][0], obstacles[j][1], color=obstacles[j][2]))


########## Task 2 ##########


### Path Length

def f_L(x):
    # For loop version... No bueno
     sum = 0.0
     for i in range(1, len(x)-2):
         sum += abs(x[i+1]-x[i])**2
     return an.sum(sum)

    #differences = x[1:] - x[:-1]
    #return an.sum(differences)**2


def gradient_f_L(x):
    return grad(f_L)(x)


### Smoothness
def f_S(x):
    # For loop version
    # sum = 0.0
    # for i in range(1,len(x)-1):
    #     sum += abs(x[i+1]-2*x[i]+x[i-1])**2
    # return an.sum(sum)

    differences = x[2:] - 2 * x[1:-1] + x[:-2]
    return an.sum(differences**2)


def gradient_f_S(x):
    return grad(f_S)(x)


### Obstacle Avoidance
def f_O(x):
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
    return an.linalg.norm(an.sqrt((obstacle[0][0]-x[0])**2+(obstacle[0][1]-x[1])**2))


def objective_function(x, lam=1, u=10):
    # Objective Value
    objective_value = np.sum(f_L(x) + lam * f_S(x) + u * f_O(x))

    # Gradient
    gradient = gradient_f_L(x) + gradient_f_S(x) + gradient_f_O(x)

    return objective_value, gradient


best_line = x_init_line

ax.plot(best_line[:, 0], best_line[:, 1], marker='.', label="Initial Path")

epochs = 100
learning_rate = 0.02

### Gradient Descent ###
for e in range(epochs):
    new_line = np.copy(best_line)

    new_objective_value, new_gradient_array = objective_function(new_line)

    for j in range(1, len(new_line)-1):
        new_line[j] = (new_line[j] - (learning_rate * new_gradient_array[j])) #Update step


    #if objective_function(new_line)[0] < objective_function(best_line)[0]:
    best_line = new_line

    #ax.plot(new_line[:, 0], new_line[:, 1], marker='.', label=f"{e}")

ax.plot(best_line[:, 0], best_line[:, 1], marker='.', label="Best Path")

ax.set_xlim(-0.5, 11)
ax.set_ylim(-0.5, 11)
ax.legend()

plt.show()

# Adding something to penalize big spacing between points. Potentially recreating the line
# points so they are equally spread?

# Path length doesn't work, but maybe a longest distance between two points should be a objective value as well.

# For case 2, think of doing homemade optimizers.