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


### Path Length

def f_L(x):
    # For loop version... No bueno
    # sum = 0.0
    # for i in range(len(x)-1):
    #     sum += abs(x[i+1]-x[i])**2
    # return an.sum(sum)

    differences = x[1:] - x[:-1]
    return an.sum(differences**2)


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
    for i in range(len(x)):
        sum += penalty_2(x[i],obstacles)
    return an.sum(sum)

    

def gradient_f_O(x):
    return grad(f_O)(x)



### Penalties

def penalty_2(x, obstacles, alpha=1):
    penalty = 0.0
    for i in range(len(obstacles)):
        penalty += an.exp(-alpha*(circular_obstacle(x, obstacles[i])**2-obstacles[i][1]**2))
    return penalty

def circular_obstacle(x, obstacle):
    return abs(x-obstacle[0])

def objective_function(x, lam=1, u=10):
    # Objective Value
    objective_value = np.sum(f_L(x)+lam*f_S(x)+u*f_O(x))

    # Gradient
    gradient = gradient_f_L(x) + gradient_f_S(x) + gradient_f_O(x)

    return objective_value, gradient


epochs = 20
best_line = x_init_line
best_objective_value, best_gradient_array = objective_function(best_line)

ax.plot(best_line[:, 0], best_line[:, 1], marker='.', label="Initial Path")

# NOT WORKING -> needing gradient fixes | Momentum
velocity = np.zeros_like(best_line)

def momentum_step(x,gradient,velocity,lr=0.005,beta=0.9):
    velocity = beta * velocity - lr * gradient
    x[1:-1] = x[1:-1] + velocity[1:-1]
    return x, velocity 

def gradient_descent(starting_points, learning_rate=0.005, iterations=100):
    x = starting_points
    for i in range(iterations):
        new_objective_value, new_gradient_array = objective_function(x)
        for j in range(len(x)):
            x[j] = x[j] - learning_rate * new_gradient_array[j] # update step
    return x

test = gradient_descent(x_init_line)

ax.plot(test[:, 0], test[:, 1], marker='.', label=f"Gradient Descent Path")

for e in range(epochs):
    new_line = np.copy(best_line)

    # Old without optimizers
    # for n in range(1, len(new_line - 1)):
    #     # new_line[n] = new_line[n] + 0.01*(penalty_2(new_line[n]) * np.gradient(new_line[n]))
    #     # new_line[n] = new_line[n] + np.array([2,0.3])
    
    new_objective_value, new_gradient_array = objective_function(new_line)    

    best_line, velocity = momentum_step(best_line, new_gradient_array, velocity, lr=0.02, beta=0.6)

    #ax.plot(new_line[:, 0], new_line[:, 1], marker='.', label=f"New Path no {e}")
    


    # if new_objective_value < best_objective_value:
    #     best_line = new_line



ax.plot(best_line[:, 0], best_line[:, 1], marker='.', label="Best Path")

ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)
ax.legend()

plt.show()



# Adding something to penalize big spacing between points. Potentially recreating the line
# points so they are equally spread?

# Path length doesn't work, but maybe a longest distance between two points should be a objective value as well.

# For case 2, think of doing homemade optimizers.