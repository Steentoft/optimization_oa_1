import math
import numpy as np
import matplotlib.pyplot as plt

import autograd.numpy as an
from autograd import grad
import scipy


########## Task 1 ##########
x_start = np.array([0.0,0.0])
x_end = np.array ([7.0, 10.0])

# (Point, Radius, Color)
obstacles = [
    (np.array([2,4]), 1.3, 'blue'),
    (np.array([5,7]), 1.0,'orange'),
    # (np.array([3.5,5]), 2.5, 'blue'),
    # (np.array([9,2]), 0.5, 'orange'),
    # (np.array([0,3.5]), 3.0,'red'),
    # (np.array([5,3]), 3.0,'pink')
]

fig, ax = plt.subplots(figsize=(6, 6))

n_points = 50

x_init_line = np.linspace(x_start,x_end,n_points)

# Add some noise to the init line to make sure it doesn't get a stuck gradient
noise = np.random.normal(0,0.001,size=x_init_line.shape)
noise[0] = [0.0,0.0]
noise[-1] = [0.0,0.0]

x_init_line += noise


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
def penalty_1(x, obstacles):
    penalty = 0.0
    for i in range(len(obstacles)):
        dist = circular_obstacle(x, obstacles[i])
        r = obstacles[i][1]

        penalty += an.where(dist > r, 1 / (dist - r)**2, an.inf)

    return penalty

def penalty_2(x, obstacles, alpha=1):
    penalty = 0.0
    for i in range(len(obstacles)):
        penalty += an.exp(-alpha*(circular_obstacle(x, obstacles[i])**2-obstacles[i][1]**2))
    return penalty

def circular_obstacle(x, obstacle):
    return an.linalg.norm(obstacle[0] - x)

def objective_function(x, lam=1, u=10):
    # Objective Value
    objective_value = np.sum(f_L(x)+lam*f_S(x)+u*f_O(x))

    # Gradient
    gradient = gradient_f_L(x) + gradient_f_S(x) + gradient_f_O(x)

    return objective_value, gradient


epochs = 1000
best_line = x_init_line
best_objective_value, best_gradient_array = objective_function(best_line)

ax.plot(best_line[:, 0], best_line[:, 1], marker='.', label="Initial Path")

# Momentum
velocity = np.zeros_like(best_line)

def momentum_step(x,gradient,velocity,lr=0.005,beta=0.9):
    velocity = beta * velocity - lr * gradient
    x[1:-1] = x[1:-1] + velocity[1:-1]
    return x, velocity 

v_adam = np.zeros_like(best_line)
s_adam = np.zeros_like(best_line)
t = 0

def adamw(x, adam_gradient, v, s, t, lr=0.001, gamma_v=0.9, gamma_s=0.999, epsilon=1e-8, weight_decay=0.01):
    t += 1

    v = gamma_v * v - (lr * adam_gradient)
    s = gamma_s * s + (1 - gamma_s) * (adam_gradient**2)

    v_hat = v / (1 - gamma_v**t)
    s_hat = s / (1 - gamma_s**t)

    decay = lr * weight_decay * x[1:-1]
    next_x = (1.0 / (epsilon + an.sqrt(s_hat[1:-1]))) * v_hat[1:-1]

    x[1:-1] = x[1:-1] + next_x - decay

    return x, v, s, t

best_overall_path = np.copy(x_init_line)
min_objective_value = np.inf

for e in range(epochs):
    new_objective_value, new_gradient_array = objective_function(best_line,lam=10,u=0.5)    

    if new_objective_value < min_objective_value:
        min_objective_value = new_objective_value
        best_overall_path = np.copy(best_line)

    #best_line, velocity = momentum_step(best_line, new_gradient_array, velocity, lr=0.002, beta=0.6)
    best_line, v_adam, s_adam, t = adamw(best_line, new_gradient_array, v_adam, s_adam, t, lr=0.0002, gamma_v=0.9, gamma_s=0.999, weight_decay=0.13)
    

    if e % 100 == 0:
        print(f"Iteration no.: {e}")
        print(f"Objective Value: {new_objective_value}")
        ax.plot(best_line[:, 0], best_line[:, 1], marker='.', label=f"New Path no {e}")


ax.plot(best_overall_path[:, 0], best_overall_path[:, 1], marker='.', label="Best Path")

ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.legend()

plt.show()




# Adding something to penalize big spacing between points. Potentially recreating the line
# points so they are equally spread?

# Path length doesn't work, but maybe a longest distance between two points should be a objective value as well.

# For case 2, think of doing homemade optimizers.