import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from objective_function import *

########## Task 1 ##########
x_start = np.array([0.0, 0.0])
x_end = np.array([7.0, 10.0])

# (Point, Radius, Color)
obstacles = [
    (np.array([2, 4]), 1.3, 'blue'),
    (np.array([5, 7]), 1.0, 'orange')
]
n_points = 20

x_init_line = np.linspace(x_start, x_end, n_points)

### Plotting
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x_init_line[:, 0], x_init_line[:, 1], marker='.', label="Initial Path")

for j in range(len(obstacles)):
    ax.add_patch(plt.Circle(obstacles[j][0], obstacles[j][1], color=obstacles[j][2]))

def plot_inner_flat_line(x):
    new_line = x_init_line.copy()
    x = x.reshape((-1, 2))
    new_line[1:-1] = x
    return new_line

########## Task 4 ##########
fun = lambda x: objective_function(x, x_init_line, obstacles)

### Optimizer ###
inner_x_init = x_init_line[1:-1].flatten()
res = minimize(fun, inner_x_init, method='CG', tol=0.001, jac=True, options={'maxiter': 100})

rebuilt_x = plot_inner_flat_line(res.x)
ax.plot(rebuilt_x[:, 0], rebuilt_x[:, 1], marker='.', label=f"Optimizer {res.fun}]")

best_line = x_init_line.copy()

### Gradient Descent ###
best_inner_line = x_init_line.copy()[1:-1].flatten()
inner_new_line = np.copy(best_inner_line)

epochs = 100
learning_rate = 0.001

for e in range(epochs):
    inner_new_line = best_inner_line

    new_objective_value, new_gradient_array = fun(inner_new_line)

    for j in range(len(inner_new_line)):
        inner_new_line[j] = (inner_new_line[j] - (learning_rate * new_gradient_array[j])) #Update step

    if fun(inner_new_line)[0] < fun(best_inner_line)[0]:
        best_inner_line = inner_new_line

best_inner_line = plot_inner_flat_line(best_inner_line)
ax.plot(best_inner_line[:, 0], best_inner_line[:, 1], marker='.', label=f"Best Path {fun(best_inner_line[1:-1].flatten())[0]}")

ax.set_xlim(-0.5, 11)
ax.set_ylim(-0.5, 11)
ax.legend()

plt.show()