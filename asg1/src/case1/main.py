from objective_function import objective_function
from gradient_descent import gradient_descent
from optimizer import CG_optimizer

import matplotlib.pyplot as plt
import numpy as np

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

iterations = 100

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

obj_fun = lambda x: objective_function(x, x_init_line, obstacles)

functions = [
    gradient_descent,
]

optimizers = [
    CG_optimizer
]

def main():

    for function in functions:

        best_line = function(x_init_line, obj_fun, iterations)

        best_line = plot_inner_flat_line(best_line)

        ax.plot(best_line[:, 0], best_line[:, 1], marker='.', label=f"Best Path")

    for optimizer in optimizers:
        new_line = x_init_line[1:-1].flatten()

        res = optimizer(new_line, obj_fun, iterations)

        rebuilt_x = plot_inner_flat_line(res.x)
        ax.plot(rebuilt_x[:, 0], rebuilt_x[:, 1], marker='.', label=f"Best Path")


    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 11)
    ax.legend()

    plt.show()

if __name__=="__main__":
    main()