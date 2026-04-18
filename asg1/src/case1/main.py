from objective_function import objective_function, objective_function_op
from gradient_descent import gradient_descent
from momentum import momentum
from adamw import adamw
from newtonmethod import newtonsmethod
from optimizer import CG_optimizer

import matplotlib.pyplot as plt
import numpy as np
import time

########## Task 1 ##########
x_start = np.array([0.0, 0.0])
x_end = np.array([7.0, 10.0])

# (Point, Radius, Color)
obstacles = [
    (np.array([2, 4]), 1.3, 'blue'),
    (np.array([4, 8]), 1.0, 'orange')
]

n_points = 50
lam = 1
u = 1

x_init_line = np.linspace(x_start, x_end, n_points)

iterations = 120

obj_fun = lambda x: objective_function(x, obstacles, lam, u)
obj_fun_op = lambda x: objective_function_op(x, x_init_line, obstacles, lam, u)

### Plotting
fig, ax = plt.subplots(1,2, figsize=(12, 6))
ax[0].plot(x_init_line[:, 0], x_init_line[:, 1], 'blue', marker='.', label=f"Initial Path | Obj. Val.: {obj_fun(x_init_line)[0]:.2f}")

for j in range(len(obstacles)):
    ax[0].add_patch(plt.Circle(obstacles[j][0], obstacles[j][1], color=obstacles[j][2]))

def plot_inner_flat_line(x):
    new_line = x_init_line.copy()
    x = x.reshape((-1, 2))
    new_line[1:-1] = x
    return new_line


functions = [
    { "func" : gradient_descent, "name" : "Gradient Descent", "col" : 'red', "args" : [iterations, 0.01]},
    { "func" : momentum, "name" : "Momentum", "col" : 'green', "args" : [iterations, 0.005, 0.9]},
    { "func" : adamw, "name" : "AdamW", "col" : 'orange', "args" : [iterations, 0.001, 0.9, 0.999, 1e-8, 0.01]},
    { "func" : newtonsmethod, "name" : "Newton's Method", "col" : 'pink', "args" : [iterations, 1e-8, 0.5]}
]

optimizers = [
    CG_optimizer
]

def main():

    for function in functions:
        print("Running function: ", function["name"])
        start_time = time.time()
        best_line, convergence_points = function["func"](x_init_line, obj_fun, function["args"])

        best_line = best_line.reshape((-1, 2))

        conv_steps, objective_val_point = zip(*convergence_points)
        ax[0].plot(best_line[:, 0], best_line[:, 1], marker='.', color=function["col"], label=f"{function["name"]} | Obj. Val.: {objective_val_point[-1]:.2f} | Runtime: {time.time()-start_time:.2f} secs")
        ax[1].plot(conv_steps,objective_val_point, marker='.', color=function["col"], label=f"{function['name']} | Init. Obj. Val.: {objective_val_point[0]:.2f}")


    for optimizer in optimizers:
        new_line = x_init_line[1:-1].flatten()

        start_time = time.time()

        convergence_points = []
        for iteration in range(iterations):
            res = optimizer(new_line, obj_fun_op, 1)
            new_line = res.x
            convergence_points.append((iteration,res.fun))

        rebuilt_x = plot_inner_flat_line(res.x)
        conv_steps, objective_val_point = zip(*convergence_points)
        ax[0].plot(rebuilt_x[:, 0], rebuilt_x[:, 1], marker='.', label=f"Optimizer Path | Final Obj. Val.: {objective_val_point[-1]:.2f} | Runtime: {time.time()-start_time:.2f} secs")
        ax[1].plot(conv_steps,objective_val_point, marker='.', label=f"Optimizer | Init. Obj. Val.: {objective_val_point[0]:.2f}")


    ax[0].set_xlim(-0.5, 11)
    ax[0].set_ylim(-0.5, 11)
    ax[0].legend(fontsize=7)

    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Objective Value")
    ax[1].set_ylim(-0.5, 50)
    ax[1].legend(fontsize=7)
    ax[1].grid()

    plt.suptitle(f"N_points: {n_points} | λ: {lam} | μ: {u} |")
    plt.savefig(f"asg1/src/case1/plots/Npoint{n_points}Lam{lam}Mu{u}Obs-2hit1")
    plt.show()

if __name__=="__main__":
    main()