import numpy as np


def momentum(x,fun,args):
    iterations, lr, beta = args[0], args[1], args[2]

    velocity = np.zeros_like(x).flatten()
    best_line = x.copy().flatten()

    convergence_points = []

    for i in range(iterations):
        this_x = np.copy(best_line)

        gradient = fun(this_x)[1]

        velocity = beta * velocity - lr * gradient
        best_line[2:-2] = best_line[2:-2] + velocity[2:-2]

        #print(f"Momentum | Obj. Val.: {fun(this_x)[0]:.2f}")

        convergence_points.append((i, fun(best_line)[0]))


    return best_line, convergence_points
    
