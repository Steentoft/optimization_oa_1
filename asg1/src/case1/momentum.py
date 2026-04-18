import numpy as np


def momentum(x,fun,args):
    this_x = np.copy(x)

    iterations, lr, beta = args[0], args[1], args[2]

    velocity = np.zeros_like(x).flatten()
    best_line = x.copy().flatten()
    min_mom_objective_value = np.inf

    convergence_points = []

    for i in range(iterations):
        this_x = np.copy(best_line)

        objective_value, gradient = fun(this_x)
        velocity = beta * velocity - lr * gradient
        this_x[2:-2] = this_x[2:-2] + velocity[2:-2]
    
        if fun(this_x)[0] < min_mom_objective_value:
            min_mom_objective_value = fun(this_x)[0]
            best_line = this_x

        #print(f"Momentum | Obj. Val.: {fun(this_x)[0]:.2f}")

        convergence_points.append(fun(best_line)[0])


    return best_line, convergence_points
    
