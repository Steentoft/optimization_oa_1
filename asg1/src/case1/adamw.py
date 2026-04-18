import numpy as np

def adamw(x, fun, args):
    v = np.zeros_like(x)
    s = np.zeros_like(x)
    t = 0
    
    iterations, lr, gamma_v, gamma_s, epsilon, weight_decay = args[0], args[1], args[2], args[3], args[4], args[5]
    
    min_adam_objective_value = np.inf
    this_x = np.copy(x)

    best_line = np.copy(x)

    convergence_points = []

    for i in range(iterations):
        adam_gradient = fun(this_x)[1]

        t += 1

        v = gamma_v * v - (lr * adam_gradient)
        s = gamma_s * s + (1 - gamma_s) * (adam_gradient**2)

        v_hat = v / (1 - gamma_v**t)
        s_hat = s / (1 - gamma_s**t)

        decay = lr * weight_decay * this_x[1:-1]
        next_x = (1.0 / (epsilon + np.sqrt(s_hat[1:-1]))) * v_hat[1:-1]

        this_x[1:-1] = this_x[1:-1] + next_x - decay

        if fun(this_x)[0] < min_adam_objective_value:
            min_adam_objective_value = fun(this_x)[0]
            best_line = this_x

        convergence_points.append((i,fun(best_line)[0]))

    return best_line, convergence_points