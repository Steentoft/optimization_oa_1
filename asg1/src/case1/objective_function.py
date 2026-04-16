import autograd.numpy as an
from autograd import grad

########## Task 2 ##########
### Path Length
def f_L(x):
    x = x.reshape((-1, 2))
    differences = x[1:] - x[:-1]
    return an.sum(differences ** 2)

def gradient_f_L(x):
    return grad(f_L)(x)

### Smoothness
def f_S(x):
    x = x.reshape((-1, 2))
    differences = x[2:] - 2 * x[1:-1] + x[:-2]
    return an.sum(differences ** 2)

def gradient_f_S(x):
    return grad(f_S)(x)

### Obstacle Avoidance
def f_O(x, obstacles):
    x = x.reshape((-1, 2))
    # For loop version, should also be vectorized.
    sum = 0.0
    for i in range(1, len(x)-1):
        sum += penalty_2(x[i], obstacles)
    return an.sum(sum)

def gradient_f_O(x, obstacles):
    return grad(f_O)(x, obstacles)

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

def objective_function(x, x_init_line, obstacles, lam=3, u=15):
    flat_x = x_init_line.copy()
    flat_x[1:-1] = x.reshape((-1, 2))

    x = flat_x.flatten()

    # Objective Value
    objective_value = an.sum(f_L(x) + lam * f_S(x) + u * f_O(x, obstacles))

    # Gradient
    gradient = gradient_f_L(x) + lam * gradient_f_S(x) + u * gradient_f_O(x, obstacles)
    gradient_full = gradient.reshape((-1, 2))
    gradient_interior = gradient_full[1:-1].flatten()

    return objective_value, gradient_interior