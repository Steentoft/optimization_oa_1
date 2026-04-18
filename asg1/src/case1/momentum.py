import numpy as np


def momentum_step(x,gradient):
    velocity = np.zeros_like(x)

    for i in range(iterations):
        velocity = beta * velocity - lr * gradient
        x[1:-1] = x[1:-1] + velocity[1:-1]
    
    return x, velocity
    
