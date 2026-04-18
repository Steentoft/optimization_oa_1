import numpy as np
velocity = np.zeros_like(x_init_line)

def momentum_step(x,gradient,velocity,lr=0.005,beta=0.9):
    
    velocity = beta * velocity - lr * gradient
    x[1:-1] = x[1:-1] + velocity[1:-1]
    return x, velocity 