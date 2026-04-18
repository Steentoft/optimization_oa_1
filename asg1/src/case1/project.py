import math
import numpy as np
import matplotlib.pyplot as plt
import time

import autograd.numpy as an
from autograd import grad, hessian
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

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

n_points = 100
epochs = 1000

lam = 1
u = 0.1
conv_points = []

x_init_line = np.linspace(x_start,x_end,n_points)

# Add some noise to the init line to make sure it doesn't get a stuck gradient
noise = np.random.normal(0,0.001,size=x_init_line.shape)
noise[0] = [0.0,0.0]
noise[-1] = [0.0,0.0]

x_init_line += noise


for j in range(len(obstacles)):
    ax[0].add_patch(plt.Circle(obstacles[j][0],obstacles[j][1],color=obstacles[j][2]))


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

def objective_function(x, lam=1, u=2):
    # Objective Value
    objective_value = np.sum(f_L(x)+lam*f_S(x)+u*f_O(x))

    # Gradient
    gradient = gradient_f_L(x) + gradient_f_S(x) + gradient_f_O(x)

    return objective_value, gradient


ax[0].plot(x_init_line[:, 0], x_init_line[:, 1], marker='.', label="Initial Path")

# Momentum
velocity = np.zeros_like(x_init_line)

def momentum_step(x,gradient,velocity,lr=0.005,beta=0.9):
    velocity = beta * velocity - lr * gradient
    x[1:-1] = x[1:-1] + velocity[1:-1]
    return x, velocity 

v_adam = np.zeros_like(x_init_line)
s_adam = np.zeros_like(x_init_line)
t = 0

def adamw_step(x, adam_gradient, v, s, t, lr=0.001, gamma_v=0.9, gamma_s=0.999, epsilon=1e-8, weight_decay=0.01):
    t += 1

    v = gamma_v * v - (lr * adam_gradient)
    s = gamma_s * s + (1 - gamma_s) * (adam_gradient**2)

    v_hat = v / (1 - gamma_v**t)
    s_hat = s / (1 - gamma_s**t)

    decay = lr * weight_decay * x[1:-1]
    next_x = (1.0 / (epsilon + an.sqrt(s_hat[1:-1]))) * v_hat[1:-1]

    x[1:-1] = x[1:-1] + next_x - decay

    return x, v, s, t

def newton_step(x, lam=1, u=2, epsilon=1e-8, stepsize=0.5, stopcrit=50):
    x_0 = np.copy(x)
    step = 1
    Delta_val = 1.0

    # Objective fnc for flattened array
    def f_obj(x_flat):
        x_reshaped = x_flat.reshape(-1,2)
        full_path = an.vstack([x[0], x_reshaped, x[-1]])
        return f_L(full_path) + lam * f_S(full_path) + u * f_O(full_path)

    while Delta_val > epsilon and step < stopcrit:
        x_current = x[1:-1].flatten()

        grad_f = grad(f_obj)(x_current)
        H = hessian(f_obj)(x_current)

        eigenvalues, eigenvectors = np.linalg.eigh(H)
        eigenvalues = np.maximum(eigenvalues, epsilon)
        H_mod = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        Delta_arr = np.linalg.inv(H_mod) @ grad_f

        if stopcrit / step > 0.50:
            x_current = x_current - stepsize * Delta_arr
        else:
            x_current = x_current - Delta_arr

        x[1:-1] = x_current.reshape(-1,2)
        Delta_val = np.linalg.norm(Delta_arr)
        print(f"This is iter.: {step} | Obj. Val.: {(f_obj)(x_current)} | DeltaNorm: {Delta_val:.8f}")
        conv_points.append((step, (f_obj)(x_current)))
        step += 1
    
    return x


best_line = x_init_line

current_mom_path = np.copy(x_init_line)
current_adamw_path = np.copy(x_init_line)
current_newton_path = np.copy(x_init_line)

min_mom_objective_value = np.inf
min_adamw_objective_value = np.inf

best_momentum = np.copy(x_init_line)
best_adamw = np.copy(x_init_line)
best_newton = np.copy(x_init_line)

# for e in range(epochs):
#     # Momentum
#     mom_objective_value, mom_gradient_array = objective_function(current_mom_path,lam=lam,u=u)    
    
#     if mom_objective_value < min_mom_objective_value:
#         min_mom_objective_value = mom_objective_value
#         best_momentum = np.copy(current_mom_path)

#     current_mom_path, velocity = momentum_step(current_mom_path, mom_gradient_array, velocity, lr=0.002, beta=0.6)
    
#     # AdamW
#     adamw_objective_value, adamw_gradient_array = objective_function(current_adamw_path,lam=lam,u=u)    

#     if adamw_objective_value < min_adamw_objective_value:
#         min_adamw_objective_value = adamw_objective_value
#         best_adamw = np.copy(current_adamw_path)

#     current_adamw_path, v_adam, s_adam, t = adamw_step(current_adamw_path, adamw_gradient_array, v_adam, s_adam, t, lr=0.0002, gamma_v=0.9, gamma_s=0.999, weight_decay=0.13)

# Newtons Method
start_time = time.time()
best_newton = newton_step(current_newton_path,lam=lam,u=u,stopcrit=100)
newton_objective_value, newton_gradient_array = objective_function(best_newton,lam=lam,u=u)    

# ax[0].plot(best_momentum[:, 0], best_momentum[:, 1], marker='.', label=f"Best Momentum Path | Obj. Val. {mom_objective_value:.2f}")
# ax[0].plot(best_adamw[:, 0], best_adamw[:, 1], marker='.', label=f"Best AdamW Path | Obj. Val. {adamw_objective_value:.2f}")
ax[0].plot(best_newton[:, 0], best_newton[:, 1], marker='.', label=f"Best Newton Path | Obj. Val. {newton_objective_value:.2f}")

ax[0].set_xlim(-1, 11)
ax[0].set_ylim(-1, 11)
ax[0].legend()

conv_steps, objective_val_point = zip(*conv_points)

ax[1].plot(conv_steps, objective_val_point, color="blue")
ax[1].set_ylabel("Objective Value")
#ax[1].legend()
ax[1].grid(True)


file_name = f"N_points: {n_points} | λ: {lam} | μ: {u} | Obj. Val.: {newton_objective_value:.2f} | Penalty 2 | RunTime: {(time.time() - start_time):.4f} secs"
ax[0].set_xlabel(file_name)
plt.suptitle("Newtons Method", fontsize=16)
plt.savefig(f"asg1/src/case1/plots/NewtonsMethhodN{n_points}Lam{lam}Mu{u}Penalty2ObjVal{newton_objective_value:.2f}.png")
plt.show()


