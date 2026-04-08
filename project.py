import math

import numpy as np
import matplotlib.pyplot as plt


########## Task 1 ##########
x_start = np.array([0.0,0.0])
x_end = np.array ([7.0, 10.0])

obstacles = [
    plt.Circle((2, 4), 1.3, color='blue'),
    plt.Circle((5, 7), 1.0, color='orange'),
]

fig, ax = plt.subplots(figsize=(6, 6))

n_points = 75

x_init_line = np.linspace(x_start,x_end,n_points)

for i in obstacles:
    ax.add_patch(i)


ax.plot(x_init_line[:,0],x_init_line[:,1],marker='.',label="Initial Path")
ax.plot(x_start[0],x_start[1], 'go', markersize=5, label="start")
ax.plot(x_end[0],x_end[1], 'ro', markersize=5, label="end")

ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.legend()

plt.show()




########## Task 2 ##########
# obj_func = f_L + lambda * f_S + my * f_O

# def obj_func ():
    # path_length = 

def path_length(x):
    sum = 0
    for i in range(x-1):
        sum += abs(x[i+1]-x[i])**2
    return sum

def favour_smoothness(x):
    sum = 0
    for i in range(x-1):
        sum += abs(x[i+1]-2*x[i]+x[x-1])**2
    return sum

def avoid_obstacles(x):
    sum = 0
    for i in range(x):
        sum += penalty_1(x[i])
    return sum



def penalty_1(x):
    if True:
        1/(x)

def penalty_2(x):
    pass


# Adding something to penalize big spacing between points. Potentially recreating the line
# points so they are equally spread?

# For case 2, think of doing homemade optimizers.