import math

import numpy as np
import matplotlib.pyplot as plt


########## Task 1 ##########
x_start = np.array([0.0,0.0])
x_end = np.array ([7.0, 10.0])

# (Point, Radius, Color)
obstacles = [
    (np.array([2,4]), 1.3, 'blue'),
    (np.array([5,7]), 1.0,'orange')
]

fig, ax = plt.subplots(figsize=(6, 6))

n_points = 75

x_init_line = np.linspace(x_start,x_end,n_points)

for i in range(len(obstacles)):
    ax.add_patch(plt.Circle(obstacles[i][0],obstacles[i][1],color=obstacles[i][2]))


ax.plot(x_init_line[:,0],x_init_line[:,1],marker='.',label="Initial Path")
ax.plot(x_start[0],x_start[1], 'go', markersize=5, label="start")
ax.plot(x_end[0],x_end[1], 'ro', markersize=5, label="end")

ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.legend()

plt.show()

########## Task 2 ##########
# obj_func = f_L + lambda * f_S + my * f_O


def path_length(x):
    sum = 0.0
    for i in range(len(x)-1):
        sum += abs(x[i+1]-x[i])**2
    return sum

def favour_smoothness(x):
    sum = 0.0
    for i in range(len(x)-1):
        sum += abs(x[i+1]-2*x[i]+x[x-1])**2
    return sum

def avoid_obstacles(x):
    sum = 0.0
    for i in range(len(x)):
        sum += penalty_1(x[i])
    return sum

def circular_obstacle(x, obstacle):
    return abs(x-obstacle[0])


def penalty_1(x):
    penalty = 0.0
    for i in range(len(obstacles)):
        if circular_obstacle(x, obstacles[i]) > obstacles[i][1]:
            penalty += 1/(circular_obstacle(x, obstacles[i])-obstacles[i][1])**2
        else:
            penalty += math.inf
    return penalty


def penalty_2(x):
    pass

def objective_function(x, lam=1, u=1):
    return path_length(x)+lam*favour_smoothness(x)+u*avoid_obstacles(x)

epochs = 10
for e in range(epochs):
    objective_function(x_init_line)


# Adding something to penalize big spacing between points. Potentially recreating the line
# points so they are equally spread?

# For case 2, think of doing homemade optimizers.