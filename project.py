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

for j in range(len(obstacles)):
    ax.add_patch(plt.Circle(obstacles[j][0],obstacles[j][1],color=obstacles[j][2]))


########## Task 2 ##########

def path_length(x):
    sum = 0.0
    for i in range(len(x)-1):
        sum += abs(x[i+1]-x[i])**2
    return sum

def favour_smoothness(x):
    sum = 0.0
    for i in range(len(x)-1):
        sum += abs(x[i+1]-2*x[i]+x[i-1])**2
    return sum

def avoid_obstacles(x):
    sum = 0.0
    for i in range(len(x)):
        sum += penalty_2(x[i])
    return sum

def circular_obstacle(x, obstacle):
    return abs(x-obstacle[0])

# Attempt at point length penalizing
def longest_point(x):
    longest_distance = np.linalg.norm(x[1]-x[0])
    for i in range(len(x)-1):
        point_dist = np.linalg.norm(x[i+1]-x[i])
        if longest_distance < point_dist:
            longest_distance = point_dist
    return longest_distance


def penalty_1(x):
    penalty = 0.0
    for i in range(len(obstacles)):
        if (np.linalg.norm(circular_obstacle(x, obstacles[i]) > obstacles[i][1])):
            penalty += 1/(circular_obstacle(x, obstacles[i])-obstacles[i][1])**2
        else:
            penalty += math.inf
    return penalty

def penalty_2(x, alpha=1):
    penalty = 0.0
    for i in range(len(obstacles)):
        penalty += np.exp(-alpha*(circular_obstacle(x, obstacles[i])**2-obstacles[i][1]**2))
    return penalty

def objective_function(x, lam=1, u=1, epsilon=1):
    return sum(path_length(x)+lam*favour_smoothness(x)+u*avoid_obstacles(x)+longest_point(x)*epsilon), np.gradient(x)

# NOT WORKING
def momentum_step(x,mom_v,lr=0.1,mom_decay=0.1):
    mom_v = np.zeros_like(x[:-1])
    print(x[:-1])
    mom_v = mom_decay * mom_v - lr * np.linalg.norm(np.gradient(x[:-1]))
    return x[:-1] + mom_v

epochs = 20
best_line = x_init_line
best_objective_value, best_gradient_array = objective_function(best_line)

ax.plot(best_line[:, 0], best_line[:, 1], marker='.', label="Initial Path")

for e in range(epochs):
    new_line = np.copy(best_line)

    # for n in range(1, len(new_line - 1)):
    #     # new_line[n] = new_line[n] + 0.01*(penalty_2(new_line[n]) * np.gradient(new_line[n]))
    #     # new_line[n] = new_line[n] + np.array([2,0.3])
    
    new_objective_value, new_gradient_array = objective_function(new_line)    
    
    new_line = momentum_step(new_line,new_gradient_array)

    ax.plot(new_line[:, 0], new_line[:, 1], marker='.', label=f"New Path no {e}")


    if new_objective_value < best_objective_value:
        best_line = new_line



ax.plot(best_line[:, 0], best_line[:, 1], marker='.', label="Best Path")
print(longest_point(best_line))

ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)
ax.legend()

plt.show()



# Adding something to penalize big spacing between points. Potentially recreating the line
# points so they are equally spread?

# Path length doesn't work, but maybe a longest distance between two points should be a objective value as well.

# For case 2, think of doing homemade optimizers.