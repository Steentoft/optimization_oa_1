import numpy as np
import matplotlib.pyplot as plt


########## Task 1 ##########
x_start = np.array([0.0,0.0])
x_end = np.array ([7.0, 10.0])

obstacles = [
    {"center":np.array([2,4]),"radius":1.3,"col":'blue'},
    {"center":np.array([5,7]),"radius":1.0,"col":'orange'}
]

fig, ax = plt.subplots(figsize=(6, 6))

n_points = 75

x_init_line = np.linspace(x_start,x_end,n_points)

for i in range(len(obstacles)):
    ax.add_patch(plt.Circle(obstacles[i]["center"],obstacles[i]["radius"],color=obstacles[i]["col"]))


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




# Adding something to penalize big spacing between points. Potentially recreating the line
# points so they are equally spread?

# For case 2, think of doing homemade optimizers.