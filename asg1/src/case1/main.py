from objective_function import objective_function, objective_function_op
from gradient_descent import gradient_descent
from momentum import momentum
from adamw import adamw
from newtonmethod import newtonsmethod
from optimizer import *

import matplotlib.pyplot as plt
import numpy as np
import time
for z in [20,50, 100]:
    for x in [0.1,1,10]:
        for y in [0.1, 1, 10]:
            ########## Task 1 ##########
            x_start = np.array([0.0, 0.0])
            x_end = np.array([7.0, 10.0])

            # (Point, Radius, Color)
            obstacles = [
                # (np.array([2, 4]), 1.3, 'blue'),
                # (np.array([4, 8]), 1.0, 'orange')
                #(np.array([2,4]), 1.3, 'blue'),
                #(np.array([5,7]), 1.0,'orange'),
                (np.array([3.5,5]), 2.5, 'blue'),
            ]

            n_points = z
            lam = x
            u = y
            learning_rate = 0.02
            # Normally learning_rate is 0.002

            x_init_line = np.linspace(x_start, x_end, n_points)

            # Add some noise to the init line to make sure it doesn't get a stuck gradient
            noise = np.random.normal(0,0.001,size=x_init_line.shape)
            noise[0] = [0.0,0.0]
            noise[-1] = [0.0,0.0]

            x_init_line += noise

            iterations = 500

            obj_fun = lambda x: objective_function(x, obstacles, lam, u)
            obj_fun_op = lambda x: objective_function_op(x, x_init_line, obstacles, lam, u)

            ### Plotting
            fig, ax = plt.subplots(2,2, figsize=(12, 12))

            init_obj_val = obj_fun(x_init_line)[0]

            ax[0][0].plot(x_init_line[:, 0], x_init_line[:, 1], 'purple', marker='.', label=f"Initial Path | Obj. Val.: {init_obj_val:.2f}")
            ax[1][0].plot(x_init_line[:, 0], x_init_line[:, 1], 'purple', marker='.', label=f"Initial Path | Obj. Val.: {init_obj_val:.2f}")

            print(init_obj_val)

            for j in range(len(obstacles)):
                ax[0][0].add_patch(plt.Circle(obstacles[j][0], obstacles[j][1], color=obstacles[j][2]))
                ax[1][0].add_patch(plt.Circle(obstacles[j][0], obstacles[j][1], color=obstacles[j][2]))


            def plot_inner_flat_line(x):
                new_line = x_init_line.copy()
                x = x.reshape((-1, 2))
                new_line[1:-1] = x
                return new_line


            functions = [
                { "func" : gradient_descent, "name" : "Gradient Descent", "col" : 'red', "args" : [iterations, learning_rate]},
                { "func" : momentum, "name" : "Momentum", "col" : 'green', "args" : [iterations, learning_rate, 0.9]},
                { "func" : adamw, "name" : "AdamW", "col" : 'orange', "args" : [iterations, learning_rate, 0.9, 0.999, 1e-8, 0.01]},
                { "func" : newtonsmethod, "name" : "Newton's Method", "col" : 'pink', "args" : [iterations, 1e-8, 0.5]}
            ]

            optimizers = [
                { "func" : CG_optimizer, "name" : "CG-OPT", "col" : "goldenrod"},
                { "func" : BFGS_optimizer, "name": "BFGS-OPT", "col": "aqua"},
            # { "func" : Nelder_mead_optimizer, "name": "Nelder-Mead-OPT", "col": "lime"},
            ]



            def main():
                best_obj_vals = np.inf

                for function in functions:
                    print("Running function: ", function["name"])
                    start_time = time.time()
                    best_line, convergence_points = function["func"](x_init_line, obj_fun, function["args"])

                    best_line = best_line.reshape((-1, 2))

                    conv_steps, objective_val_point = zip(*convergence_points)
                    
                    init_val = objective_val_point[0]
                    final_val = objective_val_point[-1]
                    
                    f_init = f"{init_val:.2e}" if abs(init_val) >= 10000 else f"{init_val:.2f}"
                    f_final = f"{final_val:.2e}" if abs(final_val) >= 10000 else f"{final_val:.2f}"
                    
                    ax[0][0].plot(best_line[:, 0], best_line[:, 1], marker='.', color=function["col"], label=f"{function["name"]} | Obj. Val.: {f_final} | Runtime: {time.time()-start_time:.2f} secs")
                    ax[0][1].plot(conv_steps,objective_val_point, marker='.', color=function["col"], label=f"{function['name']} | Init. Obj. Val.: {f_init}")

                    best_obj_vals = final_val if best_obj_vals > final_val else best_obj_vals


                for optimizer in optimizers:
                    print("Running function: ", optimizer["name"])

                    new_line = x_init_line[1:-1].flatten()

                    start_time = time.time()

                    convergence_points = []
                    for iteration in range(iterations):
                        res = optimizer["func"](new_line, obj_fun_op, 1)
                        new_line = res.x
                        convergence_points.append((iteration,res.fun))

                    rebuilt_x = plot_inner_flat_line(res.x)
                    conv_steps, objective_val_point = zip(*convergence_points)
                    ax[1][0].plot(rebuilt_x[:, 0], rebuilt_x[:, 1], marker='.', color=optimizer["col"], label=f"{optimizer["name"]} | Final Obj. Val.: {objective_val_point[-1]:.2f} | Runtime: {time.time()-start_time:.2f} secs")
                    ax[1][1].plot(conv_steps,objective_val_point, marker='.', color=optimizer["col"], label=f"{optimizer["name"]} | Init. Obj. Val.: {objective_val_point[0]:.2f}")

                ax[0][0].set_xlim(-0.5, 11)
                ax[0][0].set_ylim(-0.5, 11)
                ax[1][0].set_xlim(-0.5, 11)
                ax[1][0].set_ylim(-0.5, 11)

                ax[0][0].legend(fontsize=7)
                ax[1][0].legend(fontsize=7)

                ax[0][1].set_xlabel("Iterations")
                ax[0][1].set_ylabel("Objective Value")
                ax[1][1].set_xlabel("Iterations")
                ax[1][1].set_ylabel("Objective Value")

                ax[0][1].set_ylim(-0.5, init_obj_val+5)
                ax[1][1].set_ylim(-0.5, init_obj_val+5)

                ax[0][1].legend(fontsize=7)
                ax[1][1].legend(fontsize=7)

                ax[0][1].grid()
                ax[1][1].grid()

                print(f"{(best_obj_vals / init_obj_val) * 100:.2f}%")

                plt.suptitle(f"N_points: {n_points} | λ: {lam} | μ: {u} | Learning Rate: {learning_rate}")
                plt.savefig(f"asg1/src/case1/plots/Npoint{n_points}Lam{str(lam).replace('.', ',')}Mu{str(u).replace('.', ',')}Obs-2hit2Pen1")
                #plt.show()

            if __name__=="__main__":
                main()