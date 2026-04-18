import numpy as np
from autograd import grad, hessian
import autograd.numpy as an


def newtonsmethod(x, fun, args):
    step = 1
    Delta_val = 1.0
    this_x = np.copy(x)

    convergence_points = []
    stopcrit, epsilon, stepsize = args[0], args[1], args[2]
    obj_val_only = lambda x_val: fun(x_val)[0]
    H_func = hessian(obj_val_only)

    while Delta_val > epsilon and step < stopcrit:
        x_current = this_x[1:-1].flatten()

        objective_value, grad_f = fun(x_current)

        H = H_func(x_current)

        eigenvalues, eigenvectors = np.linalg.eigh(H)
        eigenvalues = np.maximum(eigenvalues, epsilon)
        H_mod = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        Delta_arr = np.linalg.inv(H_mod) @ grad_f

        if stopcrit / step > 0.50:
            x_current = x_current - stepsize * Delta_arr
        else:
            x_current = x_current - Delta_arr

        this_x[1:-1] = x_current.reshape(-1,2)
        Delta_val = np.linalg.norm(Delta_arr)
        convergence_points.append((step,fun(x_current)[0]))
        step += 1

    return this_x, convergence_points