import numpy as np
from autograd import grad, hessian

def newtonsmethod(x, fun, stopcrit, args):
    x_0 = np.copy(x)
    step = 1
    Delta_val = 1.0

    stopcrit, lam, u, epsilon, stepsize = args[0], args[1], args[2], args[3], args[4] 

    while Delta_val > epsilon and step < stopcrit:
        x_current = x[1:-1].flatten()

        grad_f = grad(fun)(x_current)
        H = hessian(fun)(x_current)

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
        print(f"This is iter.: {step} | Obj. Val.: {(fun)(x_current)} | DeltaNorm: {Delta_val:.8f}")
        #conv_points.append((step, (fun)(x_current)))
        step += 1
    
    return x