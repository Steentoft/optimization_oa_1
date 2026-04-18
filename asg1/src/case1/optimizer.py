from scipy.optimize import minimize

def CG_optimizer(x, fun, iterations=100):
    new_x = x.copy()

    res = minimize(fun, new_x, method='CG', tol=1e-6, jac=True, options={'maxiter': iterations})

    return res

def BFGS_optimizer(x, fun, iterations=100):
    new_x = x.copy()

    res = minimize(fun, new_x, method='BFGS', tol=1e-6, jac=True, options={'maxiter': iterations})

    return res

def Nelder_mead_optimizer(x, fun, iterations=100):
    new_x = x.copy()

    res = minimize(fun, new_x, method='Nelder-Mead', tol=1e-6, jac=True, options={'maxiter': iterations})

    return res
