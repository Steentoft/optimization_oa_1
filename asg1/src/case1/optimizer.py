from scipy.optimize import minimize

def CG_optimizer(x, fun, iterations=100):
    new_x = x.copy()

    res = minimize(fun, new_x, method='CG', tol=1e-6, jac=True, options={'maxiter': iterations})

    return res

def BFGS_optimizer(x, fun, iterations=100):
    new_x = x.copy()

    res = minimize(fun, new_x, method='BFGS', tol=1e-6, jac=True, options={'maxiter': iterations})

    return res

def Newton_optimizer(x, fun, iterations=100):
    new_x = x.copy()

    res = minimize(fun, new_x, method='Newton-CG', tol=1e-6, jac=True, options={'maxiter': iterations})

    return res

def L_BFGS_B_optimizer(x, fun, iterations=100):
    new_x = x.copy()

    res = minimize(fun, new_x, method='L-BFGS-B', tol=1e-6, jac=True, options={'maxiter': iterations})

    return res

def TNC_optimizer(x, fun, iterations=100):
    new_x = x.copy()

    res = minimize(fun, new_x, method='TNC', tol=1e-6, jac=True, options={'maxfun': iterations})

    return res
