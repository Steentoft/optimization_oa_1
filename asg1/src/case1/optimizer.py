from scipy.optimize import minimize

def CG_optimizer(x, fun, iterations=100):
    new_x = x.copy()

    res = minimize(fun, new_x, method='CG', tol=0.01, jac=True, options={'maxiter': iterations})

    return res