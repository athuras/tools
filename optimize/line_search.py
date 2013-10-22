import numpy as np

def als(f, x0, grad_0, alpha0=1., beta=0.95, sigma=0.1, **kwargs):
    '''Armijo Line Search, w/ gradient descent, returns best step size t in
    the direction of the negative gradient.
    * f: objective function, accepting 1d-array
    * x0: initial solution
    * grad_0: gradient of f, evaluated at x0, 1d-array
    * alpha0: initial/max step size
    * beta: step depreciation coefficient, 0 < beta < 1.
    * sigma: boundary conditioner, 0 < sigma < 1

    optional keyword arguments:
    * 'maxiter': maximum number of iterations before break
    * 'dx': direction of descent, defaults to -1*unit(grad_0). 1d-array
    '''
    maxiter = kwargs.get('maxiter', 50)
    dx = kwargs.get('dx', grad_0)

    t = alpha0
    f_x0 = f(x0)
    d = -grad_0.dot(np.c_[dx / np.linalg.norm(dx)])

    i = 0
    while f(x0 + t*d) > sigma * t * np.dot(d.T, grad_0) + f_x0:
        if i > maxiter:
            break
        elif t < 10 * np.finfo(np.float).eps:
            break
        i += 1
        t *= beta

    return t


def armijo_qls(f, x0, grad_0, alpha0=1.5, beta=0.95, sigma=0.1, **kwargs):
    '''Min-Quadratic Search, Moves to the approximated quadratic vertex
    if the step isn't too big. Uses Armijo Line Search in event of failure.'''
    maxiter = kwargs.get('maxiter', 40)
    dx = kwargs.get('dx', grad_0)
    quad_limit = kwargs.get('quad_limit', 10)

    t = alpha0
    f_x0 = f(x0)
    d = -grad_0.dot(np.c_[dx / np.linalg.norm(dx)])
    i = 0

    # Quadratic Coefficients
    c = f_x0
    b = d
    a = f(x0 + t*grad_0) - b * t - c / t**2
    t = -b / (2. * a)

    while f(x0 + t*grad_0) > f_x0 + sigma + t * d:
        i += 1
        t *= beta
        a = f(x0 + t * grad_0) - b * t - c / t**2
        t = -b / (2. * a)
        if i > quad_limit:
            print 'Warning: Using Line-Search Instead'
            t = als(f, x0, grad_0)
        if i > maxiter:
            break
    return t
