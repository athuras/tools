import numpy as np
import utils

def lif(J, tau_ref=0.002, tau_rc=0.02):
    '''Leaky Integrate-and-Fire neuron Tuning Curve/Surface/Tensor...
    J, tau_ref, tau_rc must be array-like.'''
    calc_activity = lambda x: 1. / (tau_ref - tau_rc * np.log(1. - 1. / x))
    a = np.zeros_like(J)
    sel = np.where(J > 1.)
    a[sel] = calc_activity(J[sel])
    return a

def rect_lin(J, gain=1.):
    '''Rectified Linear with slope=gain'''
    return np.maximum(J, 0) if gain == 1. else np.maximum(J * gain, 0)

def activity(x, encoders, alpha, bias):
    '''Implements J = alpha * x.dot(encoders) + bias, expands dimensions if
    all arguments are 1-D. Result is of shape (x.shape[0], encoders.shape[0])
    * x: The range over which to render the functions
    * encoders: Vectors which dot-x
    * alpha: 1D gain coefficient
    * bias: 1D additive bias term'''
    (x, encoders, alpha, bias) = utils.force_array((x, encoders, alpha, bias))
    return utils.dot_expand(x, encoders) * alpha + bias

def linear_tuning_fit(x_targets, x_intercepts, y_targets, encoders):
    '''Returns [[alpha, J_bias]] for rect_linear tuning curve'''

    (x_targets, x_intercepts,
     y_targets, encoders) = utils.force_array((x_targets, x_intercepts,
                                               y_targets, encoders))
    alphas = None
    J_biases = None
    alphas = y_targets / utils.diag_dot(x_targets - x_intercepts, encoders)
    J_biases = -utils.diag_dot(x_intercepts, encoders) * alphas
    return np.vstack((alphas, J_biases))

def lif_fit(x_max, x_intercepts, y_targets, encoders, t_ref=0.002, t_rc=0.02):
    '''Returns [[alpha, J_bias]] for 'leaky-integrate-and-fire'/lif tuning model
    '''
    (x_max, x_intercepts,
     y_targets, encoders) = utils.force_array((x_max, x_intercepts,
                                               y_targets, encoders))
    B = np.exp((1./t_rc) * (t_ref - 1 / y_targets))
    alpha = (1. / (1. - B) - 1.) / utils.diag_dot(x_max - x_intercepts, encoders)
    bias = 1. - alpha * utils.diag_dot(x_intercepts, encoders)
    return np.vstack((alpha, bias))

def lstsq_decode(A, x, dx=None, noise=None):
    return lin_decode(A, x, dx, noise, lstsq=True)

def lin_decode(A, x, dx=None, noise=None, lstsq=False):
    '''Given the "Activity" Matrix A, determine the least-squares linear decoder.
    *A: Activity Matrix, of shape (m, n), where
        m = encoder_range / dx,
        n = number of neurons
    *x: Original Signal (value that was encoded)
    *noise: stationary, independent noise variance, optional
    *lstsq: Use linalg.lstsq regression to solve, otherwise use 'solve' routine
    '''
    m, n = A.shape
    dx = (x.max() - x.min()) / float(m) if dx is None else dx
    Gamma = np.dot(A.T, A) * dx
    Upsilon = np.dot(A.T, x) * dx
    if noise is not None:
        Gamma += np.diag(np.repeat(noise, n))

    if lstsq:  # Use Least-Squares Regression
        return np.linalg.lstsq(Gamma, Upsilon)[0]
    else:  # Use LAPACK _gesv
        return np.linalg.solve(Gamma, Upsilon)

