module Neurons

using utils, tuning

export activity, ddot, decode, lif_fit, lin_fit


activity(x, enc, p) = x * enc' .* p[:,1]' .+ p[:,2]'

ddot(x, y, k) = sum(x .* y, k)

# where noise must be scalar, stationary independent etc.
function decode(A, x, dx, noise=0)
    Gamma = A' * A * dx
    Gamma += diagm(repmat([noise], size(Gamma)[1]))
    Upsilon = A' * x * dx
    return Gamma \ Upsilon
end


# Fit alpha, and bias terms for so-called 'leaky integrate and fire'
# tuning curves.
function lif_fit(x_max, x_int, y_max, enc, t_ref=0.002, t_rc=0.02)
    const B = exp((1. / t_rc) .* (t_ref - 1. / y_max))
    alpha = (1 ./ (1. - B) - 1) ./ ddot(x_max - x_int, enc, 2)
    bias = 1. - alpha .* ddot(x_int, enc, 2)
    return cat(2, alpha, bias)
end


# linear tuning fit
# returns an array of alpha, bias terms, one for each of x_max
function lin_fit(x_max, x_int, y_max, enc)
    alpha = y_max ./ ddot((x_max - x_int), enc, 2)
    bias = - ddot(x_int, enc, 2) .* alpha
    return cat(2, alpha, bias)
end
end
