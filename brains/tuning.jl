# Functions that transform current, into 'activity'
module Tuning

export linear, lif

function linear(J, gain=1)
    if gain == 1
        return max(J, 0)
    else
        return max(J .* gain, 0)
    end
end

function lif(J, t_ref=0.002, t_rc=0.02)
    act(x) = 1. / (t_ref - t_rc * log(1 - 1 / x))
    a = zeros(size(J))
    const sel = J .> 1.
    a[sel] = act(J[sel])
    return a
end

end
