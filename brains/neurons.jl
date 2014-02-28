module Neurons

using utils, tuning

export activity, ddot, decode, lif_fit, lin_fit
export lif_sim, current, lif_delta


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

# Calculate Params
function dual_point_lif_fit(x, a; t_ref=0.002, t_rc=0.02)
    const eps = t_rc / t_ref
    r = 1. / (t_ref .* a)
    f = (r .- 1) ./ eps
    alpha = (1. / exp(f[2]) - 1) - 1. / (exp(f[1]) - 1) ./ (x[2] - x[1])
    x_threshold = x[1] - 1 ./ (alpha * exp(f[2]) - 1)
    Jbias = 1 - alpha .* x_threshold
    return [alpha Jbias]
end

# linear tuning fit
# returns an array of alpha, bias terms, one for each of x_max
function lin_fit(x_max, x_int, y_max, enc)
    alpha = y_max ./ ddot((x_max - x_int), enc, 2)
    bias = - ddot(x_int, enc, 2) .* alpha
    return cat(2, alpha, bias)
end

# Possibly the same as 'activity', I try not to think about it...
function current(x::Array{Float64}, encoders::Array{Float64}, params::Array{Float64})
    return params[:,1] .* (encoders * x') .+ params[:,2]
end

lif_delta(J::Array{Float64}, V::Array{Float64}, t_rc=0.002) = (J .- V) ./ t_rc

# WARNING: Stuff is presumed to be Column-major for mad performance, i.e.
# x looks like [<x_0>;<x_1>;...<x_k>] (one vector per column)
# Type Annotations all over the place, mainly because when I was debugging shit
# was all over the place.
function lif_sim(x::Array{Float64},
                 dt::Float64,
                 encoders::Array{Float64},
                 params::Array{Float64};
                 t_ref=0.002, t_rc=0.02, spike_ref=1.)
    const k = size(x, 1)
    const n = size(params, 1)
    refractory_steps = int(t_ref / dt)  # BEWARE roundoff error

    V_sim = Array(Float64, n, k)
    V_sim[:,1] = 0.  # Initial state of zero
    J = current(x, encoders, params)::Array{Float64}
    ref_state = zeros(Int64, n)  # The refractory state of each neuron

    # Yo dawg, we heard you like indexing...
    for i in 2:k
        dv_dt = lif_delta(J[:, i], V_sim[:, i - 1], t_rc) .* dt
        V_sim[:, i] = V_sim[:, i - 1] .+ dv_dt::Array{Float64}  # Accumulate, business as usual
        V_sim[ref_state .> 0, i] = 0.  # floor the recovering neurons
        spikes = V_sim[:, i] .> 1.
        V_sim[spikes, i] = spike_ref  # Set the spikes to the reference value
        V_sim[:, i] = max(V_sim[:, i], 0)  # because shenanigans happened before ...
        ref_state -= 1
        ref_state[spikes] = refractory_steps  # Set the spike counters
    end
    return V_sim  # The voltage 'chart' one row-per-neuron.
end
end
