module Signals

import Kernels: ideal_LPF

export rms, rescale_rms, wgn_coeffs, mask_coeffs
export gen_wgn_signal, wgn_signal

rms(x) = sqrt((1. / prod(size(x))) * sum(x .* x))
rescale_rms(x, z::Real) = (z / rms(x)) .* x

# generate a colored noise signal, whose color is based on the application
# of a DFT coefficient 'mask_fn'. For examples of mask_fns, see module
# Kernels
function gen_wgn_signal(T::Real, dt::Real, rms_power::Real,
                        mask_fn::Function=ideal_LPF,
                        mask_params...)
    const n = int(T / dt)
    freqs, coeffs = wgn_coeffs(T, dt)
    coeffs = mask_coeffs(freqs, coeffs, mask_fn, mask_params...)
    x = irfft(coeffs, n)  # Shenanigans with the size ...
    return rescale_rms(x, rms_power)
end

# Multi-dimensional version of gen_wgn_signal
function wgn_signal(d::Integer, T, dt, rms=Real;
        mask_fn::Function=ideal_LPF, mask_params=[15])
    z = [gen_wgn_signal(T, dt, rms, mask_fn, mask_params...)'
            for i in 1:d]
    return cat(1, z...)
end

# Return the frequencies, and fourier coefficients of white noise spectrum
# of a Real signal of period T, sample_period dt.
function wgn_coeffs(T::Real, dt::Real)
    const n = int(T / dt)
    const n_half = div(n, 2)
    freqs = 2pi/dt .* [i / n for i in 0:n_half]
    coeffs = complex(randn(n_half + 1), randn(n_half + 1))
    coeffs[1] = 0  # No DC component
    return freqs, coeffs
end

# Apply the mask function to the DFT coefficients coeffs.
function mask_coeffs(freqs, coeffs, mask_fn::Function, mask_params...)
    mask = mask_fn(freqs, mask_params...)
    return coeffs .* mask
end
end
