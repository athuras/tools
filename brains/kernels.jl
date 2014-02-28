module Kernels
export ideal_LPF, gaussian_LPF

# Returns multiplicative mask. Suitable for application on
# a half-fourier coefficient array.
function ideal_LPF(freqs, cutoff)
    const sel = freqs .> 2pi*cutoff
    k = ones(size(freqs))
    k[sel] = 0
    return k
end

# elementwise gaussian kernel evaluation on frequency coeffs
function gaussian_LPF(freqs, bandwidth)
    const cutoff = 2pi*bandwidth
    k = exp(-(freqs .* freqs) ./ (cutoff * cutoff))
    return k
end
end
