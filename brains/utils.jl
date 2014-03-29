module Utils
# A collection of vaguely useful functions common to NEF.

export asel, meshgrid, rmse
export nhypot, normalise
export sphere_to_euclidean, sphere_scaling, rand_sphere, rand_normal

function sphere_to_euclidean(x)
    n, k = size(x)
    if n == 1
        return x
    end

    r = x[1,:]
    phi = x[2:end,:]

    sines = begin
        z = Array(Float64, n, k)
        z[1,:] = 1.
        z[2:end,:] = sin(phi)
        z
    end
    cosines = begin
        z = Array(Float64, n, k)
        z[1:end-1,:] = cos(phi)
        z[end,:] = 1.
        z
    end
    return sines .* cosines .* r
end

# d -> [1, ...2pi..., pi]
function sphere_scaling(d, r=1)
    pies = max(d - 2, 0)
    final = max(d - pies - 1, 0)
    scaling = cat(1, [r], repmat([pi], pies), repmat([2pi], final))
    return scaling
end

function rand_sphere(d, n; r=1.)
    scale = sphere_scaling(d, r)
    z = rand(d, n) .* scale
    # Make roughly half of the radii negative
    s = [-1, 1]
    mult = s[rand(1:2, n)]
    z[1, :] .*= r ./ abs(z[1, :])  .* mult' # Radius normalization
    return z
end

function asel(a, sel)
    [a[s] for s in sel]
end

# Naive hypot, can overflow
# Col-major
function nhypot(a)
   return sqrt(sum(a .* a, 1))
end

# Col-major
function normalise(a)
    return a ./ nhypot(a)
end

function meshgrid{T}(vx::AbstractVector{T}, vy::AbstractVector{T})
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repmat(vx, m, 1), repmat(vy, 1, n))
end

rmse(x) = sqrtm(mean(x.^2))
random_normal(mean, stddev, size) = randn(size...) .* stddev .+ mean
end
