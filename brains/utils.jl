module Utils
export asel, last_dim, meshgrid, rmse

function asel(a, sel)
    [a[s] for s in sel]
end

last_dim(x) = length(size(x))

function meshgrid{T}(vx::AbstractVector{T}, vy::AbstractVector{T})
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repmat(vx, m, 1), repmat(vy, 1, n))
end

rmse(x) = sqrtm(mean(x.^2))
end
