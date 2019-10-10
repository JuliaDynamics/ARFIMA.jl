module ARFIMA

using Random, Distributions, LinearAlgebra, StaticArrays
export arfima, SVector, @SVector

"""
    arfima([rng,] N, σ, d, φ=nothing, θ=nothing) -> Xₜ
Create a stochastic timeseries of length `N` that follows the ARFIMA
process, or any of its subclasses, like e.g. ARMA, AR, ARIMA, etc., see below.
`σ` is the standard deviation of the white noise used to generate the
process. The first optional argument is an `AbstractRNG`, a random
number generator to establish reproducibility.

The generating equation for `Xₜ` is:
```math
\\left( 1 - \\sum_{i=1}^p \\phi_i B^i \\right)
\\left( 1-B \\right)^d X_t
=
\\left( 1 + \\sum_{i=1}^q \\theta_i B^i \\right) \\varepsilon_t
```
with ``B`` the backshift operator and ``\\varepsilon_t`` white noise.

This equation encapsulates all possible variants of ARFIMA and Julia's
multiple dispatch system decides which will be the simulated variant,
based on the types of `d, φ, θ`.

## Variants
The ARFIMA parameters are (p, d, q) with `p = length(φ)` and `q = length(θ)`,
with `p, q` describing the autoregressive or moving average "orders" while
`d` is the differencing "order".
Both `φ, θ` can be of two types: `Nothing` or `SVector`. If they are `Nothing`
the corresponding components of autoregressive (φ) and moving average (θ)
are not done. Otherwise, the static vectors simply contain their values.

If `d` is `Nothing`, then the differencing (integrated)
part is not done and the process is in fact AR/MA/ARMA.
If `d` is of type `Int`, then the simulated process is in fact ARIMA,
while if `d` is `AbstractFloat` then the process is AR**F**IMA.
In the last case it must hold that `d ∈ (-0.5, 0.5)`.

If all `d, φ, θ` are `nothing`, white noise is returned.
"""
arfima(N::Int, args...) = arfima(Random.GLOBAL_RNG, N, args...)
arfima(rng::AbstractRNG, N, σ, d) = arfima(rng, N, σ, d, nothing, nothing)
arfima(rng::AbstractRNG, N, σ, d, φ) = arfima(rng, N, σ, d, φ, nothing)

arfima(rng, N, σ, ::Nothing, ::Nothing, θ) = generate_noise(rng, N, σ, θ) # MA

function arfima(rng, N, σ, d, φ::SVector{P}, θ) where {P} # AR(F)IMA
    L = estimate_past(φ)
    Z = arfima(rng, L+N, σ, d, nothing, θ) # Generate FIMA or IMA
    X = autoregressive(N, Z, φ) # autoregress on the result
end

function arfima(rng, N, σ, d::AbstractFloat, φ::Nothing, θ) # FIMA
    @assert -0.5 ≤ d ≤ 0.5 "For ARFIMA, it must be d ∈ (-0.5, 0.5)"
    M = 2N # infinite summation becomes summation over 2N
    noise = generate_noise(rng, M+N-1, σ, θ)
    # coefficients of truncated sum in inf. moving average representation
    ψ = zeros(M); ψ[1] = 1
    for k in 0:M-2
        @inbounds ψ[k+2] = ψ[k+1] * (d+k)/(k+1)
    end
    # Here we create the actual arfima process Xₜ
    X = zeros(N)
    for k in 1:N
        # TODO: optimize this dot product... (write loop)
        X[k] = dot(ψ, view(noise, k:k+M-1))
    end
    return X
end

function arfima(rng, N, σ, d::Int, φ::Nothing, θ) # IMA
    @assert d>0
    M = N + d
    noise = generate_noise(rng, M, σ, θ)
    differencing = SVector{d}(binomial.(d, 1:d)...)
    X = zeros(N)
    for k in 1:N
        X[k] = bdp(differencing, noise, d+k) + noise[d+k]
    end
    return X
end

function arfima(rng, N, σ, d::Nothing, φ::SVector{P}, θ) where {P} # AR(MA)
     noise = generate_noise(rng, N + P, σ, θ)
     L = estimate_past(φ)
     Z = generate_noise(rng, N + L, σ, θ) # white noise
     X = autoregressive(N, Z, φ)
end


generate_noise(rng, N, σ, θ::Nothing) = rand(rng, Normal(0, σ), N) # white noise
function generate_noise(rng, N, σ, θ::SVector{Q}) where {Q} # MA
    ε = generate_noise(rng, N+Q, σ, nothing)
    noise = zeros(N)
    θ = -θ # this change is necessary due to the defining equation
    # simply now do the average process
    for i in 1:N
        noise[i] = bdp(θ, ε, i+Q)
    end
    return noise
end

# TODO: after tests, add @inbounds
"""
    bdp(φ::SVector, X::AbstractVector, t)
Perform the backshift dot product between `φ` and `X`, with starting index `t`:
``\\sum_i \\phi_i X_{t-i}``.
"""
@generated function bdp(φ::SVector{P}, X::AbstractVector, t) where {P}
    exprs = [:(φ[$i]*X[t-$i]) for i in 1:P]
    ex = :(+($(exprs...)))
    return ex
end

"Estimate how long into the past to go for accurate AR process."
estimate_past(φ::SVector{P}) where {P} =
max(P+1, ceil(Int, log(0.001)/log(maximum(abs, φ))))


"""
    autoregressive(N, Z, φ::SVector{P}) -> X
Generate an autoregressive process based on input noise term `Z`.
This is used in both ARFIMA and ARMA.
"""
function autoregressive(N, Z, φ::SVector{P}) where {P}
    L = length(Z) - N; @assert L > P
    tmp = zeros(P)

    # Generate correct inital condition: the first P values of X
    for i = 1:L-P;
        y = bdp(φ, tmp, P+1) + Z[i];
        tmp[1:end-1] .= tmp[2:end] # shift values and add the new value
        tmp[end] = y
    end

    # X0 is now the "correct" X0, after L steps in advance
    X = zeros(N); X[1:P] .= tmp
    for i in (P+1):N
        X[i] = bdp(φ, X, i) + Z[i+L]
    end
    return X
end

end # module
