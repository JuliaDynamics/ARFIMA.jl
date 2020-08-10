using Test, ARFIMA, Statistics

N = 1000
σ = 2.3

x = arfima(N, σ, nothing) # white noise
@test abs(std(x) - σ) < 1
@test length(x) == N

x = arfima(N, σ, nothing, SVector(0.2, 0.2))
@test length(x) == N
@test all(!isinf, x)
x = arfima(N, σ, nothing, SVector(0.2,0.2), SVector(0.1))
@test length(x) == N
@test all(!isinf, x)
x = arfima(N, σ, 1, SVector(0.02,0.02), SVector(0.01))
@test length(x) == N
@test all(!isinf, x)
x = arfima(N, σ, 0.25, SVector(0.02,0.02), SVector(0.01))
@test length(x) == N
@test all(!isinf, x)
