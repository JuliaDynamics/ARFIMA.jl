using Test, ARFIMA, Statistics, Random, StatsBase

N = 1000
σ = 2.3

@testset "Size Validation" begin
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
end

@testset "ARMA" begin
    # white noise
    rng = Random.MersenneTwister(1)
    noise = arma(rng, N, σ, nothing, nothing);
    @test std(noise) ≈ σ atol=0.01

    # AR
    @testset "AR Noise" for φ in [SVector(zeros(1)...), SVector(zeros(3)...), SVector(zeros(5)...)]
        rng = Random.MersenneTwister(1)
        x = arma(rng, N, σ, φ, nothing); # produce white noise
        @test autocor(noise, 1:2) ≈ autocor(x, 1:2) atol=0.05
        @test mean(noise - x) ≈ 0.0 atol=0.05
    end

    rng = Random.MersenneTwister(1)
    φ = 0.8
    ar = arma(rng, 100*N, σ, SVector(φ), nothing);
    @test [φ, φ^2] ≈ autocor(ar, 1:2) atol=0.01

    # MA
    @testset "MA Noise" for θ in [SVector(zeros(1)...), SVector(zeros(3)...), SVector(zeros(5)...)]
        rng = Random.MersenneTwister(1)
        x = arma(rng, N, σ, nothing, θ); # produce white noise
        @test autocor(noise, 1:2) ≈ autocor(x, 1:2) atol=0.05
        @test mean(noise - x) ≈ 0.0 atol=0.05
    end

    rng = Random.MersenneTwister(1)
    θ = 0.96
    ma = arma(rng, 100*N, σ, nothing, SVector(θ));
    @test [-θ/(1+θ^2),0] ≈ autocor(ma, 1:2) atol=0.01

    # ARMA
    @testset "ARMA Noise" for θ in [SVector(zeros(1)...), SVector(zeros(2)...)],
                              φ in [SVector(zeros(1)...), SVector(zeros(3)...), SVector(zeros(5)...)]
        rng = Random.MersenneTwister(1)
        x = arma(rng, N, σ, φ, θ); # produce white noise
        @test autocor(noise, 1:2) ≈ autocor(x, 1:2) atol=0.05
        @test mean(noise - x) ≈ 0.0 atol=0.05
    end

    rng = Random.MersenneTwister(1)
    φ = 0.7
    θ = 0.2
    x = arma(rng, 100*N, σ, SVector(φ), SVector(θ));
    ρ(k) = k == 0 ? (1+φ*θ)*(φ+θ)/(1+2φ*θ+θ^2) : φ*ρ(k-1)
    @test ρ.(1:3) ≈ autocor(x, 1:3) atol=0.05
end
