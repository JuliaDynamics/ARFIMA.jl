using PyPlot, ARFIMA, Random

Ns = [10_000, 50_000, 100_000]
σ = 0.5
φ = SVector(0.8)
θ = SVector(2.0)
d1 = 0.4; d2 = 2

xs = [zeros(length(Ns)) for i in 1:4]
args1 = (d1, nothing, nothing)
args2 = (d2, nothing, nothing)
args3 = (d1, φ, nothing)
args4 = (nothing, φ, θ)
args = [args1, args2, args3, args4]

labels = ["ARFIMA(0, d=$d1, 0)", "ARIMA(0, d=$d2, 0)",
 "ARFIMA(φ=$(φ[1]), d=$d1, 0)", "ARMA(φ=$(φ[1]), θ=$(θ[1]))"]

# compile
for (x, arg, label) in zip(xs, args, labels)
    t = arfima(MersenneTwister(5), 1000, σ, arg...)
end
time()

# %% Compute
println("Starting to bencmark:")
for (x, arg, label) in zip(xs, args, labels)
    println("\nProcess: $label")
    for (i, N) in enumerate(Ns)
        z = time()
        t = arfima(MersenneTwister(5), N, σ, arg...)
        x[i] = time() - z
        println("For N = $N, ", round(x[i], digits=7), " seconds")
        GC.gc()
    end
end

# %% Plot
# using PyPlot
# figure()
# subplot(121)
# for i in (1, 3)
#     x = xs[i]
#     loglog(Ns, x, label = labels[i], marker = "o")
# end
# legend()
# xlabel("timeseries length N")
# ylabel("time (sec)")
# subplot(122)
# for i in (2, 4)
#     x = xs[i]
#     semilogx(Ns, x, label = labels[i], marker = "o")
# end
# legend()
# xlabel("timeseries length N")
# ylabel("time (sec)")
# tight_layout()
