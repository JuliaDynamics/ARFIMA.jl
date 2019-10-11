using PyPlot, BenchmarkTools, ARFIMA, Random

Ns = @. round(Int, 10 ^ (3.0:0.5:6))
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
 "ARFIMA(φ=$(φ[1]), d=$d1)", "ARMA(φ=$(φ[1]), θ=$(θ[1]))"]

# %%
time(); figure()
for (x, arg, label) in zip(xs, args, labels)
    for (i, N) in enumerate(Ns)
        z = time()
        t = arfima(MersenneTwister(5), N, σ, arg...)
        x[i] = time() - z
        println("x[$i] = ", x[i])
    end
    loglog(Ns, x, label = label)
end
xlabel("timeseries length N")
ylabel("time (sec)")
