import Pkg; Pkg.add("PProf")
using Profile
using PProf
using Random
Random.seed!(42)

i64_256m = rand(Int64, 1 << 28)
f64_256m = randn(Float64, 1 << 28)

include("impl.jl")

# Collect a profile
Profile.clear()
@profile topk(i64_256m, 1 << 10)
@profile topk(f64_256m, 1 << 10)

# Export pprof profile and open interactive profiling web interface.
pprof()
