# import Pkg; Pkg.add("PProf")
using Profile
using PProf
using Random
Random.seed!(42)

include("impl.jl")

i64_16m = rand(Int64, 1 << 24)
f64_16m = randn(Float64, 1 << 24)

# Collect a profile
Profile.clear()
@profile topk(i64_16m, 1 << 8)
@profile topk(f64_16m, 1 << 8)

# Export pprof profile and open interactive profiling web interface.
pprof()
