# import Pkg; Pkg.add("BenchmarkTools")
using BenchmarkTools
using Random
Random.seed!(42)

include("impl.jl")

i64_16m = rand(Int64, 1 << 24)
f64_16m = randn(Float64, 1 << 24)

@benchmark topk(i64_16m, 1 << 8)
@benchmark topk(f64_16m, 1 << 8)

# julia> @benchmark topk(i64_16m, 1 << 8)
# BenchmarkTools.Trial: 250 samples with 1 evaluation per sample.
#  Range (min … max):   9.851 ms … 56.513 ms  ┊ GC (min … max):  0.00% … 69.18%
#  Time  (median):     18.544 ms              ┊ GC (median):    44.30%
#  Time  (mean ± σ):   19.997 ms ±  5.818 ms  ┊ GC (mean ± σ):  46.24% ± 11.09%

#             █▇▅                                                
#   ▄▁▁▁▁▁▁▃▃▆███▆▇▄▃▃▂▃▂▃▃▃▃▃▂▁▂▁▁▃▂▁▁▁▁▁▁▁▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂ ▃
#   9.85 ms         Histogram: frequency by time        52.9 ms <

#  Memory estimate: 257.73 MiB, allocs estimate: 2023.

# julia> @benchmark topk(f64_16m, 1 << 8)
# BenchmarkTools.Trial: 154 samples with 1 evaluation per sample.
#  Range (min … max):  19.942 ms … 70.095 ms  ┊ GC (min … max):  0.00% … 67.72%
#  Time  (median):     30.604 ms              ┊ GC (median):    33.26%
#  Time  (mean ± σ):   32.673 ms ±  5.869 ms  ┊ GC (mean ± σ):  34.07% ±  9.93%

#                    █▄▇▂▁                                       
#   ▃▁▁▁▁▁▁▁▁▁▁▁▃▃▆▆▇██████▁▄▇▄▄▄▃▁▅▄▅▃▆▁▃▄▃▃▃▁▁▃▁▁▁▁▁▃▃▁▁▁▃▁▁▃ ▃
#   19.9 ms         Histogram: frequency by time        50.8 ms <

#  Memory estimate: 262.75 MiB, allocs estimate: 2153.