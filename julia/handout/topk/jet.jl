using Pkg; Pkg.add("JET")
using JET
using Random
Random.seed!(42)

i64_16m = rand(Int64, 1 << 24)
f64_16m = randn(Float64, 1 << 24)

include("impl.jl")

# @code_warntype topk(i64_16m, 1 << 8)
# @code_warntype topk(f64_16m, 1 << 8)

@report_opt topk(i64_16m, 1 << 8)
@report_opt topk(f64_16m, 1 << 8)

@report_call topk(i64_16m, 1 << 8)
@report_call topk(f64_16m, 1 << 8)
