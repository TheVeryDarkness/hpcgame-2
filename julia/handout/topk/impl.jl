# 原子类型定义，用于线程安全的操作
mutable struct Atomic{T}; @atomic x::T; end

@inbounds function topk_part(data::AbstractVector{T}, k::Int64, start::Int64) where T
    partialsortperm(data, 1:k, rev=true) .+ start
end

@inbounds function topk(data::AbstractVector{T}, k) where T
    # n = Threads.nthreads()
    # chunk_size = length(data) ÷ n
    chunk_size = 1 << 17
    n = ceil(Int, length(data) / chunk_size)

    chunk_indices::Vector{Vector{Int}} = fill(Vector{Int}(), n)
    Threads.@threads for (i, chunk) in collect(enumerate(Iterators.partition(data, chunk_size)))
        chunk_indices[i] = topk_part(chunk, k, (i - 1) * chunk_size)
    end
    all_indices = reduce(vcat, chunk_indices)
    # println("Max = ", maximum(chunk_indices), " ", length(data))
    partialsort(all_indices, 1:k, rev=true, by=x->data[x])

    # partialsortperm(data, 1:k, rev=true)
end


# Threads.@threads for (i, chunk) = collect(enumerate(Iterators.partition([1, 2, 3], 2)))
#     println(chunk)
# end

# Threads.@threads for (i, chunk) = enumerate(Iterators.partition([1, 2, 3], 2))
#     println(chunk)
# end

# Threads.@threads for i = 1:10
#     println(Threads.threadid())
# end
