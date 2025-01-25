# 原子类型定义，用于线程安全的操作
mutable struct Atomic{T}; @atomic x::T; end

@inbounds function topk_part(data::AbstractVector{T}, k::Int64, start::Int64) where T
    map(partialsortperm(data, 1:k, rev=true)) do x
        (data[x], x + start)
    end
end

@inbounds function topk(data::AbstractVector{T}, k) where T
    # n = Threads.nthreads()
    # chunk_size = length(data) ÷ n
    chunk_size = max(1 << 17, min(1 << 20, length(data) ÷ 256))
    # n = ceil(Int, length(data) / chunk_size)
    t = Threads.nthreads()

    # l = fill(ReentrantLock(), t)

    chunk_indices::Vector{Vector{Tuple{T, Int32}}} = Vector{Tuple{T, Int32}}()
    for _ in 1:t
        push!(chunk_indices, Vector{Tuple{T, Int32}}())
    end
    chunks = collect(enumerate(Iterators.partition(data, chunk_size)))
    # println(chunks)
    Threads.@threads for (i, chunk) in chunks
        # println(Threads.threadid(), " ", i)
        # chunk_indices[i] = topk_part(chunk, k, (i - 1) * chunk_size)

        j = Threads.threadid()
        # @assert j <= t

        sorted = topk_part(chunk, k, (i - 1) * chunk_size)
        # println(Threads.threadid(), " ", i, " ", chunk_size, " ", sorted)

        # lock(l[j])
        append!(chunk_indices[j], sorted)
        partialsort!(chunk_indices[j], 1:k, by=x->x[1], rev=true)
        resize!(chunk_indices[j], k)
        # unlock(l[j])

    end

    # Threads.@threads for chunk_index in chunk_indices
    #     partialsort!(chunk_index, 1:k, by=x->x[1], rev=true)
    #     resize!(chunk_index, k)
    # end
    all_indices = reduce(vcat, chunk_indices)
    # println(length(all_indices))
    map(partialsort(all_indices, 1:k, rev=true, by=x->x[1])) do x
        x[2]
    end

    # return chunk_indices

    # partialsortperm(data, 1:k, rev=true)
end
