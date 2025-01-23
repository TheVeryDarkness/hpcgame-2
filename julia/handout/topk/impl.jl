# 原子类型定义，用于线程安全的操作
mutable struct Atomic{T}; @atomic x::T; end

function topk_part(data::AbstractVector{T}, k::Int64, start::Int64) where T
    map(partialsortperm(data, 1:k, rev=true)) do i
        i + start
    end
end

@inbounds function topk(data::AbstractVector{T}, k) where T
    # n = Threads.nthreads()
    n = 64
    chunk_size = length(data) ÷ n
    chunks = enumerate(Iterators.partition(data, chunk_size))
    tasks = map(chunks) do (i, chunk)
        Threads.@spawn topk_part(chunk, k, i * chunk_size)
    end
    chunk_indices = reduce(vcat, fetch.(tasks))
    # println("Max = ", maximum(chunk_indices), " ", length(data))
    partialsort(chunk_indices, 1:k, rev=true, by=x->data[x])

    # partialsortperm(data, 1:k, rev=true)
end
