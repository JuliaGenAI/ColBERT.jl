function batch(group::Vector, bsize::Int, provide_offset = false)
    vtype = provide_offset ? 
        Vector{Tuple{Int, typeof(group)}} :
        Vector{typeof(group)} 
    batches = vtype()
    offset = 0
    while offset <= length(group)
        if provide_offset
            push!(batches, (offset, group[offset:offset + bsize - 1]))
        else
            push!(batches, group[offset:offset + bsize - 1])
        end
        offset += bsize
    end
end
