function batch(group::Vector, bsize::Int; provide_offset::Bool = false)
    vtype = provide_offset ? 
        Vector{Tuple{Int, typeof(group)}} :
        Vector{typeof(group)} 
    batches = vtype()
    offset = 1
    while offset <= length(group)
        if provide_offset
            push!(batches, (offset, group[offset:min(length(group), offset + bsize - 1)]))
        else
            push!(batches, group[offset:min(length(group), offset + bsize - 1)])
        end
        offset += bsize
    end
    batches
end
