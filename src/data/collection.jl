# for now, we load collections in memory.
# will be good to implement on-disk data structures too.
struct Collection
    path::String
    data::Vector{String}
end

function Collection(path::String)
    file = CSV.File(path; delim='\t', header = [:pid, :text], types = Dict(:pid => Int, :text => String), debug=true, quoted=false)
    @info "Loaded $(length(file.text)[1]) passages."
    Collection(path, file.text)
end

function get_chunksize(collection::Collection, nranks::Int)
    min(25000, 1 + floor(length(collection.data) / nranks))
end

function enumerate_batches(collection::Collection, chunksize::Union{Int, Missing} = missing, nranks::Union{Int, Missing} = missing)
    if ismissing(chunksize)
        if ismissing(nranks)
            error("Atleast one of the arguments chunksize or nranks must be specified!")
        end
        chunksize = get_chunksize(collection, nranks)
    end

    num_passages = length(collection.data)
    batches = Vector{Tuple{Int, Int, Vector{String}}}()
    chunk_idx, offset = 1, 1
    while true
        push!(batches, (chunk_idx, offset, collection.data[offset:min(offset + chunksize - 1, num_passages)]))
        chunk_idx += 1
        offset += chunksize

        if offset > num_passages
            break
        end
    end
    batches
end

