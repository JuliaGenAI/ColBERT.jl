struct IndexScorer
    metadata::Dict
    codec::ResidualCodec
    ivf::Vector{Int}
    ivf_lengths::Vector{Int}
    doclens::Vector{Int}
    codes::Vector{Int}
    residuals::Matrix{UInt8}
    emb2pid::Vector{Int}
end

"""

# Examples

```julia-repl
julia> IndexScorer(index_path) 

```
"""
function IndexScorer(index_path::String)
    @info "Loading the index from {index_path}."

    # loading the config from the index path
    config = JLD2.load(joinpath(index_path, "config.jld2"))["config"]

    # the metadata
    metadata_path = joinpath(index_path, "metadata.json")
    metadata = JSON.parsefile(metadata_path) 

    # loading the codec
    codec = load_codec(index_path)

    # loading ivf into a StridedTensor
    ivf_path = joinpath(index_path, "ivf.jld2")
    ivf_dict = JLD2.load(ivf_path)
    ivf, ivf_lengths = ivf_dict["ivf"], ivf_dict["ivf_lengths"]
    # ivf = StridedTensor(ivf, ivf_lengths)

    # loading all doclens
    doclens = Vector{Int}() 
    for chunk_idx in 1:metadata["num_chunks"]
        doclens_file = joinpath(index_path, "doclens.$(chunk_idx).jld2") 
        chunk_doclens = JLD2.load(doclens_file, "doclens")
        append!(doclens, chunk_doclens)
    end

    # loading all embeddings
    num_embeddings = metadata["num_embeddings"]
    dim, nbits = config.doc_settings.dim, config.indexing_settings.nbits
    @assert (dim * nbits) % 8 == 0
    codes = zeros(Int, num_embeddings) 
    residuals = zeros(UInt8, Int((dim  / 8) * nbits), num_embeddings)
    codes_offset = 1
    for chunk_idx in 1:metadata["num_chunks"]
        chunk_codes = load_codes(codec, chunk_idx) 
        chunk_residuals = load_residuals(codec, chunk_idx) 
        
        codes_endpos = codes_offset + length(chunk_codes) - 1
        codes[codes_offset:codes_endpos] = chunk_codes
        residuals[:, codes_offset:codes_endpos] = chunk_residuals

        codes_offset = codes_offset + length(chunk_codes)
    end

    # the emb2pid mapping
    @info "Building the emb2pid mapping."
    @assert isequal(sum(doclens), metadata["num_embeddings"])
    emb2pid = zeros(Int, metadata["num_embeddings"])

    offset_doclens = 1
    for (pid, dlength) in enumerate(doclens)
            emb2pid[offset_doclens:offset_doclens + dlength - 1] .= pid
            offset_doclens += dlength
    end

    IndexScorer(
        metadata,
        codec,
        ivf,
        ivf_lengths,
        doclens,
        codes,
        residuals,
        emb2pid,
    )
end

function retrieve(ranker::IndexScorer, config::ColBERTConfig, Q::Array{<:AbstractFloat})
    @assert isequal(size(Q)[2], config.query_settings.query_maxlen)     # Q: (128, 32, 1)

    Q = reshape(Q, size(Q)[1:end .!= end]...)           # squeeze out the last dimension 
    @assert isequal(length(size(Q)), 2)

    # score of each query embedding with each centroid and take top nprobe centroids
    cells = transpose(Q) * ranker.codec.centroids
    cells = mapslices(row -> partialsortperm(row, 1:config.search_settings.nprobe, rev=true), cells, dims = 2)          # take top nprobe centroids for each query 
    centroid_ids = sort(unique(vec(cells)))

    # get all embedding IDs contained in centroid_ids using ivf
    centroid_ivf_offsets = cat([1], 1 .+ cumsum(ranker.ivf_lengths)[1:end .!= end], dims = 1)
    eids = Vector{Int}()
    for centroid_id in centroid_ids
        offset = centroid_ivf_offsets[centroid_id]
        length = ranker.ivf_lengths[centroid_id]
        append!(eids, ranker.ivf[offset:offset + length - 1])
    end
    @assert isequal(length(eids), sum(ranker.ivf_lengths[centroid_ids]))
    eids = sort(unique(eids))

    # get pids from the emb2pid mapping
    pids = sort(unique(ranker.emb2pid[eids]))
    pids
end

function rank(ranker::IndexScorer, config::ColBERTConfig, Q::Array{Float64}, k::Int)
    # TODO: call retrieve to get pids for embeddings for the closest nprobe centroids
    pids = retrieve(config, Q)
    # TODO: call score_pids to score those pids
end 
