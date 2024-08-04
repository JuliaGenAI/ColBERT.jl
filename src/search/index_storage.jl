struct IndexScorer
    metadata::Dict
    codec::ResidualCodec
    ivf::Vector{Int}
    ivf_lengths::Vector{Int}
    doclens::Vector{Int}
    codes::Vector{UInt32}
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

    # loading all compressed embeddings
    num_embeddings = metadata["num_embeddings"]
    dim, nbits = config.doc_settings.dim, config.indexing_settings.nbits
    @assert (dim * nbits) % 8 == 0 "(dim, nbits): $((dim, nbits))"
    codes = zeros(UInt32, num_embeddings) 
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
    @assert isequal(sum(doclens), metadata["num_embeddings"]) "sum(doclens): $(sum(doclens)), num_embeddings: $(metadata["num_embeddings"])"
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

"""

Return a candidate set of `pids` for the query matrix `Q`. This is done as follows: the nearest `nprobe` centroids for each query embedding are found. This list is then flattened and the unique set of these centroids is built. Using the `ivf`, the list of all unique embedding IDs contained in these centroids is computed. Finally, these embedding IDs are converted to `pids` using `emb2pid`. This list of `pids` is the final candidate set.
"""
function retrieve(ranker::IndexScorer, config::ColBERTConfig, Q::AbstractArray{Float32})
    @assert isequal(size(Q)[2], config.query_settings.query_maxlen) "size(Q): $(size(Q)), query_maxlen: $(config.query_settings.query_maxlen)"     # Q: (128, 32, 1)

    Q = reshape(Q, size(Q)[1:end .!= end]...)           # squeeze out the last dimension 
    @assert isequal(length(size(Q)), 2) "size(Q): $(size(Q))"

    # score of each query embedding with each centroid and take top nprobe centroids
    cells = Flux.gpu(transpose(Q)) * Flux.gpu(ranker.codec.centroids) |> Flux.cpu
    # TODO: how to take topk entries using GPU code?
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
    @assert isequal(length(eids), sum(ranker.ivf_lengths[centroid_ids])) "length(eids): $(length(eids)), sum(ranker.ivf_lengths[centroid_ids]): $(sum(ranker.ivf_lengths[centroid_ids]))"
    eids = sort(unique(eids))

    # get pids from the emb2pid mapping
    pids = sort(unique(ranker.emb2pid[eids]))
    pids
end

"""
- Get the decompressed embedding matrix for all embeddings in `pids`. Use `doclens` for this.
"""
function score_pids(ranker::IndexScorer, config::ColBERTConfig, Q::AbstractArray{Float32}, pids::Vector{Int}) 
    # get codes and residuals for all embeddings across all pids
    num_embs = sum(ranker.doclens[pids]) 
    codes_packed = zeros(UInt32, num_embs)  
    residuals_packed = zeros(UInt8,  size(ranker.residuals)[1], num_embs)
    pid_offsets = cat([1], 1 .+ cumsum(ranker.doclens)[1:end .!= end], dims=1)

    offset = 1
    for pid in pids
        pid_offset = pid_offsets[pid]
        num_embs_pid = ranker.doclens[pid]
        codes_packed[offset: offset + num_embs_pid - 1] = ranker.codes[pid_offset: pid_offset + num_embs_pid - 1] 
        residuals_packed[:, offset: offset + num_embs_pid - 1] = ranker.residuals[:, pid_offset: pid_offset + num_embs_pid - 1] 
        offset += num_embs_pid
    end
    @assert offset == num_embs + 1 "offset: $(offset), num_embs + 1: $(num_embs + 1)"

    # decompress these codes and residuals to get the original embeddings
    D_packed = decompress(ranker.codec, codes_packed, residuals_packed) 
    @assert ndims(D_packed) == 2 "ndims(D_packed): $(ndims(D_packed))"
    @assert size(D_packed)[1] == config.doc_settings.dim "size(D_packed): $(size(D_packed)), config.doc_settings.dim: $(config.doc_settings.dim)"
    @assert size(D_packed)[2] == num_embs "size(D_packed): $(size(D_packed)), num_embs: $(num_embs)"
    @assert D_packed isa AbstractMatrix{Float32} "$(typeof(D_packed))"

    # get the max-sim scores
    if size(Q)[3] > 1
        error("Only one query is supported at the moment!")
    end
    @assert size(Q)[3] == 1 "size(Q): $(size(Q))"
    Q = reshape(Q, size(Q)[1:2]...)
    
    scores = Vector{Float32}()
    query_doc_scores = Flux.gpu(transpose(Q)) * Flux.gpu(D_packed)                    # (num_query_tokens, num_embeddings)
    offset = 1
    for pid in pids
        num_embs_pid = ranker.doclens[pid]
        pid_scores = query_doc_scores[:, offset:min(num_embs, offset + num_embs_pid - 1)]
        push!(scores, sum(maximum(pid_scores, dims = 2)))

        offset += num_embs_pid
    end
    @assert offset == num_embs + 1 "offset: $(offset), num_embs + 1: $(num_embs + 1)"

    scores
end

function rank(ranker::IndexScorer, config::ColBERTConfig, Q::AbstractArray{Float32})
    pids = retrieve(ranker, config, Q)
    scores = score_pids(ranker, config, Q, pids)
    indices = sortperm(scores, rev=true) 
    
    pids[indices], scores[indices]
end 
