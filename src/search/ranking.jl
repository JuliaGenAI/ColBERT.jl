"""
Return a candidate set of `pids` for the query matrix `Q`. This is done as follows: the nearest `nprobe` centroids for each query embedding are found. This list is then flattened and the unique set of these centroids is built. Using the `ivf`, the list of all unique embedding IDs contained in these centroids is computed. Finally, these embedding IDs are converted to `pids` using `emb2pid`. This list of `pids` is the final candidate set.
"""
function retrieve(ivf::Vector{Int}, ivf_lengths::Vector{Int}, centroids::Matrix{Float32},
        emb2pid::Vector{Int}, nprobe::Int, Q::AbstractMatrix{Float32})
    # score of each query embedding with each centroid and take top nprobe centroids
    cells = Flux.gpu(transpose(Q)) * Flux.gpu(centroids) |> Flux.cpu
    # TODO: how to take topk entries using GPU code?
    cells = mapslices(
        row -> partialsortperm(row, 1:(nprobe), rev = true),
        cells, dims = 2)          # take top nprobe centroids for each query 
    centroid_ids = sort(unique(vec(cells)))

    # get all embedding IDs contained in centroid_ids using ivf
    centroid_ivf_offsets = cat(
        [1], 1 .+ cumsum(ivf_lengths)[1:end .!= end], dims = 1)
    eids = Vector{Int}()
    for centroid_id in centroid_ids
        offset = centroid_ivf_offsets[centroid_id]
        length = ivf_lengths[centroid_id]
        append!(eids, ivf[offset:(offset + length - 1)])
    end
    @assert isequal(length(eids), sum(ivf_lengths[centroid_ids]))
    "length(eids): $(length(eids)), sum(ranker.ivf_lengths[centroid_ids]):" *
    "$(sum(ivf_lengths[centroid_ids]))"
    eids = sort(unique(eids))

    # get pids from the emb2pid mapping
    pids = sort(unique(emb2pid[eids]))
    pids
end

function _collect_compressed_embs_for_pids(doclens::Vector{Int}, codes::Vector{UInt32},
        residuals::Matrix{UInt8}, pids::Vector{Int})
    num_embeddings = sum(doclens[pids])
    codes_packed = zeros(UInt32, num_embeddings)
    residuals_packed = zeros(UInt8, size(residuals)[1], num_embeddings)
    pid_offsets = cat([1], 1 .+ cumsum(doclens)[1:end .!= end], dims = 1)
    offset = 1
    for pid in pids
        pid_offset = pid_offsets[pid]
        num_embs_pid = doclens[pid]
        codes_packed[offset:(offset + num_embs_pid - 1)] = codes[pid_offset:(pid_offset + num_embs_pid - 1)]
        residuals_packed[:, offset:(offset + num_embs_pid - 1)] = residuals[
            :, pid_offset:(pid_offset + num_embs_pid - 1)]
        offset += num_embs_pid
    end
    @assert offset==num_embeddings + 1 "offset: $(offset), num_embs + 1: $(num_embeddings + 1)"
    codes_packed, residuals_packed
end

function maxsim(
        Q::Matrix{Float32}, D::Matrix{Float32}, pids::Vector{Int}, doclens::Vector{Int})
    scores = zeros(Float32, length(pids))
    num_embeddings = sum(doclens[pids])
    query_doc_scores = Flux.gpu(transpose(Q)) * Flux.gpu(D)                    # (num_query_tokens, num_embeddings)
    offset = 1
    for (idx, pid) in enumerate(pids)
        num_embs_pids = doclens[pid]
        offset_end = min(num_embeddings, offset + num_embs_pids - 1)
        pid_scores = query_doc_scores[:, offset:offset_end]
        scores[idx] = sum(maximum(pid_scores, dims = 2))
        offset += num_embs_pids 
    end
    @assert offset == num_embeddings + 1 "offset: $(offset), num_embs + 1: $(num_embeddings + 1)"
    scores
end

"""
  - Get the decompressed embedding matrix for all embeddings in `pids`. Use `doclens` for this.
"""
function score_pids(config::ColBERTConfig, centroids::Matrix{Float32},
        bucket_weights::Vector{Float32}, doclens::Vector{Int}, codes::Vector{UInt32},
        residuals::Matrix{UInt8}, Q::Matrix{Float32}, pids::Vector{Int})
    codes_packed, residuals_packed = _collect_compressed_embs_for_pids(
        doclens, codes, residuals, pids)
    D_packed = decompress(
        config.dim, config.nbits, centroids, bucket_weights, codes_packed, residuals_packed)
    @assert ndims(D_packed)==2 "ndims(D_packed): $(ndims(D_packed))"
    @assert size(D_packed)[1] == config.dim
        "size(D_packed): $(size(D_packed)), config.dim: $(config.dim)"
    @assert size(D_packed)[2] == sum(doclens[pids])
        "size(D_packed): $(size(D_packed)), num_embs: $(sum(doclens[pids]))"
    @assert D_packed isa AbstractMatrix{Float32} "$(typeof(D_packed))"
    maxsim(Q, D_packed, pids, doclens)
end
