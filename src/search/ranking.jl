"""
    _cids_to_eids!(eids::Vector{Int}, centroid_ids::Vector{Int},
        ivf::Vector{Int}, ivf_lengths::Vector{Int})

Get the set of embedding IDs contained in `centroid_ids`.
"""
function _cids_to_eids!(eids::Vector{Int}, centroid_ids::Vector{Int},
        ivf::Vector{Int}, ivf_lengths::Vector{Int})
    @assert length(eids) == sum(ivf_lengths[centroid_ids])
    centroid_ivf_offsets = cumsum([1; _head(ivf_lengths)])
    eid_offsets = cumsum([1; _head(ivf_lengths[centroid_ids])])
    for (idx, centroid_id) in enumerate(centroid_ids)
        eid_offset = eid_offsets[idx]
        batch_length = ivf_lengths[centroid_id]
        ivf_offset = centroid_ivf_offsets[centroid_id]
        eids[eid_offset:(eid_offset + batch_length - 1)] .= ivf[ivf_offset:(ivf_offset + batch_length - 1)]
    end
end

function retrieve(
        ivf::Vector{Int}, ivf_lengths::Vector{Int}, centroids::AbstractMatrix{Float32},
        emb2pid::Vector{Int}, nprobe::Int, Q::AbstractMatrix{Float32})
    # score each query against each centroid
    cells = Q' * centroids                                          # (num_query_embeddings, num_centroids)

    # TODO: how to take topk entries using GPU code?
    cells = cells |> Flux.cpu
    cells = _topk(cells, nprobe, dims = 2)                          # (num_query_embeddings, nprobe)
    centroid_ids = sort(unique(vec(cells)))

    # get all embedding IDs contained in centroid_ids using ivf
    eids = Vector{Int}(undef, sum(ivf_lengths[centroid_ids]))       # (sum(ivf_lengths[centroid_ids]), 1)
    _cids_to_eids!(eids, centroid_ids, ivf, ivf_lengths)

    # get unique eids
    eids = sort(unique(eids))

    # get pids from the emb2pid mapping
    pids = sort(unique(emb2pid[eids]))
    pids
end

function _collect_compressed_embs_for_pids(
        doclens::Vector{Int}, codes::Vector{UInt32},
        residuals::Matrix{UInt8}, pids::Vector{Int})
    # get offsets of pids in codes and residuals and the resultant arrays
    pid_offsets = cumsum([1; _head(doclens)])
    offsets = cumsum([1; _head(doclens[pids])])

    # collecting the codes and residuals for pids
    num_embeddings = sum(doclens[pids])
    codes_packed = zeros(UInt32, num_embeddings)
    residuals_packed = zeros(UInt8, size(residuals, 1), num_embeddings)
    for (idx, pid) in enumerate(pids)
        offset = offsets[idx]
        pid_offset = pid_offsets[pid]
        num_embs_pid = doclens[pid]
        codes_packed[offset:(offset + num_embs_pid - 1)] .= codes[
            pid_offset:(pid_offset + num_embs_pid - 1)]
        residuals_packed[:, offset:(offset + num_embs_pid - 1)] .= residuals[
            :, pid_offset:(pid_offset + num_embs_pid - 1)]
    end
    codes_packed, residuals_packed
end

function maxsim(Q::AbstractMatrix{Float32}, D::AbstractMatrix{Float32},
        pids::Vector{Int}, doclens::Vector{Int})
    scores = zeros(Float32, length(pids))
    num_embeddings = sum(doclens[pids])
    query_doc_scores = Q' * D
    offsets = cumsum([1; _head(doclens[pids])])
    for (idx, pid) in enumerate(pids)
        num_embs_pids = doclens[pid]
        offset = offsets[idx]
        offset_end = min(num_embeddings, offset + num_embs_pids - 1)
        pid_scores = query_doc_scores[:, offset:offset_end]
        scores[idx] = sum(maximum(pid_scores, dims = 2))
        offset += num_embs_pids
    end
    scores
end
