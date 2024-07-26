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
    num_embeddings = metadata["num_embeddings"] + 512                   # 512 added for access with strides 
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

    # the embd2pid mapping
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

function retrieve(config::ColBERTConfig, Q::Array{Float64})
    # TODO: retrieve pids for embeddings contained in the cloest nprobe centroids 
end

function rank(ranker::IndexScorer, config::ColBERTConfig, Q::Array{Float64}, k::Int)
    # TODO: call retrieve to get pids for embeddings for the closest nprobe centroids
    # TODO: call score_pids to score those pids
end 
