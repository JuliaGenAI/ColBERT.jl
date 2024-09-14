"""
    save_codec(
        index_path::String, centroids::Matrix{Float32}, bucket_cutoffs::Vector{Float32},
        bucket_weights::Vector{Float32}, avg_residual::Float32)

Save compression/decompression information from the index path.

# Arguments

  - `index_path`: The path of the index.
  - `centroids`: The matrix of centroids of the index.
  - `bucket_cutoffs`: Cutoffs used to determine buckets during residual compression.
  - `bucket_weights`: Weights used to determine the decompressed values during decompression.
  - `avg_residual`: The average residual value, computed from the heldout set (see [`_compute_avg_residuals`](@ref)).
"""
function save_codec(
        index_path::String, centroids::Matrix{Float32}, bucket_cutoffs::Vector{Float32},
        bucket_weights::Vector{Float32}, avg_residual::Float32)
    centroids_path = joinpath(index_path, "centroids.jld2")
    avg_residual_path = joinpath(index_path, "avg_residual.jld2")
    bucket_cutoffs_path = joinpath(index_path, "bucket_cutoffs.jld2")
    bucket_weights_path = joinpath(index_path, "bucket_weights.jld2")
    @info "Saving codec to $(centroids_path), $(avg_residual_path), $(bucket_cutoffs_path) and $(bucket_weights_path)."

    JLD2.save_object(centroids_path, centroids)
    JLD2.save_object(avg_residual_path, avg_residual)
    JLD2.save_object(bucket_cutoffs_path, bucket_cutoffs)
    JLD2.save_object(bucket_weights_path, bucket_weights)
end

"""
    save_chunk(
        config::ColBERTConfig, codec::Dict, chunk_idx::Int, passage_offset::Int,
        embs::AbstractMatrix{Float32}, doclens::AbstractVector{Int})

Save a single chunk of compressed embeddings and their relevant metadata to disk.

The codes and compressed residuals for the chunk are saved in files named `<chunk_idx>.codes.jld2`.
and `<chunk_idx>.residuals.jld2` respectively. The document lengths are saved in a file named
`doclens.<chunk_idx>.jld2`. Relevant metadata, including number of documents in the chunk,
number of embeddings and the passage offsets are saved in a file named `<chunk_idx>.metadata.json`.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) being used.
  - `chunk_idx`: The index of the current chunk being saved.
  - `passage_offset`: The index of the first passage in the chunk.
  - `embs`: The embeddings matrix for the current chunk.
  - `doclens`: The document lengths vector for the current chunk.
"""
function save_chunk(
        index_path::String, codes::AbstractVector{UInt32}, residuals::AbstractMatrix{UInt8},
        chunk_idx::Int, passage_offset::Int, doclens::AbstractVector{Int})
    path_prefix = joinpath(index_path, string(chunk_idx))

    # saving the compressed embeddings
    codes_path = "$(path_prefix).codes.jld2"
    residuals_path = "$(path_prefix).residuals.jld2"
    @info "Saving compressed codes to $(codes_path) and residuals to $(residuals_path)"
    JLD2.save_object(codes_path, codes)
    JLD2.save_object(residuals_path, residuals)

    # saving doclens
    doclens_path = joinpath(
        index_path, "doclens.$(chunk_idx).jld2")
    @info "Saving doclens to $(doclens_path)"
    JLD2.save_object(doclens_path, doclens)

    # the metadata
    metadata_path = joinpath(
        index_path, "$(chunk_idx).metadata.json")
    @info "Saving metadata to $(metadata_path)"
    open(metadata_path, "w") do io
        JSON.print(io,
            Dict(
                "passage_offset" => passage_offset,
                "num_passages" => length(doclens),
                "num_embeddings" => length(codes)
            ),
            4
        )
    end
end

"""
    save(config::ColBERTConfig)

Save a [`ColBERTConfig`](@ref) to disk in JSON.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) to save.

# Examples

```jldoctest
julia> using ColBERT;

julia> config = ColBERTConfig(
           use_gpu = true,
           collection = "/home/codetalker7/documents",
           index_path = "./local_index"
       );

julia> ColBERT.save(config);

```
"""
function save(config::ColBERTConfig)
    properties = [Pair{String, Any}(string(field), getproperty(config, field))
                  for field in fieldnames(ColBERTConfig)]
    isdir(config.index_path) || mkdir(config.index_path)
    open(joinpath(config.index_path, "config.json"), "w+") do io
        JSON.print(
            io,
            Dict(properties),
            4
        )
    end
end

function save_chunk_metadata_property(
        index_path::String, property::String, properties::Vector{T}) where {T}
    plan_metadata = JSON.parsefile(joinpath(index_path, "plan.json"))
    @assert plan_metadata["num_chunks"] == length(properties)
    for chunk_idx in 1:length(properties)
        chunk_metadata = JSON.parsefile(joinpath(
            index_path, "$(chunk_idx).metadata.json"))
        chunk_metadata[property] = properties[chunk_idx]
        open(joinpath(index_path, "$(chunk_idx).metadata.json"), "w") do io
            JSON.print(io,
                chunk_metadata,
                4
            )
        end
    end
end
