"""
    IndexSaver(config::ColBERTConfig, codec::Union{Missing, ResidualCodec} = missing)

A structure to load/save various indexing components.

# Arguments

- `config`: A [`ColBERTConfig`](@ref). 
- `codec`: A codec to encode and decode the embeddings. 
"""
Base.@kwdef mutable struct IndexSaver
    config::ColBERTConfig
    codec::Union{Missing, ResidualCodec} = missing
end

"""
    load_codec!(saver::IndexSaver)

Load a codec from disk into `saver`. 

The path of of the codec is inferred from the config stored in `saver`.

# Arguments

- `saver`: An [`IndexSaver`](@ref) into which the codec is to be loaded.
"""
function load_codec!(saver::IndexSaver)
    index_path = saver.config.indexing_settings.index_path
    centroids = JLD2.load(joinpath(index_path, "centroids.jld2"), "centroids")
    avg_residual = JLD2.load(joinpath(index_path, "avg_residual.jld2"), "avg_residual")
    buckets = JLD2.load(joinpath(index_path, "buckets.jld2"))
    saver.codec = ResidualCodec(saver.config, centroids, avg_residual, buckets["bucket_cutoffs"], buckets["bucket_weights"]) 
end

"""
    save_codec(saver::IndexSaver)

Save the codec used by the `saver` to disk. 

This will create three files in the directory specified by the indexing path:
 - `centroids.jld2` containing the centroids.
 - `avg_residual.jld2` containing the average residual.
 - `buckets.jld2` containing the bucket cutoffs and weights.

Also see [`train`](@ref).

# Arguments

- `saver::IndexSaver`: The index saver to use.
"""
function save_codec(saver::IndexSaver)
    index_path = saver.config.indexing_settings.index_path
    centroids_path = joinpath(index_path, "centroids.jld2") 
    avg_residual_path = joinpath(index_path, "avg_residual.jld2") 
    buckets_path = joinpath(index_path, "buckets.jld2") 
    @info "Saving codec to $(centroids_path), $(avg_residual_path) and $(buckets_path)"

    JLD2.save(centroids_path, Dict("centroids" => saver.codec.centroids))
    JLD2.save(avg_residual_path, Dict("avg_residual" => saver.codec.avg_residual))
    JLD2.save(
        buckets_path, 
        Dict(
            "bucket_cutoffs" => saver.codec.bucket_cutoffs,
            "bucket_weights" => saver.codec.bucket_weights,
        )
    )
end

"""
    save_chunk(saver::IndexSaver, chunk_idx::Int, offset::Int, embs::Matrix{Float64}, doclens::Vector{Int})

Save a single chunk of compressed embeddings and their relevant metadata to disk.

The codes and compressed residuals for the chunk are saved in files named `<chunk_idx>.codec.jld2`. The document lengths are saved in a file named `doclens.<chunk_idx>.jld2`. Relevant metadata, including number of documents in the chunk, number of embeddings and the passage offsets are saved in a file named `<chunk_idx>.metadata.json`.

# Arguments

- `saver`: The [`IndexSaver`](@ref) containing relevant information to save the chunk.
- `chunk_idx`: The index of the current chunk being saved.
- `offset`: The offset in the original document collection where this chunk starts.
- `embs`: The embeddings matrix for the current chunk.
- `doclens`: The document lengths vector for the current chunk.
"""
function save_chunk(saver::IndexSaver, chunk_idx::Int, offset::Int, embs::Matrix{Float64}, doclens::Vector{Int})
    codes, residuals = compress(saver.codec, embs)
    path_prefix = joinpath(saver.config.indexing_settings.index_path, string(chunk_idx))
    @assert length(codes) == size(embs)[2]

    # saving the compressed embeddings
    codes_path = "$(path_prefix).codes.jld2"
    residuals_path = "$(path_prefix).residuals.jld2"
    @info "Saving compressed codes to $(codes_path) and residuals to $(residuals_path)"
    JLD2.save(codes_path, Dict("codes" => codes))
    JLD2.save(residuals_path, Dict("residuals" => residuals))

    # saving doclens
    doclens_path = joinpath(saver.config.indexing_settings.index_path, "doclens.$(chunk_idx).jld2")
    @info "Saving doclens to $(doclens_path)"
    JLD2.save(doclens_path, Dict("doclens" => doclens))

    # the metadata
    metadata_path = joinpath(saver.config.indexing_settings.index_path, "$(chunk_idx).metadata.json")
    @info "Saving metadata to $(metadata_path)"
    open(metadata_path, "w") do io
        JSON.print(io,
            Dict(
                 "passage_offset" => offset,
                 "num_passages" => length(doclens),
                 "num_embeddings" => length(codes),
            ),
            4                                                               # indent
        )
    end
end

"""
    check_chunk_exists(saver::IndexSaver, chunk_idx::Int)

Check if the index chunk exists for the given `chunk_idx`.

# Arguments

- `saver`: The `IndexSaver` object that contains the indexing settings.
- `chunk_idx`: The index of the chunk to check.

# Returns

A boolean indicating whether all relevant files for the chunk exist. 
"""
function check_chunk_exists(saver::IndexSaver, chunk_idx::Int)
    index_path = saver.config.indexing_settings.index_path
    path_prefix = joinpath(index_path, string(chunk_idx))
    codes_path = "$(path_prefix).codes.jld2"
    residuals_path = "$(path_prefix).residuals.jld2"
    doclens_path = joinpath(index_path, "doclens.$(chunk_idx).jld2")
    metadata_path = joinpath(index_path, "$(chunk_idx).metadata.json")

    for file in [codes_path, residuals_path, doclens_path, metadata_path]
        if !isfile(file)
            return false
        end
    end

    true
end
