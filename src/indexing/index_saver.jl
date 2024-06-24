using .ColBERT: ResidualCodec

Base.@kwdef mutable struct IndexSaver
    config::ColBERTConfig
    codec::Union{Missing, ResidualCodec} = missing
end

function load_codec!(saver::IndexSaver)
    index_path = saver.config.indexing_settings.index_path
    centroids = load(joinpath(index_path, "centroids.jld2"), "centroids")
    avg_residual = load(joinpath(index_path, "avg_residual.jld2"), "avg_residual")
    buckets = load(joinpath(index_path, "buckets.jld2"))
    saver.codec = ResidualCodec(saver.config, centroids, avg_residual, buckets["bucket_cutoffs"], buckets["bucket_weights"]) 
end

function save_codec(saver::IndexSaver)
    index_path = saver.config.indexing_settings.index_path
    centroids_path = joinpath(index_path, "centroids.jld2") 
    avg_residual_path = joinpath(index_path, "avg_residual.jld2") 
    buckets_path = joinpath(index_path, "buckets.jld2") 
    @info "Saving codec to $(centroids_path), $(avg_residual_path) and $(buckets_path)"

    save(centroids_path, Dict("centroids" => saver.codec.centroids))
    save(avg_residual_path, Dict("avg_residual" => saver.codec.avg_residual))
    save(
        buckets_path, 
        Dict(
            "bucket_cutoffs" => saver.codec.bucket_cutoffs,
            "bucket_weights" => saver.codec.bucket_weights,
        )
    )
end

function save_chunk(saver::IndexSaver, chunk_idx::Int, offset::Int, embs::Matrix{Float64}, doclens::Vector{Int})
    codes, residuals = compress(saver.codec, embs)
    path_prefix = joinpath(saver.config.indexing_settings.index_path, string(chunk_idx))
    
    # saving the compressed embeddings
    codes_path = "$(path_prefix).codes.jld2"
    residuals_path = "$(path_prefix).residuals.jld2"
    @info "Saving compressed codes to $(codes_path) and residuals to $(residuals_path)"
    save(codes_path, Dict("codes" => codes))
    save(residuals_path, Dict("residuals" => residuals))

    # saving doclens
    doclens_path = joinpath(saver.config.indexing_settings.index_path, "doclens.$(chunk_idx).jld2")
    @info "Saving doclens to $(doclens_path)"
    save(doclens_path, Dict("doclens" => doclens))

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
