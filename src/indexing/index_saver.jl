Base.@kwdef mutable struct IndexSaver
    config::ColBERTConfig
    codec::Union{Missing, ResidualCodec} = missing
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

# function save_chunk(saver::IndexSaver, chunk_idx::Int, offset::Int, embs::Matrix{Float64}, doclens::Vector{Int})
#     compressed_embs = compress
# end
