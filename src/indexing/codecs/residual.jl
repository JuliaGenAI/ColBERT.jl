using .ColBERT: ColBERTConfig 

mutable struct ResidualCodec
    config::ColBERTConfig
    centroids::Matrix{Float64}
    avg_residual::Float64
    bucket_cutoffs::Vector{Float64}
    bucket_weights::Vector{Float64}
end

function compress_into_codes(codec::ResidualCodec, embs::Matrix{Float64})
    codes = []

    bsize = Int(floor((1 << 29) / size(codec.centroids)[2]))
    offset = 1 
    while (offset <= size(embs)[2])                             # batch on the second dimension
        dot_products = transpose(embs[:, offset:min(size(embs)[2], offset + bsize - 1)]) * codec.centroids 
        indices = (cartesian_index -> cartesian_index.I[2]).(argmax(dot_products, dims = 2)[:, 1])
        append!(codes, indices)
        offset += bsize
    end

    codes
end

function save_codec(codec::ResidualCodec, index_path::String)
    centroids_path = joinpath(index_path, "centroids.jld2") 
    avg_residual_path = joinpath(index_path, "avg_residual.jld2") 
    buckets_path = joinpath(index_path, "buckets.jld2") 
    @info "Saving codec to $(centroids_path), $(avg_residual_path) and $(buckets_path)"

    save(centroids_path, Dict("centroids" => codec.centroids))
    save(avg_residual_path, Dict("avg_residual" => codec.avg_residual))
    save(
        buckets_path, 
        Dict(
            "bucket_cutoffs" => codec.bucket_cutoffs,
            "bucket_weights" => codec.bucket_weights,
        )
    )
end
