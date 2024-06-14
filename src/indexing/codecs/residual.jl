using ..ColBERT: ColBERTConfig 
using ProtoStructs

@proto mutable struct ResidualCodec
    config::ColBERTConfig
    centroids::Matrix{Float64}
    avg_residual:: Float64
    bucket_cutoffs::Vector{Float64}
    bucket_weights::Vector{Float64}
end
