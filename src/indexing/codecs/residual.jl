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

function binarize(codec::ResidualCodec, residuals::Matrix{Float64})
    dim = codec.config.doc_settings.dim
    nbits = codec.config.indexing_settings.nbits
    num_embeddings = size(residuals)[2]

    if dim % (nbits * 8) != 0
        error("The embeddings dimension must be a multiple of nbits * 8!")
    end

    # need to subtract one here, to preserve the number of options (2 ^ nbits) 
    bucket_indices = (x -> searchsortedfirst(codec.bucket_cutoffs, x)).(residuals) .- 1  # torch.bucketize
    bucket_indices = stack([bucket_indices for i in 1:nbits], dims = 1)                  # add an nbits-wide extra dimension
    positionbits = fill(1, (nbits, 1, 1))
    for i in 1:nbits
        positionbits[i, :, :] .= 1 << (i - 1)
    end

    bucket_indices = Int.(floor.(bucket_indices ./ positionbits))                        # divide by 2^bit for each bit position
    bucket_indices = bucket_indices .& 1                                                 # apply mod 1 to binarize
    residuals_packed = reinterpret(UInt8, BitArray(vec(bucket_indices)).chunks)          # flatten out the bits, and pack them into UInt8
    residuals_packed = reshape(residuals_packed, (Int(dim / 8) * nbits, num_embeddings)) # reshape back to get compressions for each embedding
end

function compress(codec::ResidualCodec, embs::Matrix{Float64})
    codes, residuals = Vector{Int}(), Vector{Matrix{UInt8}}() 

    offset = 1
    bsize = 1 << 18
    while (offset <= size(embs[2]))                # batch on second dimension
        batch = embs[:, offset:min(size(embs)[2], offset + bsize - 1)]
        codes_ = compress_into_codes(codec, batch) # get centroid codes
        centroids_ = codec.centroids[:, codes_]    # get corresponding centroids
        residuals_ = batch - centroids_ 
        append!(codes, codes_) 
        push!(residuals, binarize(codec, residuals_))
    end
    residuals = cat(residuals..., dims = 2)

    codes, residuals
end
