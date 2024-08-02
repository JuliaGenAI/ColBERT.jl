"""
    ResidualCodec(config::ColBERTConfig, centroids::AbstractMatrix{Float32}, avg_residual::Float32, bucket_cutoffs::AbstractVector{Float32}, bucket_weights::AbstractVector{Float32})

A struct that represents a compressor for ColBERT embeddings. 

It stores information about the configuration of the model, the centroids used to quantize the residuals, the average residual value, and the cutoffs and weights used to determine which buckets each residual belongs to.

# Arguments

- `config`: A [`ColBERTConfig`](@ref), representing all configuration parameters related to various ColBERT components.
- `centroids`: A matrix of centroids used to quantize the residuals. Has shape `(D, N)`, where `D` is the embedding dimension and `N` is the number of clusters.
- `avg_residual`: The average residual value.
- `bucket_cutoffs`: A vector of cutoff values used to determine which buckets each residual belongs to.
- `bucket_weights`: A vector of weights used to determine the importance of each bucket.

# Returns

A `ResidualCodec` object.
"""
mutable struct ResidualCodec
    config::ColBERTConfig
    centroids::AbstractMatrix{Float32}
    avg_residual::Float32
    bucket_cutoffs::AbstractVector{Float32}
    bucket_weights::AbstractVector{Float32}
end

"""

# Examples

```julia-repl
julia> codec = load_codec(index_path); 
```
"""
function load_codec(index_path::String)
    config = load(joinpath(index_path, "config.jld2"), "config")
    centroids = load(joinpath(index_path, "centroids.jld2"), "centroids")
    avg_residual = load(joinpath(index_path, "avg_residual.jld2"), "avg_residual")
    buckets = load(joinpath(index_path, "buckets.jld2"))
    ResidualCodec(config, centroids, avg_residual, buckets["bucket_cutoffs"], buckets["bucket_weights"])
end

"""
    compress_into_codes(codec::ResidualCodec, embs::AbstractMatrix{Float32})

Compresses a matrix of embeddings into a vector of codes using the given [`ResidualCodec`](@ref), where the code for each embedding is its nearest centroid ID. 

# Arguments

- `codec`: The [`ResidualCodec`](@ref) used to compress the embeddings.
- `embs`: The matrix of embeddings to be compressed.

# Returns

A `Vector{UInt32}` of codes, where each code corresponds to the nearest centroid ID for the embedding.
```
"""
function compress_into_codes(codec::ResidualCodec, embs::AbstractMatrix{Float32})
    use_gpu = codec.config.run_settings.use_gpu
    codes = Vector{UInt32}() 

    if use_gpu
        codes = codes |> Flux.gpu
    end

    bsize = Int(floor((1 << 29) / size(codec.centroids)[2]))
    offset = 1 
    while (offset <= size(embs)[2])                             # batch on the second dimension
        if !use_gpu
            dot_products = transpose(embs[:, offset:min(size(embs)[2], offset + bsize - 1)]) * codec.centroids 
        else
            dot_products = transpose(Flux.gpu(embs[:, offset:min(size(embs)[2], offset + bsize - 1)])) * Flux.gpu(codec.centroids)
        end
        indices = (cartesian_index -> cartesian_index.I[2]).(argmax(dot_products, dims = 2)[:, 1])
        append!(codes, indices)
        offset += bsize
    end

    @assert length(codes) == size(embs)[2] "length(codes): $(length(codes)), size(embs): $(size(embs))"
    @assert codes isa AbstractVector{UInt32} "$(typeof(codes))"
    codes
end

"""
    binarize(codec::ResidualCodec, residuals::AbstractMatrix{Float32})

Convert a matrix of residual vectors into a matrix of integer residual vector using `nbits` bits (specified by the underlying `config`). 

# Arguments

- `codec`: A [`ResidualCodec`](@ref) object containing the compression information. 
- `residuals`: The matrix of residuals to be converted.

# Returns

A matrix of compressed integer residual vectors. 
"""
function binarize(codec::ResidualCodec, residuals::AbstractMatrix{Float32})
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
    @assert ndims(residuals_packed) == 2 "ndims(residuals_packed): $(ndims(residuals_packed))"
    @assert size(residuals_packed)[2] == size(residuals)[2] "size(residuals_packed): $(size(residuals_packed)), size(residuals): $(size(residuals))"
    @assert residuals_packed isa AbstractMatrix{UInt8} "$(typeof(residuals_packed))"

    residuals_packed
end

"""
    compress(codec::ResidualCodec, embs::AbstractMatrix{Float32})

Compress a matrix of embeddings into a compact representation using the specified [`ResidualCodec`](@ref). 

All embeddings are compressed to their nearest centroid IDs and their quantized residual vectors (where the quantization is done in `nbits` bits, specified by the `config` of  `codec`). If `emb` denotes an embedding and `centroid` is is nearest centroid, the residual vector is defined to be `emb - centroid`.

# Arguments

- `codec`: A [`ResidualCodec`](@ref) object containing the centroids and other parameters for the compression algorithm.
- `embs`: The input embeddings to be compressed.

# Returns

A tuple containing a vector of codes and the compressed residuals matrix.
"""
function compress(codec::ResidualCodec, embs::AbstractMatrix{Float32})
    codes, residuals = Vector{UInt32}(), Vector{Matrix{UInt8}}() 

    offset = 1
    bsize = 1 << 18
    while (offset <= size(embs)[2])                # batch on second dimension
        batch = embs[:, offset:min(size(embs)[2], offset + bsize - 1)]
        codes_ = compress_into_codes(codec, batch) # get centroid codes
        centroids_ = codec.centroids[:, codes_]    # get corresponding centroids
        residuals_ = batch - centroids_ 
        append!(codes, codes_) 
        push!(residuals, binarize(codec, residuals_))
        offset += bsize
    end
    residuals = cat(residuals..., dims = 2)

    @assert ndims(codes) == 1 "ndims(codes): $(ndims(codes))"
    @assert ndims(residuals) == 2 "ndims(residuals): $(ndims(residuals))"
    @assert length(codes) == size(embs)[2] "length(codes): $(length(codes)), size(embs): $(size(embs))" 
    @assert size(residuals)[2] == size(embs)[2] "size(residuals): $(size(residuals)), size(embs): $(size(embs))"
    @assert codes isa AbstractVector{UInt32} "$(typeof(codes))"
    @assert residuals isa AbstractMatrix{UInt8} "$(typeof(residuals))"

    codes, residuals
end

function decompress_residuals(codec::ResidualCodec, binary_residuals::AbstractMatrix{UInt8})
    dim = codec.config.doc_settings.dim
    nbits = codec.config.indexing_settings.nbits

    @assert ndims(binary_residuals) == 2 "ndims(binary_residuals): $(ndims(binary_residuals))" 
    @assert size(binary_residuals)[1] == (dim / 8) * nbits "size(binary_residuals): $(size(binary_residuals)), (dim / 8) * nbits: $((dim / 8) * nbits)"

    # unpacking UInt8 into bits
    unpacked_bits = BitVector() 
    for byte in vec(binary_residuals) 
        append!(unpacked_bits, [byte & (0x1<<n) != 0 for n in 0:7])
    end
    
    # reshaping into dims (nbits, dim, num_embeddings); inverse of what binarize does
    unpacked_bits = reshape(unpacked_bits, nbits, dim, size(binary_residuals)[2])

    # get decimal value for coordinate of the nbits-wide dimension; again, inverse of binarize
    positionbits = fill(1, (nbits, 1, 1))
    for i in 1:nbits
        positionbits[i, :, :] .= 1 << (i - 1)
    end

    # multiply by 2^(i - 1) for the ith bit, and take sum to get the original bin back
    unpacked_bits = unpacked_bits .* positionbits
    unpacked_bits = sum(unpacked_bits, dims=1)
    unpacked_bits = unpacked_bits .+ 1                          # adding 1 to get correct bin indices

    # reshaping to get rid of the nbits wide dimension
    unpacked_bits = reshape(unpacked_bits, size(unpacked_bits)[2:end]...)
    embeddings = codec.bucket_weights[unpacked_bits]

    @assert ndims(embeddings) == 2 "ndims(embeddings): $(ndims(embeddings))"
    @assert size(embeddings)[2] == size(binary_residuals)[2] "size(embeddings): $(size(embeddings)), size(binary_residuals): $(size(binary_residuals)) "
    @assert embeddings isa AbstractMatrix{Float32} "$(typeof(embeddings))" 

    embeddings
end

function decompress(codec::ResidualCodec, codes::Vector{UInt32}, residuals::AbstractMatrix{UInt8})
    @assert ndims(codes) == 1 "ndims(codes): $(ndims(codes))"
    @assert ndims(residuals) == 2 "ndims(residuals): $(ndims(residuals))"
    @assert length(codes) == size(residuals)[2] "length(codes): $(length(codes)), size(residuals): $(size(residuals))"

    # decompress in batches
    D = Vector{AbstractMatrix{Float32}}() 
    bsize = 1 << 15
    batch_offset = 1
    while batch_offset <= length(codes)
        batch_codes = codes[batch_offset:min(batch_offset + bsize - 1, length(codes))]
        batch_residuals = residuals[:, batch_offset:min(batch_offset + bsize - 1, length(codes))]

        centroids_ = codec.centroids[:, batch_codes]
        residuals_ = decompress_residuals(codec, batch_residuals)
    
        batch_embeddings = centroids_ + residuals_
        batch_embeddings = mapslices(v -> iszero(v) ? v : normalize(v), batch_embeddings, dims = 1)
        push!(D, batch_embeddings)

        batch_offset += bsize
    end
    embeddings = cat(D..., dims = 2)

    @assert ndims(embeddings) == 2 "ndims(embeddings): $(ndims(embeddings))"
    @assert size(embeddings)[2] == length(codes) "size(embeddings): $(size(embeddings)),  length(codes): $(length(codes))"  
    @assert embeddings isa AbstractMatrix{Float32} "$(typeof(embeddings))"

    embeddings
end

"""
    load_codes(codec::ResidualCodec, chunk_idx::Int)

Load the codes from disk for a given chunk index. The codes are stored in the file `<chunk_idx>.codes.jld2` located inside the 
`index_path` provided by the configuration.

# Arguments

- `codec`: The [`ResidualCodec`](@ref) object containing the compression information.
- `chunk_idx`: The chunk index for which the codes should be loaded.

# Returns

A vector of codes for the specified chunk.
"""
function load_codes(codec::ResidualCodec, chunk_idx::Int)
    codes_path = joinpath(codec.config.indexing_settings.index_path, "$(chunk_idx).codes.jld2")
    codes = JLD2.load(codes_path, "codes")
    codes
end

function load_residuals(codec::ResidualCodec, chunk_idx::Int)
    residual_path = joinpath(codec.config.indexing_settings.index_path, "$(chunk_idx).residuals.jld2")
    residuals = JLD2.load(residual_path, "residuals")
    residuals 
end
