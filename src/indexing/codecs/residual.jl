"""
    load_codec(index_path::String)

Load compression/decompression information from the index path.

# Arguments

  - `index_path`: The path of the index.
"""
function load_codec(index_path::String)
    centroids_path = joinpath(index_path, "centroids.jld2")
    avg_residual_path = joinpath(index_path, "avg_residual.jld2")
    bucket_cutoffs_path = joinpath(index_path, "bucket_cutoffs.jld2")
    bucket_weights_path = joinpath(index_path, "bucket_weights.jld2")
    @info "Loading codec from $(centroids_path), $(avg_residual_path), $(bucket_cutoffs_path) and $(bucket_weights_path)."

    centroids = JLD2.load_object(centroids_path)
    avg_residual = JLD2.load_object(avg_residual_path)
    bucket_cutoffs = JLD2.load_object(bucket_cutoffs_path)
    bucket_weights = JLD2.load_object(bucket_weights_path)

    @assert centroids isa Matrix{Float32}
    @assert avg_residual isa Float32
    @assert bucket_cutoffs isa Vector{Float32}
    @assert bucket_weights isa Vector{Float32}

    Dict(
        "centroids" => centroids,
        "avg_residual" => avg_residual,
        "bucket_cutoffs" => bucket_cutoffs,
        "bucket_weights" => bucket_weights
    )
end

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
    compress_into_codes(
        centroids::AbstractMatrix{Float32}, embs::AbstractMatrix{Float32})

Compresses a matrix of embeddings into a vector of codes using the given `centroids`,
where the code for each embedding is its nearest centroid ID.

# Arguments

  - `centroids`: The matrix of centroids.
  - `embs`: The matrix of embeddings to be compressed.

# Returns

A `Vector{UInt32}` of codes, where each code corresponds to the nearest centroid ID for the embedding.
"""
function compress_into_codes(
        centroids::AbstractMatrix{Float32}, embs::AbstractMatrix{Float32})
    codes = Vector{UInt32}()

    bsize = Int(floor((1 << 29) / size(centroids)[2]))
    offset = 1
    while (offset <= size(embs)[2])                             # batch on the second dimension
        dot_products = transpose(Flux.gpu(embs[
            :, offset:min(size(embs)[2], offset + bsize - 1)])) * Flux.gpu(centroids)
        indices = (cartesian_index -> cartesian_index.I[2]).(argmax(dot_products, dims = 2)[
            :, 1])
        append!(codes, indices)
        offset += bsize
    end
    @assert length(codes)==size(embs)[2] "length(codes): $(length(codes)), size(embs): $(size(embs))"
    @assert codes isa AbstractVector{UInt32} "$(typeof(codes))"

    codes
end

"""
    binarize(dim::Int, nbits::Int, bucket_cutoffs::Vector{Float32},
        residuals::AbstractMatrix{Float32})

Convert a matrix of residual vectors into a matrix of integer residual vector
using `nbits` bits.

# Arguments

  - `dim`: The embedding dimension (see [`ColBERTConfig`](@ref)).
  - `nbits`: Number of bits to compress the residuals into.
  - `bucket_cutoffs`: Cutoffs used to determine residual buckets.
  - `residuals`: The matrix of residuals ot be compressed.

# Returns

A `AbstractMatrix{UInt8}` of compressed integer residual vectors.
"""
function binarize(dim::Int, nbits::Int, bucket_cutoffs::Vector{Float32},
        residuals::AbstractMatrix{Float32})
    num_embeddings = size(residuals)[2]

    if dim % (nbits * 8) != 0
        error("The embeddings dimension must be a multiple of nbits * 8!")
    end

    # need to subtract one here, to preserve the number of options (2 ^ nbits) 
    bucket_indices = (x -> searchsortedfirst(bucket_cutoffs, x)).(residuals) .- 1  # torch.bucketize
    bucket_indices = stack([bucket_indices for i in 1:nbits], dims = 1)                  # add an nbits-wide extra dimension
    positionbits = fill(1, (nbits, 1, 1))
    for i in 1:nbits
        positionbits[i, :, :] .= 1 << (i - 1)
    end

    bucket_indices = Int.(floor.(bucket_indices ./ positionbits))                        # divide by 2^bit for each bit position
    bucket_indices = bucket_indices .& 1                                                 # apply mod 1 to binarize
    residuals_packed = reinterpret(UInt8, BitArray(vec(bucket_indices)).chunks)          # flatten out the bits, and pack them into UInt8
    residuals_packed = reshape(residuals_packed, (Int(dim / 8) * nbits, num_embeddings)) # reshape back to get compressions for each embedding
    @assert ndims(residuals_packed)==2 "ndims(residuals_packed): $(ndims(residuals_packed))"
    @assert size(residuals_packed)[2]==size(residuals)[2] "size(residuals_packed): $(size(residuals_packed)), size(residuals): $(size(residuals))"
    @assert residuals_packed isa AbstractMatrix{UInt8} "$(typeof(residuals_packed))"

    residuals_packed
end

"""
    compress(centroids::Matrix{Float32}, bucket_cutoffs::Vector{Float32},
        dim::Int, nbits::Int, embs::AbstractMatrix{Float32})

Compress a matrix of embeddings into a compact representation.

All embeddings are compressed to their nearest centroid IDs and
their quantized residual vectors (where the quantization is done
in `nbits` bits). If `emb` denotes an embedding and `centroid`
is is nearest centroid, the residual vector is defined to be
`emb - centroid`.

# Arguments

  - `centroids`: The matrix of centroids.
  - `bucket_cutoffs`: Cutoffs used to determine residual buckets.
  - `dim`: The embedding dimension (see [`ColBERTConfig`](@ref)).
  - `nbits`: Number of bits to compress the residuals into.
  - `embs`: The input embeddings to be compressed.

# Returns

A tuple containing a vector of codes and the compressed residuals matrix.
"""
function compress(centroids::Matrix{Float32}, bucket_cutoffs::Vector{Float32},
        dim::Int, nbits::Int, embs::AbstractMatrix{Float32})
    codes, residuals = Vector{UInt32}(), Vector{Matrix{UInt8}}()

    offset = 1
    bsize = 1 << 18
    while (offset <= size(embs)[2])                # batch on second dimension
        batch = embs[:, offset:min(size(embs)[2], offset + bsize - 1)]
        codes_ = compress_into_codes(centroids, batch) # get centroid codes
        centroids_ = centroids[:, codes_]    # get corresponding centroids
        residuals_ = batch - centroids_
        append!(codes, codes_)
        push!(residuals, binarize(dim, nbits, bucket_cutoffs, residuals_))
        offset += bsize
    end
    residuals = cat(residuals..., dims = 2)

    @assert ndims(codes)==1 "ndims(codes): $(ndims(codes))"
    @assert ndims(residuals)==2 "ndims(residuals): $(ndims(residuals))"
    @assert length(codes)==size(embs)[2] "length(codes): $(length(codes)), size(embs): $(size(embs))"
    @assert size(residuals)[2]==size(embs)[2] "size(residuals): $(size(residuals)), size(embs): $(size(embs))"
    @assert codes isa AbstractVector{UInt32} "$(typeof(codes))"
    @assert residuals isa AbstractMatrix{UInt8} "$(typeof(residuals))"

    codes, residuals
end

function decompress_residuals(dim::Int, nbits::Int, bucket_weights::Vector{Float32},
        binary_residuals::AbstractMatrix{UInt8})
    @assert ndims(binary_residuals)==2 "ndims(binary_residuals): $(ndims(binary_residuals))"
    @assert size(binary_residuals)[1]==(dim / 8) * nbits "size(binary_residuals): $(size(binary_residuals)), (dim / 8) * nbits: $((dim / 8) * nbits)"

    # unpacking UInt8 into bits
    unpacked_bits = BitVector()
    for byte in vec(binary_residuals)
        append!(unpacked_bits, [byte & (0x1 << n) != 0 for n in 0:7])
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
    unpacked_bits = sum(unpacked_bits, dims = 1)
    unpacked_bits = unpacked_bits .+ 1                          # adding 1 to get correct bin indices

    # reshaping to get rid of the nbits wide dimension
    unpacked_bits = reshape(unpacked_bits, size(unpacked_bits)[2:end]...)
    embeddings = bucket_weights[unpacked_bits]

    @assert ndims(embeddings)==2 "ndims(embeddings): $(ndims(embeddings))"
    @assert size(embeddings)[2]==size(binary_residuals)[2] "size(embeddings): $(size(embeddings)), size(binary_residuals): $(size(binary_residuals)) "
    @assert embeddings isa AbstractMatrix{Float32} "$(typeof(embeddings))"

    embeddings
end

function decompress(
        dim::Int, nbits::Int, centroids::Matrix{Float32}, bucket_weights::Vector{Float32},
        codes::Vector{UInt32}, residuals::AbstractMatrix{UInt8})
    @assert ndims(codes)==1 "ndims(codes): $(ndims(codes))"
    @assert ndims(residuals)==2 "ndims(residuals): $(ndims(residuals))"
    @assert length(codes)==size(residuals)[2] "length(codes): $(length(codes)), size(residuals): $(size(residuals))"

    # decompress in batches
    D = Vector{AbstractMatrix{Float32}}()
    bsize = 1 << 15
    batch_offset = 1
    while batch_offset <= length(codes)
        batch_codes = codes[batch_offset:min(batch_offset + bsize - 1, length(codes))]
        batch_residuals = residuals[
            :, batch_offset:min(batch_offset + bsize - 1, length(codes))]

        centroids_ = centroids[:, batch_codes]
        residuals_ = decompress_residuals(dim, nbits, bucket_weights, batch_residuals)

        batch_embeddings = centroids_ + residuals_
        batch_embeddings = mapslices(
            v -> iszero(v) ? v : normalize(v), batch_embeddings, dims = 1)
        push!(D, batch_embeddings)

        batch_offset += bsize
    end
    embeddings = cat(D..., dims = 2)

    @assert ndims(embeddings)==2 "ndims(embeddings): $(ndims(embeddings))"
    @assert size(embeddings)[2]==length(codes) "size(embeddings): $(size(embeddings)),  length(codes): $(length(codes))"
    @assert embeddings isa AbstractMatrix{Float32} "$(typeof(embeddings))"

    embeddings
end
