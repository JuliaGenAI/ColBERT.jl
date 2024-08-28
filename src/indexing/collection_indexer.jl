"""
    _sample_pids(num_documents::Int)

Sample PIDs from the collection to be used to compute clusters using a ``k``-means clustering
algorithm.

# Arguments

  - `num_documents`: The total number of documents in the collection. It is assumed that each
    document has an ID (aka PID) in the range of integers between `1` and `num_documents`
    (both inclusive).

# Returns

A `Set` of `Int`s containing the sampled PIDs.
"""
function _sample_pids(num_documents::Int)
    typical_doclen = 120
    num_sampled_pids = 16 * sqrt(typical_doclen * num_documents)
    num_sampled_pids = Int(min(1 + floor(num_sampled_pids), num_documents))
    sampled_pids = Set(sample(1:num_documents, num_sampled_pids))
    @info "# of sampled PIDs = $(length(sampled_pids))"
    sampled_pids
end

"""
    _sample_embeddings(config::ColBERTConfig, checkpoint::Checkpoint,
        collection::Vector{String}, sampled_pids::Set{Int})

Compute embeddings for the PIDs sampled by [`_sample_pids`](@ref).

The embedding array has shape `(D, N)`, where `D` is the
embedding dimension (`128`, after applying the linear layer of the ColBERT model) and `N` is the
total number of embeddings over all documents.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) to be used.
  - `checkpoint`: The [`Checkpoint`] used to encode the passages.
  - `collection`: The underlying collection of passages to get the samples from.
  - `sampled_pids`: Set of PIDs sampled by [`_sample_pids`](@ref).

# Returns

A `Dict` containing the average document length (i.e number of attended tokens) computed
from the sampled documents, and the embedding matrix for the local samples. The matrix has
shape `(D, N)`, where `D` is the embedding dimension (`128`) and `N` is the total number
of embeddings over all the sampled passages.
"""
function _sample_embeddings(bert::HF.HGFBertModel, linear::Layers.Dense,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        dim::Int, index_bsize::Int, doc_token::String,
        skiplist::Vector{Int}, collection::Vector{String})
    # collect all passages with pids in sampled_pids
    sampled_pids = _sample_pids(length(collection))
    sorted_sampled_pids = sort(collect(sampled_pids))
    local_sample = collection[sorted_sampled_pids]

    # get the local sample embeddings
    local_sample_embs, local_sample_doclens = encode_passages(
        bert, linear, tokenizer, local_sample,
        dim, index_bsize, doc_token, skiplist)

    @assert size(local_sample_embs, 2)==sum(local_sample_doclens) "size(local_sample_embs): $(size(local_sample_embs)), sum(local_sample_doclens): $(sum(local_sample_doclens))"
    @assert length(local_sample) == length(local_sample_doclens)

    avg_doclen_est = length(local_sample_doclens) > 0 ?
                     Float32(sum(local_sample_doclens) /
                             length(local_sample_doclens)) :
                     zero(Float32)
    @info "avg_doclen_est = $(avg_doclen_est) \t length(local_sample) = $(length(local_sample))"
    avg_doclen_est, local_sample_embs
end

function _heldout_split(
        sample::AbstractMatrix{Float32}; heldout_fraction::Float32 = 0.05f0)
    num_sample_embs = size(sample, 2)
    sample = sample[:, shuffle(1:num_sample_embs)]
    heldout_size = Int(max(
        1, floor(min(50000, heldout_fraction * num_sample_embs))))
    sample, sample_heldout = sample[
        :, 1:(num_sample_embs - heldout_size)],
    sample[:, (num_sample_embs - heldout_size + 1):num_sample_embs]
    sample, sample_heldout
end

"""
    setup(config::ColBERTConfig, checkpoint::Checkpoint, collection::Vector{String})

Initialize the index by computing some indexing-specific estimates and save the indexing plan
to disk.

The number of chunks into which the document embeddings will be stored is simply computed using
the number of documents and the size of a chunk. A bunch of pids used for initializing the
centroids for the embedding clusters are sampled using the [`_sample_pids`](@ref)
and [`_sample_embeddings`](@ref) functions, and these samples are used to calculate the
average document lengths and the estimated number of embeddings which will be computed across
all documents. Finally, the number of clusters to be used for indexing is computed, and is
proportional to ``16\\sqrt{\\text{Estimated number of embeddings}}``.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) being used to set up the indexing.
  - `checkpoint`: The [`Checkpoint`](@ref) used to compute embeddings.
  - `collection`: The underlying collection of passages to initialize the index for.

# Returns

A `Dict` containing the indexing plan.
"""
function setup(collection::Vector{String}, avg_doclen_est::Float32,
        num_clustering_embs::Int, chunksize::Union{Missing, Int}, nranks::Int)
    chunksize = ismissing(chunksize) ?
                min(25000, 1 + fld(length(collection), nranks)) :
                chunksize
    num_chunks = cld(length(collection), chunksize)

    # computing the number of partitions, i.e clusters
    num_passages = length(collection)
    num_embeddings_est = num_passages * avg_doclen_est
    num_partitions = Int(min(num_clustering_embs,
        floor(2^(floor(log2(16 * sqrt(num_embeddings_est)))))))

    @info "Creating $(num_partitions) clusters."
    @info "Estimated $(num_embeddings_est) embeddings."

    Dict{String, Any}(
        "chunksize" => chunksize,
        "num_chunks" => num_chunks,
        "num_partitions" => num_partitions,
        "num_documents" => length(collection),
        "num_embeddings_est" => num_embeddings_est,
        "avg_doclen_est" => avg_doclen_est
    )
end

"""
    _compute_avg_residuals(
        nbits::Int, centroids::AbstractMatrix{Float32},
        heldout::AbstractMatrix{Float32})

Compute the average residuals and other statistics of the held-out sample embeddings.

# Arguments

  - `nbits`: The number of bits used to compress the residuals.
  - `centroids`: A matrix containing the centroids of the computed using a ``k``-means
    clustering algorithm on the sampled embeddings. Has shape `(D, indexer.num_partitions)`,
    where `D` is the embedding dimension (`128`) and `indexer.num_partitions` is the number
    of clusters.
  - `heldout`: A matrix containing the held-out embeddings, computed using
    [`_concatenate_and_split_sample`](@ref).

# Returns

A tuple `bucket_cutoffs, bucket_weights, avg_residual`, which will be used in
compression/decompression of residuals.
"""
function _compute_avg_residuals!(
        nbits::Int, centroids::AbstractMatrix{Float32},
        heldout::AbstractMatrix{Float32}, codes::AbstractVector{UInt32})
    @assert length(codes) == size(heldout, 2)

    compress_into_codes!(codes, centroids, heldout)                         # get centroid codes
    heldout_reconstruct = centroids[:, codes]                               # get corresponding centroids
    heldout_avg_residual = heldout - heldout_reconstruct                    # compute the residual

    avg_residual = mean(abs.(heldout_avg_residual), dims = 2)               # for each dimension, take mean of absolute values of residuals

    # computing bucket weights and cutoffs
    num_options = 2^nbits
    quantiles = Vector(0:(num_options - 1)) / num_options
    bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[2:end],
    quantiles .+ (0.5 / num_options)

    bucket_cutoffs = Float32.(quantile(
        heldout_avg_residual, bucket_cutoffs_quantiles))
    bucket_weights = Float32.(quantile(
        heldout_avg_residual, bucket_weights_quantiles))
    @assert bucket_cutoffs isa AbstractVector{Float32} "$(typeof(bucket_cutoffs))"
    @assert bucket_weights isa AbstractVector{Float32} "$(typeof(bucket_weights))"

    @info "Got bucket_cutoffs_quantiles = $(bucket_cutoffs_quantiles) and bucket_weights_quantiles = $(bucket_weights_quantiles)"
    @info "Got bucket_cutoffs = $(bucket_cutoffs) and bucket_weights = $(bucket_weights)"

    bucket_cutoffs, bucket_weights, mean(avg_residual)
end

"""
    train(sample::AbstractMatrix{Float32}, heldout::AbstractMatrix{Float32},
        num_partitions::Int, nbits::Int, kmeans_niters::Int)

Compute centroids using a ``k``-means clustering algorithn, and store the compression information
on disk.

Average residuals and other compression data is computed via the [`_compute_avg_residuals`](@ref)
function.

# Arguments

  - `sample`: The matrix of sampled embeddings used to compute clusters.
  - `heldout`: The matrix of sample embeddings used to compute the residual information.
  - `num_partitions`: The number of clusters to compute.
  - `nbits`: The number of bits used to encode the residuals.
  - `kmeans_niters`: The maximum number of iterations in the ``k``-means algorithm.

# Returns

A `Dict` containing the residual codec, i.e information used to compress/decompress residuals.
"""
function train(
        sample::AbstractMatrix{Float32}, heldout::AbstractMatrix{Float32},
        num_partitions::Int, nbits::Int, kmeans_niters::Int)
    # computing clusters
    sample = sample |> Flux.gpu
    centroids = sample[:, randperm(size(sample, 2))[1:num_partitions]]
    # TODO: put point_bsize in the config!
    kmeans_gpu_onehot!(
        sample, centroids, num_partitions; max_iters = kmeans_niters)
    @assert(size(centroids, 2)==num_partitions,
        "size(centroids): $(size(centroids)), num_partitions: $(num_partitions)")
    @assert(centroids isa AbstractMatrix{Float32}, "$(typeof(centroids))")

    # computing average residuals
    heldout = heldout |> Flux.gpu
    codes = zeros(UInt32, size(heldout, 2)) |> Flux.gpu
    bucket_cutoffs, bucket_weights, avg_residual = _compute_avg_residuals!(
        nbits, centroids, heldout, codes)
    @info "avg_residual = $(avg_residual)"

    Flux.cpu(centroids), bucket_cutoffs, bucket_weights, avg_residual
end

"""
    index(config::ColBERTConfig, checkpoint::Checkpoint, collection::Vector{String})

Build the index using `indexer`.

The documents are processed in batches of size `chunksize`, determined by the config
(see [`ColBERTConfig`](@ref) and [`setup`](@ref)). Embeddings and document lengths are
computed for each batch (see [`encode_passages`](@ref)), and they are saved to disk
along with relevant metadata (see [`save_chunk`](@ref)).

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) being used.
  - `checkpoint`: The [`Checkpoint`](@ref) to compute embeddings.
  - `collection`: The collection to index.
"""
function index(index_path::String, bert::HF.HGFBertModel, linear::Layers.Dense,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        collection::Vector{String}, dim::Int, index_bsize::Int,
        doc_token::String, skiplist::Vector{Int}, num_chunks::Int,
        chunksize::Int, centroids::AbstractMatrix{Float32},
        bucket_cutoffs::AbstractVector{Float32}, nbits::Int)
    for (chunk_idx, passage_offset) in zip(
        1:num_chunks, 1:chunksize:length(collection))
        passage_end_offset = min(
            length(collection), passage_offset + chunksize - 1)

        # get embeddings for batch
        embs, doclens = encode_passages(bert, linear, tokenizer,
            collection[passage_offset:passage_end_offset],
            dim, index_bsize, doc_token, skiplist)
        @assert embs isa AbstractMatrix{Float32} "$(typeof(embs))"
        @assert doclens isa AbstractVector{Int} "$(typeof(doclens))"

        # compress embeddings
        codes, residuals = compress(centroids, bucket_cutoffs, dim, nbits, embs)

        # save the chunk
        @info "Saving chunk $(chunk_idx): \t $(passage_end_offset - passage_offset + 1) passages and $(size(embs)[2]) embeddings. From passage #$(passage_offset) onward."
        save_chunk(
            index_path, codes, residuals, chunk_idx, passage_offset, doclens)
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
function check_chunk_exists(index_path::String, chunk_idx::Int)
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

function _check_all_files_are_saved(index_path::String)
    plan_metadata = JSON.parsefile(joinpath(index_path, "plan.json"))

    @info "Checking if all files are saved."
    for chunk_idx in 1:(plan_metadata["num_chunks"])
        if !(check_chunk_exists(index_path, chunk_idx))
            @error "Some files for chunk $(chunk_idx) are missing!"
        end
    end
    @info "Found all files!"
end

function _collect_embedding_id_offset(chunk_emb_counts::Vector{Int})
    length(chunk_emb_counts) > 0 || return zeros(Int, 1)
    chunk_embedding_offsets = cat([1], chunk_emb_counts[1:(end - 1)], dims = 1)
    chunk_embedding_offsets = cumsum(chunk_embedding_offsets)
    sum(chunk_emb_counts), chunk_embedding_offsets
end

function _build_ivf(codes::Vector{UInt32}, num_partitions::Int)
    ivf, values = sortperm(codes), sort(codes)
    ivf_lengths = counts(values, num_partitions)
    ivf, ivf_lengths
end
