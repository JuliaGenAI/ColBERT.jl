"""
    CollectionIndexer(config::ColBERTConfig, encoder::CollectionEncoder, saver::IndexSaver)

Structure which performs all the index-building operations, including sampling initial centroids, clustering, computing document embeddings, compressing and building the `ivf`.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) used to build the model.
  - `encoder`: The [`CollectionEncoder`](@ref) to be used for encoding documents.
  - `saver`: The [`IndexSaver`](@ref), responsible for saving the index to disk.

# Returns

A [`CollectionIndexer`](@ref) object, containing all indexing-related information. See the [`setup`](@ref), [`train`](@ref), [`index`](@ref) and [`finalize`](@ref) functions for building the index.
"""
mutable struct CollectionIndexer
    config::ColBERTConfig
    encoder::CollectionEncoder
    saver::IndexSaver
    plan_path::String
    num_chunks::Int
    num_embeddings_est::Float64
    num_partitions::Int
    num_sample_embs::Int
    avg_doclen_est::Float64
    embeddings_offsets::Vector{Int}
    num_embeddings::Int
    metadata_path::String
end

function CollectionIndexer(
        config::ColBERTConfig, encoder::CollectionEncoder, saver::IndexSaver)
    plan_path = joinpath(config.index_path, "plan.json")
    metadata_path = joinpath(config.index_path, "metadata.json")

    CollectionIndexer(
        config,
        encoder,
        saver,
        plan_path,
        0,              # num_chunks
        0.0,            # num_embeddings_est
        0,              # num_partitions
        0,              # num_sample_embs
        0.0,            # avg_doclen_est
        [],             # embeddings_offsets
        0,              # num_embeddings
        metadata_path
    )
end

"""
    _sample_pids(indexer::CollectionIndexer)

Sample PIDs from the collection to be used to compute clusters using a ``k``-means clustering algorithm.

# Arguments

  - `indexer`: The collection indexer object containing the collection of passages to be indexed.

# Returns

A `Set` of `Int`s containing the sampled PIDs.
"""
function _sample_pids(indexer::CollectionIndexer)
    num_passages = length(indexer.config.collection.data)
    typical_doclen = 120
    num_sampled_pids = 16 * sqrt(typical_doclen * num_passages)
    num_sampled_pids = Int(min(1 + floor(num_sampled_pids), num_passages))

    sampled_pids = Set(sample(1:num_passages, num_sampled_pids))
    @info "# of sampled PIDs = $(length(sampled_pids))"
    sampled_pids
end

"""
    _sample_embeddings(indexer::CollectionIndexer, sampled_pids::Set{Int})

Compute embeddings for the PIDs sampled by [`_sample_pids`](@ref), compute the average document length using the embeddings, and save the sampled embeddings to disk.

The embeddings for the sampled documents are saved in a file named `sample.jld2` with it's path specified by the indexing directory. This embedding array has shape `(D, N)`, where `D` is the embedding dimension (`128`, after applying the linear layer of the ColBERT model) and `N` is the total number of embeddings over all documents.

Sample the passages with `pid` in `sampled_pids` from the `collection` and compute the average passage length. The function returns a tuple containing the embedded passages and the average passage length.

# Arguments

  - `indexer`: An instance of `CollectionIndexer`.
  - `sampled_pids`: Set of PIDs sampled by [`_sample_pids`](@ref).

# Returns

The average document length (i.e number of attended tokens) computed from the sampled documents.
"""
function _sample_embeddings(indexer::CollectionIndexer, sampled_pids::Set{Int})
    # collect all passages with pids in sampled_pids
    collection = indexer.config.collection
    sorted_sampled_pids = sort(collect(sampled_pids))
    local_sample = collection.data[sorted_sampled_pids]

    local_sample_embs, local_sample_doclens = encode_passages(indexer.encoder, local_sample)
    @debug "Local sample embeddings shape: $(size(local_sample_embs)), \t Local sample doclens: $(local_sample_doclens)"
    @assert size(local_sample_embs)[2]==sum(local_sample_doclens) "size(local_sample_embs): $(size(local_sample_embs)), sum(local_sample_doclens): $(sum(local_sample_doclens))"

    indexer.num_sample_embs = size(local_sample_embs)[2]
    indexer.avg_doclen_est = length(local_sample_doclens) > 0 ?
                             sum(local_sample_doclens) / length(local_sample_doclens) : 0

    sample_path = joinpath(indexer.config.index_path, "sample.jld2")
    @info "avg_doclen_est = $(indexer.avg_doclen_est) \t length(local_sample) = $(length(local_sample))"
    @info "Saving sampled embeddings to $(sample_path)."
    JLD2.save(sample_path, Dict("local_sample_embs" => local_sample_embs))

    indexer.avg_doclen_est
end

"""
    _save_plan(indexer::CollectionIndexer)

Save the indexing plan to a JSON file.

Information about the number of chunks, number of clusters, estimated number of embeddings over all documents and the estimated average document length is saved to a file named `plan.json`, with directory specified by the indexing directory.

# Arguments

  - `indexer`: The `CollectionIndexer` object that contains the index plan to be saved.
"""
function _save_plan(indexer::CollectionIndexer)
    @info "Saving the index plan to $(indexer.plan_path)."
    # TODO: export the config as json as well
    open(indexer.plan_path, "w") do io
        JSON.print(io,
            Dict(
                "num_chunks" => indexer.num_chunks,
                "num_partitions" => indexer.num_partitions,
                "num_embeddings_est" => indexer.num_embeddings_est,
                "avg_doclen_est" => indexer.avg_doclen_est
            ),
            4                                                               # indent
        )
    end
end

"""
    setup(indexer::CollectionIndexer)

Initialize `indexer` by computing some indexing-specific estimates and save the indexing plan to disk.

The number of chunks into which the document embeddings will be stored (`indexer.num_chunks`) is simply computed using the number of documents and the size of a chunk obtained from [`get_chunksize`](@ref). A bunch of pids used for initializing the centroids for the embedding clusters are sampled using the [`_sample_pids`](@ref) and [`_sample_embeddings`](@ref) functions, and these samples are used to calculate the average document lengths and the estimated number of embeddings which will be computed across all documents. Finally, the number of clusters (`indexer.num_partitions`) to be used for indexing is computed, and is proportional to ``16\\sqrt{\\text{Estimated number of embeddings}}``, and the indexing plan is saved to `plan.json` (see [`_save_plan`](@ref)) in the indexing directory.

# Arguments

  - `indexer::CollectionIndexer`: The indexer to be initialized.
"""
function setup(indexer::CollectionIndexer)
    collection = indexer.config.collection
    indexer.num_chunks = Int(ceil(length(collection.data) / get_chunksize(
        collection, indexer.config.nranks)))

    # sample passages for training centroids later
    sampled_pids = _sample_pids(indexer)
    avg_doclen_est = _sample_embeddings(indexer, sampled_pids)

    # computing the number of partitions, i.e clusters
    num_passages = length(indexer.config.collection.data)
    indexer.num_embeddings_est = num_passages * avg_doclen_est
    indexer.num_partitions = Int(floor(2^(floor(log2(16 *
                                                     sqrt(indexer.num_embeddings_est))))))

    @info "Creating $(indexer.num_partitions) clusters."
    @info "Estimated $(indexer.num_embeddings_est) embeddings."

    _save_plan(indexer)
end

"""
    _concatenate_and_split_sample(indexer::CollectionIndexer)

Randomly shuffle and split the sampled embeddings.

The sample embeddings saved by the [`setup`](@ref) function are loaded, shuffled randomly, and then split into a `sample` and a `sample_heldout` set, with `sample_heldout` containing a `0.05` fraction of the original sampled embeddings.

# Arguments

  - `indexer`: The [`CollectionIndexer`](@ref).

# Returns

The tuple `sample, sample_heldout`.
"""
function _concatenate_and_split_sample(indexer::CollectionIndexer)
    # load the sample embeddings
    sample_path = joinpath(indexer.config.index_path, "sample.jld2")
    sample = JLD2.load(sample_path, "local_sample_embs")
    @debug "Original sample shape: $(size(sample))"

    # randomly shuffle embeddings
    num_local_sample_embs = size(sample)[2]
    sample = sample[:, shuffle(1:num_local_sample_embs)]

    # split the sample to get a heldout set
    heldout_fraction = 0.05
    heldout_size = Int(floor(min(50000, heldout_fraction * num_local_sample_embs)))
    sample, sample_heldout = sample[:, 1:(num_local_sample_embs - heldout_size)],
    sample[:, (num_local_sample_embs - heldout_size + 1):num_local_sample_embs]

    @debug "Split sample sizes: sample size: $(size(sample)), \t sample_heldout size: $(size(sample_heldout))"
    sample, sample_heldout
end

"""
    _compute_avg_residuals(indexer::CollectionIndexer, centroids::AbstractMatrix{Float32},
        heldout::AbstractMatrix{Float32})

Compute the average residuals and other statistics of the held-out sample embeddings.

# Arguments

  - `indexer`: The underlying [`CollectionIndexer`](@ref).
  - `centroids`: A matrix containing the centroids of the computed using a ``k``-means clustering algorithm on the sampled embeddings. Has shape `(D, indexer.num_partitions)`, where `D` is the embedding dimension (`128`) and `indexer.num_partitions` is the number of clusters.
  - `heldout`: A matrix containing the held-out embeddings, computed using [`_concatenate_and_split_sample`](@ref).

# Returns

A tuple `bucket_cutoffs, bucket_weights, avg_residual`.
"""
function _compute_avg_residuals(
        indexer::CollectionIndexer, centroids::AbstractMatrix{Float32},
        heldout::AbstractMatrix{Float32})
    compressor = ResidualCodec(
        indexer.config, centroids, 0.0, Vector{Float32}(), Vector{Float32}())
    codes = compress_into_codes(compressor, heldout)             # get centroid codes
    @assert codes isa AbstractVector{UInt32} "$(typeof(codes))"
    heldout_reconstruct = Flux.gpu(compressor.centroids[:, codes])         # get corresponding centroids
    heldout_avg_residual = Flux.gpu(heldout) - heldout_reconstruct         # compute the residual

    avg_residual = mean(abs.(heldout_avg_residual), dims = 2)    # for each dimension, take mean of absolute values of residuals

    # computing bucket weights and cutoffs
    num_options = 2^indexer.config.nbits
    quantiles = Vector(0:(num_options - 1)) / num_options
    bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[2:end],
    quantiles .+ (0.5 / num_options)

    bucket_cutoffs = Float32.(quantile(heldout_avg_residual, bucket_cutoffs_quantiles))
    bucket_weights = Float32.(quantile(heldout_avg_residual, bucket_weights_quantiles))
    @assert bucket_cutoffs isa AbstractVector{Float32} "$(typeof(bucket_cutoffs))"
    @assert bucket_weights isa AbstractVector{Float32} "$(typeof(bucket_weights))"

    @info "Got bucket_cutoffs_quantiles = $(bucket_cutoffs_quantiles) and bucket_weights_quantiles = $(bucket_weights_quantiles)"
    @info "Got bucket_cutoffs = $(bucket_cutoffs) and bucket_weights = $(bucket_weights)"

    bucket_cutoffs, bucket_weights, mean(avg_residual)
end

"""
    train(indexer::CollectionIndexer)

Train a [`CollectionIndexer`](@ref) by computing centroids using a ``k``-means clustering algorithn, and store the compression information on disk.

Average residuals and other compression data is computed via the [`_compute_avg_residuals`](@ref) function, and the codec is saved on disk using [`save_codec`](@ref).

# Arguments

  - `indexer::CollectionIndexer`: The [`CollectionIndexer`](@ref) to be trained.
"""
function train(indexer::CollectionIndexer)
    sample, heldout = _concatenate_and_split_sample(indexer)
    @assert sample isa AbstractMatrix{Float32} "$(typeof(sample))"
    @assert heldout isa AbstractMatrix{Float32} "$(typeof(heldout))"

    centroids = kmeans(sample, indexer.num_partitions,
        maxiter = indexer.config.kmeans_niters, display = :iter).centers
    @assert size(centroids)[2]==indexer.num_partitions "size(centroids): $(size(centroids)), indexer.num_partitions: $(indexer.num_partitions)"
    @assert centroids isa AbstractMatrix{Float32} "$(typeof(centroids))"

    bucket_cutoffs, bucket_weights, avg_residual = _compute_avg_residuals(
        indexer, centroids, heldout)
    @info "avg_residual = $(avg_residual)"

    codec = ResidualCodec(
        indexer.config, centroids, avg_residual, bucket_cutoffs, bucket_weights)
    indexer.saver.codec = codec
    save_codec(indexer.saver)
end

"""
    index(indexer::CollectionIndexer; chunksize::Union{Int, Missing} = missing)

Build the index using `indexer`.

The documents are processed in batches of size `chunksize` (see [`enumerate_batches`](@ref)). Embeddings and document lengths are computed for each batch (see [`encode_passages`](@ref)), and they are saved to disk along with relevant metadata (see [`save_chunk`](@ref)).

# Arguments

  - `indexer`: The [`CollectionIndexer`](@ref) used to build the index.
  - `chunksize`: Size of a chunk into which the index is to be stored.
"""
function index(indexer::CollectionIndexer; chunksize::Union{Int, Missing} = missing)
    load_codec!(indexer.saver)                  # load the codec objects
    batches = enumerate_batches(
        indexer.config.collection, chunksize = chunksize,
        nranks = indexer.config.nranks)
    for (chunk_idx, offset, passages) in batches
        # TODO: add functionality to not re-write chunks if they already exist! 
        # TODO: add multiprocessing to this step!
        embs, doclens = encode_passages(indexer.encoder, passages)
        @assert embs isa AbstractMatrix{Float32} "$(typeof(embs))"
        @assert doclens isa AbstractVector{Int} "$(typeof(doclens))"

        @info "Saving chunk $(chunk_idx): \t $(length(passages)) passages and $(size(embs)[2]) embeddings. From offset #$(offset) onward."
        save_chunk(indexer.saver, chunk_idx, offset, embs, doclens)
    end
end

"""
    finalize(indexer::CollectionIndexer)

Finalize the indexing process by saving all files, collecting embedding ID offsets, building IVF, and updating metadata.

See [`_check_all_files_are_saved`](@ref), [`_collect_embedding_id_offset`](@ref), [`_build_ivf`](@ref) and [`_update_metadata`](@ref) for more details.

# Arguments

  - `indexer::CollectionIndexer`: The [`CollectionIndexer`](@ref) used to finalize the indexing process.
"""
function finalize(indexer::CollectionIndexer)
    _check_all_files_are_saved(indexer)
    _collect_embedding_id_offset(indexer)
    _build_ivf(indexer)
    _update_metadata(indexer)
end

function _check_all_files_are_saved(indexer::CollectionIndexer)
    @info "Checking if all files are saved."
    for chunk_idx in 1:(indexer.num_chunks)
        if !(check_chunk_exists(indexer.saver, chunk_idx))
            @error "Could not find chunk $(chunk_idx)!"
        end
    end
    @info "Found all files!"
end

function _collect_embedding_id_offset(indexer::CollectionIndexer)
    @info "Collecting embedding ID offsets."
    passage_offset = 1
    embedding_offset = 1

    embeddings_offsets = Vector{Int}()
    for chunk_idx in 1:(indexer.num_chunks)
        metadata_path = joinpath(
            indexer.config.index_path, "$(chunk_idx).metadata.json")

        chunk_metadata = open(metadata_path, "r") do io
            chunk_metadata = JSON.parse(io)
        end

        chunk_metadata["embedding_offset"] = embedding_offset
        push!(embeddings_offsets, embedding_offset)

        passage_offset += chunk_metadata["num_passages"]
        embedding_offset += chunk_metadata["num_embeddings"]

        open(metadata_path, "w") do io
            JSON.print(io, chunk_metadata, 4)
        end
    end

    indexer.num_embeddings = embedding_offset - 1
    indexer.embeddings_offsets = embeddings_offsets
end

function _build_ivf(indexer::CollectionIndexer)
    @info "Building the centroid to embedding IVF."
    codes = Vector{UInt32}()

    @info "Loading codes for each embedding."
    for chunk_idx in 1:(indexer.num_chunks)
        chunk_codes = load_codes(indexer.saver.codec, chunk_idx)
        append!(codes, chunk_codes)
    end
    @assert codes isa AbstractVector{UInt32} "$(typeof(codes))"

    @info "Sorting the codes."
    ivf, values = sortperm(codes), sort(codes)

    @info "Getting unique codes and their counts."
    ivf_lengths = counts(values, 1:(indexer.num_partitions))

    @info "Saving the IVF."
    ivf_path = joinpath(indexer.config.index_path, "ivf.jld2")
    JLD2.save(ivf_path, Dict(
        "ivf" => ivf,
        "ivf_lengths" => ivf_lengths
    ))
end

function _update_metadata(indexer::CollectionIndexer)
    @info "Saving the indexing metadata."
    metadata_path = joinpath(indexer.config.index_path, "metadata.json")

    open(metadata_path, "w") do io
        JSON.print(io,
            # TODO: export the config here as well!
            Dict(
                "num_chunks" => indexer.num_chunks,
                "num_partitions" => indexer.num_partitions,
                "num_embeddings" => indexer.num_embeddings,
                "avg_doclen" => Int(floor(indexer.num_embeddings /
                                          length(indexer.config.collection.data)))
            ),
            4
        )
    end
end
