"""
    encode_passages(
        config::ColBERTConfig, checkpoint::Checkpoint, passages::Vector{String})

Encode a list of passages using `checkpoint`.

The given `passages` are run through the underlying BERT model and the linear layer to
generate the embeddings, after doing relevant document-specific preprocessing.
See [`docFromText`](@ref) for more details.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) to be used.
  - `checkpoint`: The [`Checkpoint`](@ref) used to encode the passages.
  - `passages`: A list of strings representing the passages to be encoded.

# Returns

A tuple `embs, doclens` where:

  - `embs::AbstractMatrix{Float32}`: The full embedding matrix. Of shape `(D, N)`,
    where `D` is the embedding dimension and `N` is the total number of embeddings across all the passages.
  - `doclens::AbstractVector{Int}`: A vector of document lengths for each passage,
    i.e the total number of attended tokens for each document passage.
"""
function encode_passages(
        config::ColBERTConfig, checkpoint::Checkpoint, passages::Vector{String})
    @info "Encoding $(length(passages)) passages."

    if length(passages) == 0
        error("The list of passages to encode is empty!")
    end

    embs, doclens = Vector{AbstractMatrix{Float32}}(), Vector{Int}()
    # batching here to avoid storing intermediate embeddings on GPU
    # batching also occurs inside docFromText to do batch packing optimizations
    for passages_batch in batch(passages, config.index_bsize * 50)
        embs_, doclens_ = docFromText(config, checkpoint, passages_batch,
            config.index_bsize)
        push!(embs, embs_)
        append!(doclens, vec(doclens_))
    end
    embs = cat(embs..., dims = 2)
    embs, doclens
end

"""
    _sample_pids(num_documents::Int)

Sample PIDs from the collection to be used to compute clusters using a ``k``-means clustering algorithm.

# Arguments

  - `num_documents`: The total number of documents in the collection. It is assumed that each document has an ID
    (aka PID) in the range of integers between `1` and `num_documents` (both inclusive).

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

The embeddings for the sampled documents are saved in a file named `sample.jld2` with it's path
specified by the indexing directory. This embedding array has shape `(D, N)`, where `D` is the
embedding dimension (`128`, after applying the linear layer of the ColBERT model) and `N` is the
total number of embeddings over all documents.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) to be used.
  - `checkpoint`: The [`Checkpoint`] used to encode the passages.
  - `collection`: The underlying collection of passages to get the samples from.
  - `sampled_pids`: Set of PIDs sampled by [`_sample_pids`](@ref).

# Returns

The average document length (i.e number of attended tokens) computed from the sampled documents.
"""
function _sample_embeddings(config::ColBERTConfig, checkpoint::Checkpoint,
        collection::Vector{String}, sampled_pids::Set{Int})
    # collect all passages with pids in sampled_pids
    sorted_sampled_pids = sort(collect(sampled_pids))
    local_sample = collection[sorted_sampled_pids]

    # get the local sample embeddings
    local_sample_embs, local_sample_doclens = encode_passages(
        config, checkpoint, local_sample)
    @debug "Local sample embeddings shape: $(size(local_sample_embs)), \t Local sample doclens: $(local_sample_doclens)"
    @assert size(local_sample_embs)[2]==sum(local_sample_doclens) "size(local_sample_embs): $(size(local_sample_embs)), sum(local_sample_doclens): $(sum(local_sample_doclens))"
    @assert length(local_sample) == length(local_sample_doclens)

    num_sample_embs = size(local_sample_embs)[2]
    avg_doclen_est = length(local_sample_doclens) > 0 ?
                     sum(local_sample_doclens) / length(local_sample_doclens) : 0

    sample_path = joinpath(config.index_path, "sample.jld2")
    @info "avg_doclen_est = $(avg_doclen_est) \t length(local_sample) = $(length(local_sample))"
    @info "Saving sampled embeddings to $(sample_path)."
    JLD2.save_object(sample_path, local_sample_embs)

    avg_doclen_est
end

"""
    setup(config::ColBERTConfig, checkpoint::Checkpoint, collection::Vector{String})

Initialize the index by computing some indexing-specific estimates and save the indexing plan to disk.

The number of chunks into which the document embeddings will be stored is simply computed using the
number of documents and the size of a chunk. A bunch of pids used for initializing the centroids for
the embedding clusters are sampled using the [`_sample_pids`](@ref) and [`_sample_embeddings`](@ref)
functions, and these samples are used to calculate the average document lengths and the estimated number
of embeddings which will be computed across all documents. Finally, the number of clusters  to be used
for indexing is computed, and is proportional to ``16\\sqrt{\\text{Estimated number of embeddings}}``,
and the indexing plan is saved to `plan.json`, with the path being specified by the indexing directory.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) being used to set up the indexing.
  - `checkpoint`: The [`Checkpoint`](@ref) used to compute embeddings.
  - `collection`: The underlying collection of passages to initialize the index for.
"""
function setup(config::ColBERTConfig, checkpoint::Checkpoint, collection::Vector{String})
    isdir(config.index_path) || mkdir(config.index_path)

    chunksize = 0    
    chunksize = ismissing(config.chunksize) ? min(25000, 1 + fld(length(collection), config.nranks)) : config.chunksize
    num_chunks = cld(length(collection), chunksize)

    # sample passages for training centroids later
    sampled_pids = _sample_pids(length(collection))
    avg_doclen_est = _sample_embeddings(config, checkpoint, collection, sampled_pids)

    # computing the number of partitions, i.e clusters
    num_passages = length(collection)
    num_embeddings_est = num_passages * avg_doclen_est
    num_partitions = Int(floor(2^(floor(log2(16 * sqrt(num_embeddings_est))))))

    @info "Creating $(num_partitions) clusters."
    @info "Estimated $(num_embeddings_est) embeddings."

    @info "Saving the index plan to $(joinpath(config.index_path, "plan.json"))."
    open(joinpath(config.index_path, "plan.json"), "w") do io
        JSON.print(io,
            Dict(
                "chunksize" => chunksize,
                "num_chunks" => num_chunks,
                "num_partitions" => num_partitions,
                "num_embeddings_est" => num_embeddings_est,
                "avg_doclen_est" => avg_doclen_est
            ),
            4                                                               # indent
        )
    end

    @info "Saving the config to the indexing path."
    ColBERT.save(config)
end

"""
    _concatenate_and_split_sample(indexer::CollectionIndexer)

Randomly shuffle and split the sampled embeddings.

The sample embeddings saved by the [`setup`](@ref) function are loaded, shuffled randomly, and then split into a `sample` and a `sample_heldout` set, with `sample_heldout` containing a `0.05` fraction of the original sampled embeddings.

# Arguments

  - `index_path`: The path of the index. 

# Returns

The tuple `sample, sample_heldout`.
"""
function _concatenate_and_split_sample(index_path::String)
    # load the sample embeddings
    sample_path = joinpath(index_path, "sample.jld2")
    sample = JLD2.load_object(sample_path)
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
        nbits::Int, centroids::AbstractMatrix{Float32},
        heldout::AbstractMatrix{Float32})
    codes = compress_into_codes(centroids, heldout)                         # get centroid codes
    @assert codes isa AbstractVector{UInt32} "$(typeof(codes))"
    heldout_reconstruct = Flux.gpu(centroids[:, codes])                     # get corresponding centroids
    heldout_avg_residual = Flux.gpu(heldout) - heldout_reconstruct          # compute the residual

    avg_residual = mean(abs.(heldout_avg_residual), dims = 2)               # for each dimension, take mean of absolute values of residuals

    # computing bucket weights and cutoffs
    num_options = 2^nbits
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
function train(config::ColBERTConfig)
    sample, heldout = _concatenate_and_split_sample(config.index_path)
    @assert sample isa AbstractMatrix{Float32} "$(typeof(sample))"
    @assert heldout isa AbstractMatrix{Float32} "$(typeof(heldout))"

    # loading the indexing plan
    plan_metadata = JSON.parsefile(joinpath(config.index_path, "plan.json"))

    centroids = kmeans(sample, plan_metadata["num_partitions"],
        maxiter = config.kmeans_niters, display = :iter).centers
    @assert size(centroids)[2]==plan_metadata["num_partitions"] "size(centroids): $(size(centroids)), num_partitions: $(plan_metadata["num_partitions"])"
    @assert centroids isa AbstractMatrix{Float32} "$(typeof(centroids))"

    bucket_cutoffs, bucket_weights, avg_residual = _compute_avg_residuals(
        config.nbits, centroids, heldout)
    @info "avg_residual = $(avg_residual)"

    save_codec(config.index_path, centroids, bucket_cutoffs, bucket_weights, avg_residual)
end

"""
    save_chunk(saver::IndexSaver, chunk_idx::Int, offset::Int,
        embs::AbstractMatrix{Float32}, doclens::AbstractVector{Int})

Save a single chunk of compressed embeddings and their relevant metadata to disk.

The codes and compressed residuals for the chunk are saved in files named `<chunk_idx>.codec.jld2`. The document lengths are saved in a file named `doclens.<chunk_idx>.jld2`. Relevant metadata, including number of documents in the chunk, number of embeddings and the passage offsets are saved in a file named `<chunk_idx>.metadata.json`.

# Arguments

  - `saver`: The [`IndexSaver`](@ref) containing relevant information to save the chunk.
  - `chunk_idx`: The index of the current chunk being saved.
  - `offset`: The offset in the original document collection where this chunk starts.
  - `embs`: The embeddings matrix for the current chunk.
  - `doclens`: The document lengths vector for the current chunk.
"""
function save_chunk(config::ColBERTConfig, codec::Dict, chunk_idx::Int, passage_offset::Int,
        embs::AbstractMatrix{Float32}, doclens::AbstractVector{Int})
    codes, residuals = compress(codec["centroids"], codec["bucket_cutoffs"], config.dim, config.nbits, embs)
    path_prefix = joinpath(config.index_path, string(chunk_idx))
    @assert length(codes)==size(embs)[2] "length(codes): $(length(codes)), size(embs): $(size(embs))"

    # saving the compressed embeddings
    codes_path = "$(path_prefix).codes.jld2"
    residuals_path = "$(path_prefix).residuals.jld2"
    @info "Saving compressed codes to $(codes_path) and residuals to $(residuals_path)"
    JLD2.save_object(codes_path, codes)
    JLD2.save_object(residuals_path, residuals)

    # saving doclens
    doclens_path = joinpath(
        config.index_path, "doclens.$(chunk_idx).jld2")
    @info "Saving doclens to $(doclens_path)"
    JLD2.save_object(doclens_path, doclens)

    # the metadata
    metadata_path = joinpath(
        config.index_path, "$(chunk_idx).metadata.json")
    @info "Saving metadata to $(metadata_path)"
    open(metadata_path, "w") do io
        JSON.print(io,
            Dict(
                "passage_offset" => passage_offset,
                "num_passages" => length(doclens),
                "num_embeddings" => length(codes)
            ),
            4                                                               # indent
        )
    end
end

"""
    index(indexer::CollectionIndexer; chunksize::Union{Int, Missing} = missing)

Build the index using `indexer`.

The documents are processed in batches of size `chunksize` (see [`enumerate_batches`](@ref)). Embeddings and document lengths are computed for each batch (see [`encode_passages`](@ref)), and they are saved to disk along with relevant metadata (see [`save_chunk`](@ref)).

# Arguments

  - `indexer`: The [`CollectionIndexer`](@ref) used to build the index.
  - `chunksize`: Size of a chunk into which the index is to be stored.
"""
function index(config::ColBERTConfig, checkpoint::Checkpoint, collection::Vector{String})
    codec = load_codec(config.index_path)
    plan_metadata = JSON.parsefile(joinpath(config.index_path, "plan.json"))
    passage_offset = 1
    for chunk_idx in 1:plan_metadata["num_chunks"]
        passage_end_offset = min(length(collection), passage_offset + plan_metadata["chunksize"] - 1) 
        embs, doclens = encode_passages(config, checkpoint, collection[passage_offset:passage_end_offset])
        @assert embs isa AbstractMatrix{Float32} "$(typeof(embs))"
        @assert doclens isa AbstractVector{Int} "$(typeof(doclens))"

        @info "Saving chunk $(chunk_idx): \t $(passage_end_offset - passage_offset + 1) passages and $(size(embs)[2]) embeddings. From passage #$(passage_offset) onward."
        save_chunk(config, codec, chunk_idx, passage_offset, embs, doclens)

        passage_offset += plan_metadata["chunksize"]
    end
end

"""
    finalize(indexer)

Finalize the indexing process by saving all files, collecting embedding ID offsets, building IVF, and updating metadata.

See [`_check_all_files_are_saved`](@ref), [`_collect_embedding_id_offset`](@ref), [`_build_ivf`](@ref) and [`_update_metadata`](@ref) for more details.

# Arguments

  - `indexer::CollectionIndexer`: The [`CollectionIndexer`](@ref) used to finalize the indexing process.
"""
function finalize(index_path::String)
    _check_all_files_are_saved(index_path)
    _collect_embedding_id_offset(index_path)
    _build_ivf(index_path)
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

function _collect_embedding_id_offset(index_path::String)
    plan_metadata = JSON.parsefile(joinpath(index_path, "plan.json"))

    @info "Collecting embedding ID offsets."
    passage_offset = 1
    embedding_offset = 1

    embeddings_offsets = Vector{Int}()
    for chunk_idx in 1:(plan_metadata["num_chunks"])
        metadata_path = joinpath(
            index_path, "$(chunk_idx).metadata.json")

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
    num_embeddings = embedding_offset - 1
    @assert length(embeddings_offsets) == plan_metadata["num_chunks"] 

    @info "Saving the indexing metadata."
    metadata_path = joinpath(index_path, "metadata.json")
    open(metadata_path, "w") do io
        JSON.print(io,
            # TODO: export the config here as well!
            Dict(
                "num_embeddings" => num_embeddings,
                "embeddings_offsets" => embeddings_offsets
            ),
            4
        )
    end
end

function _build_ivf(index_path::String)
    plan_metadata = JSON.parsefile(joinpath(index_path, "plan.json"))

    @info "Building the centroid to embedding IVF."
    codes = Vector{UInt32}()

    @info "Loading codes for each embedding."
    for chunk_idx in 1:(plan_metadata["num_chunks"])
        chunk_codes = load_codes(index_path, chunk_idx)
        append!(codes, chunk_codes)
    end
    @assert codes isa AbstractVector{UInt32} "$(typeof(codes))"

    @info "Sorting the codes."
    ivf, values = sortperm(codes), sort(codes)

    @info "Getting unique codes and their counts."
    ivf_lengths = counts(values, 1:(plan_metadata["num_partitions"]))

    @info "Saving the IVF."
    ivf_path = joinpath(index_path, "ivf.jld2")
    ivf_lengths_path = joinpath(index_path, "ivf_lengths.jld2")
    JLD2.save_object(ivf_path, ivf)
    JLD2.save_object(ivf_lengths_path, ivf_lengths)
end
