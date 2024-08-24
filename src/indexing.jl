struct Indexer
    config::ColBERTConfig
    checkpoint::Checkpoint
    collection::Vector{String}
end

"""
    Indexer(config::ColBERTConfig)

Type representing an ColBERT indexer.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) used to build the index.

# Returns

An [`Indexer`] wrapping a [`ColBERTConfig`](@ref), a [`Checkpoint`](@ref) and
a collection of documents to index.
"""
function Indexer(config::ColBERTConfig)
    base_colbert = BaseColBERT(config.checkpoint)
    checkpoint = Checkpoint(base_colbert, config)
    collection = readlines(config.collection)

    @info "Loaded ColBERT layers from the $(checkpoint) HuggingFace checkpoint."
    @info "Loaded $(length(collection)) documents from $(config.collection)."

    Indexer(config, checkpoint, collection)
end

"""
    index(indexer::Indexer)

Build an index given the configuration stored in `indexer`.

# Arguments

  - `indexer`: An `Indexer` which is used to build the index on disk.
"""
function index(indexer::Indexer)
    if isdir(indexer.config.index_path)
        @info "Index at $(indexer.config.index_path) already exists! Skipping indexing."
        return
    end

    # getting and saving the indexing plan
    isdir(indexer.config.index_path) || mkdir(indexer.config.index_path)
    sample, sample_heldout, plan_dict = setup(
        indexer.config, indexer.checkpoint, indexer.collection)
    @info "Saving the index plan to $(joinpath(indexer.config.index_path, "plan.json"))."
    open(joinpath(indexer.config.index_path, "plan.json"), "w") do io
        JSON.print(io,
            plan_dict,
            4                                                               # indent
        )
    end
    @info "Saving the config to the indexing path."
    ColBERT.save(indexer.config)

    # training/clustering
    @assert sample isa AbstractMatrix{Float32} "$(typeof(sample))"
    @assert sample_heldout isa AbstractMatrix{Float32} "$(typeof(sample_heldout))"
    @info "Training the clusters."
    centroids, bucket_cutoffs, bucket_weights, avg_residual = train(
        sample, sample_heldout, plan_dict["num_partitions"],
        indexer.config.nbits, indexer.config.kmeans_niters)
    save_codec(
        indexer.config.index_path, centroids, bucket_cutoffs,
        bucket_weights, avg_residual)
    sample, sample_heldout, centroids = nothing, nothing, nothing           # these are big arrays

    # indexing
    @info "Building the index."
    index(indexer.config, indexer.checkpoint, indexer.collection)

    # finalizing
    @info "Running some final checks."
    _check_all_files_are_saved(indexer.config.index_path)
    _collect_embedding_id_offset(indexer.config.index_path)
    _build_ivf(indexer.config.index_path)
end
