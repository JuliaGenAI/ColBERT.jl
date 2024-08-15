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
    base_colbert = BaseColBERT(config)
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
    plan_dict = setup(indexer.config, indexer.checkpoint, indexer.collection)

    sample_path = joinpath(indexer.config.index_path, "sample.jld2")
    @info "Saving sampled embeddings to $(sample_path)."
    JLD2.save_object(sample_path, plan_dict["local_sample_embs"])

    @info "Saving the index plan to $(joinpath(indexer.config.index_path, "plan.json"))."
    open(joinpath(indexer.config.index_path, "plan.json"), "w") do io
        JSON.print(io,
            Dict(
                "chunksize" => plan_dict["chunksize"],
                "num_chunks" => plan_dict["num_chunks"],
                "num_partitions" => plan_dict["num_partitions"],
                "num_embeddings_est" => plan_dict["num_embeddings_est"],
                "avg_doclen_est" => plan_dict["avg_doclen_est"]
            ),
            4                                                               # indent
        )
    end

    @info "Saving the config to the indexing path."
    ColBERT.save(indexer.config)

    # training/clustering
    sample, heldout = _concatenate_and_split_sample(indexer.config.index_path)
    @assert sample isa AbstractMatrix{Float32} "$(typeof(sample))"
    @assert heldout isa AbstractMatrix{Float32} "$(typeof(heldout))"

    codec = train(sample, heldout, plan_dict["num_partitions"],
        indexer.config.nbits, indexer.config.kmeans_niters)
    save_codec(indexer.config.index_path, codec["centroids"], codec["bucket_cutoffs"],
        codec["bucket_weights"], codec["avg_residual"])

    # indexing
    index(indexer.config, indexer.checkpoint, indexer.collection)

    # finalizing
    _check_all_files_are_saved(indexer.config.index_path)
    _collect_embedding_id_offset(indexer.config.index_path)
    _build_ivf(indexer.config.index_path)
end
