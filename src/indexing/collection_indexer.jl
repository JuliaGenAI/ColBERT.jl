using .ColBERT: ColBERTConfig, CollectionEncoder, ResidualCodec

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

function CollectionIndexer(config::ColBERTConfig, encoder::CollectionEncoder, saver::IndexSaver)
    plan_path = joinpath(config.indexing_settings.index_path, "plan.json")
    metadata_path = joinpath(config.indexing_settings.index_path, "metadata.json")

    CollectionIndexer(
        config,
        encoder,
        saver,
        plan_path,
        0,              # num_chunks
        0.0,              # num_embeddings_est
        0,              # num_partitions
        0,              # num_sample_embs
        0.0,            # avg_doclen_est
        [],             # embeddings_offsets
        0,              # num_embeddings
        metadata_path,
    )
end

function _sample_pids(indexer::CollectionIndexer)
    num_passages = length(indexer.config.resource_settings.collection.data)
    typical_doclen = 120
    num_sampled_pids = 16 * sqrt(typical_doclen * num_passages)
    num_sampled_pids = Int(min(1 + floor(num_sampled_pids), num_passages))

    sampled_pids = Set(sample(1:num_passages, num_sampled_pids))
    @info "# of sampled PIDs = $(length(sampled_pids))"
    sampled_pids
end

function _sample_embeddings(indexer::CollectionIndexer, sampled_pids::Set{Int})
    # collect all passages with pids in sampled_pids
    collection = indexer.config.resource_settings.collection
    sorted_sampled_pids = sort(collect(sampled_pids))
    local_sample = collection.data[sorted_sampled_pids]

    local_sample_embs, local_sample_doclens = encode_passages(indexer.encoder, local_sample)
    indexer.num_sample_embs = size(local_sample_embs)[2]
    indexer.avg_doclen_est = length(local_sample_doclens) > 0 ? sum(local_sample_doclens) / length(local_sample_doclens) : 0

    sample_path = joinpath(indexer.config.indexing_settings.index_path, "sample.jld2")
    @info "avg_doclen_est = $(indexer.avg_doclen_est) \t length(local_sample) = $(length(local_sample))"
    @info "Saving sampled embeddings to $(sample_path)."
    save(sample_path, Dict("local_sample_embs" => local_sample_embs))

    indexer.avg_doclen_est
end

function _save_plan(indexer::CollectionIndexer)
    @info "Saving the index plan to $(indexer.plan_path)."
    # TODO: export the config as json as well
    open(indexer.plan_path, "w") do io
        JSON.print(io,
            Dict(
                "num_chunks" => indexer.num_chunks,
                "num_partitions" => indexer.num_partitions,
                "num_embeddings_est" => indexer.num_embeddings_est,
                "avg_doclen_est" => indexer.avg_doclen_est,
            ),
            4                                                               # indent
        )
    end
end

function setup(indexer::CollectionIndexer)
    collection = indexer.config.resource_settings.collection
    indexer.num_chunks = Int(ceil(length(collection.data) / get_chunksize(collection, indexer.config.run_settings.nranks)))

    # sample passages for training centroids later
    sampled_pids = _sample_pids(indexer)
    avg_doclen_est = _sample_embeddings(indexer, sampled_pids)

    # computing the number of partitions, i.e clusters
    num_passages = length(indexer.config.resource_settings.collection.data)
    indexer.num_embeddings_est = num_passages * avg_doclen_est
    indexer.num_partitions = Int(floor(2 ^ (floor(log2(16 * sqrt(indexer.num_embeddings_est))))))

    @info "Creating $(indexer.num_partitions) clusters."
    @info "Estimated $(indexer.num_embeddings_est) embeddings."

    _save_plan(indexer)
end

function _concatenate_and_split_sample(indexer::CollectionIndexer)
    # load the sample embeddings
    sample_path = joinpath(indexer.config.indexing_settings.index_path, "sample.jld2")
    sample = load(sample_path, "local_sample_embs")

    # randomly shuffle embeddings
    num_local_sample_embs = size(sample)[2]
    sample = sample[:, shuffle(1:num_local_sample_embs)]

    # split the sample to get a heldout set
    heldout_fraction = 0.05
    heldout_size = Int(floor(min(50000, heldout_fraction * num_local_sample_embs)))
    sample, sample_heldout = sample[:, 1:(num_local_sample_embs - heldout_size)], sample[:, num_local_sample_embs - heldout_size + 1:num_local_sample_embs]
    sample, sample_heldout
end

function _compute_avg_residuals(indexer::CollectionIndexer, centroids::Matrix{Float64}, heldout::Matrix{Float64})
    compressor = ResidualCodec(indexer.config, centroids, 0.0, Vector{Float64}(), Vector{Float64}()) 
    codes = compress_into_codes(compressor, heldout)             # get centroid codes
    heldout_reconstruct = compressor.centroids[:, codes]         # get corresponding centroids
    heldout_avg_residual = heldout - heldout_reconstruct         # compute the residual
    
    avg_residual = mean(abs.(heldout_avg_residual), dims = 2)    # for each dimension, take mean of absolute values of residuals
    
    # computing bucket weights and cutoffs
    num_options = 2 ^ indexer.config.indexing_settings.nbits
    quantiles = Vector(0:num_options - 1) / num_options
    bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[2:end], quantiles .+ (0.5 / num_options)  

    bucket_cutoffs = quantile(heldout_avg_residual, bucket_cutoffs_quantiles)
    bucket_weights = quantile(heldout_avg_residual, bucket_weights_quantiles)

    @info "Got bucket_cutoffs_quantiles = $(bucket_cutoffs_quantiles) and bucket_weights_quantiles = $(bucket_weights_quantiles)"
    @info "Got bucket_cutoffs = $(bucket_cutoffs) and bucket_weights = $(bucket_weights)"

    bucket_cutoffs, bucket_weights, mean(avg_residual)
end

function train(indexer::CollectionIndexer)
    sample, heldout = _concatenate_and_split_sample(indexer)
    centroids = kmeans(sample, indexer.num_partitions, maxiter = indexer.config.indexing_settings.kmeans_niters, display = :iter).centers
    bucket_cutoffs, bucket_weights, avg_residual = _compute_avg_residuals(indexer, centroids, heldout)
    @info "avg_residual = $(avg_residual)"

    codec = ResidualCodec(indexer.config, centroids, avg_residual, bucket_cutoffs, bucket_weights)
    indexer.saver.codec = codec
    save_codec(indexer.saver)
end

function index(indexer::CollectionIndexer)
    load_codec!(indexer.saver)                  # load the codec objects
    batches = enumerate_batches(indexer.config.resource_settings.collection, nranks = indexer.config.run_settings.nranks)
    for (chunk_idx, offset, passages) in batches
        # TODO: add functionality to not re-write chunks if they already exist! 
        # TODO: add multiprocessing to this step!
        embs, doclens = encode_passages(indexer.encoder, passages)
        @info "Saving chunk $(chunk_idx): \t $(length(passages)) passages and $(size(embs)[2]) embeddings. From offset #$(offset) onward." 
        save_chunk(indexer.saver, chunk_idx, offset, embs, doclens)
    end
end
