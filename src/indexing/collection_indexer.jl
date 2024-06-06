using ..ColBERT: ColBERTConfig, CollectionEncoder, IndexSaver

mutable struct CollectionIndexer
    config::ColBERTConfig
    encoder::CollectionEncoder
    saver::IndexSaver
    plan_path::String
    num_chunks::Int
    num_embeddings_est::Int
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
        0,              # num_embeddings_est
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

    sampled_pids = sample(1:num_passages, num_sampled_pids)
    @info "# of sampled PIDs = $(length(sampled_pids))"
    Set(sampled_pids)
end

function _sample_embeddings(indexer::CollectionIndexer, sampled_pids::Set{Int})
    # collect all passages with pids in sampled_pids
    collection = indexer.config.resource_settings.collection
    sorted_sampled_pids = sort(collect(sampled_pids))
    local_sample = collection.data[sorted_sampled_pids]

    local_sample_embs, local_sample_doclens = encode_passages(indexer.encoder, local_sample)
    indexer.num_sample_embs = size(local_sample_embs)[2]
    indexer.avg_doclen_est = length(local_sample_doclens) > 0 ? sum(local_sample_doclens) / length(local_sample_doclens) : 0

    @info "avg_doclen_est = $(indexer.avg_doclen_est) \t length(local_sample) = $(length(local_sample))"
    save(joinpath(indexer.config.indexing_settings.index_path, "sample.jld2"), Dict("local_sample_embs" => local_sample_embs))

    avg_doclen_est
end

function setup(indexer::CollectionIndexer)
    collection = indexer.config.resource_settings.collection
    indexer.num_chunks = Int(ceil(length(collection.data) / get_chunksize(collection, indexer.config.run_settings.nranks)))

    # sample passages for training centroids later
    # TODO: complete this!
    sampled_pids = _sample_pids(indexer)
    
end
