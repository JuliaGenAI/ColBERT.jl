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

function setup(indexer::CollectionIndexer)
    
end
