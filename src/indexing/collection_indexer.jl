using ..ColBERT: ColBERTConfig, CollectionEncoder, IndexSaver

struct CollectionIndexer
    config::ColBERTConfig
    encoder::CollectionEncoder
    saver::IndexSaver
end
