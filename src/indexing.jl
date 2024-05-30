using .ColBERT: ColBERTConfig

struct Indexer
    # index_path::String            we can just reuse the path from the config?
    checkpoint::String
    config::ColBERTConfig
end


