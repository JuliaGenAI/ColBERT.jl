using .ColBERT: ColBERTConfig

struct Indexer
    # index_path::String            we can just reuse the path from the config?
    # checkpoint::String            can also infer this from config
    config::ColBERTConfig
end

function index(indexer::Indexer)
    if isdir(indexer.config.index_path)
        @info "Index at $(indexer.config.index_path) already exists!"
    end

    # TODO: finish this!
end
