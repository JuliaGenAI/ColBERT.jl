using .ColBERT: Checkpoint, ColBERTConfig, Collection, IndexScorer

struct Searcher
    config::ColBERTConfig
    checkpoint::Checkpoint
    ranker::IndexScorer
end

function Searcher(config::ColBERTConfig)
    index_path = config.indexing_settings.index_path
    if !isdir(index_path) 
        error("Index at $(index_path) does not exist! Please build the index first and try again.")
    end

    # loading the model and saving it to prevent multiple loads
    @info "Loading ColBERT layers from HuggingFace."
    base_colbert = BaseColBERT(config.resource_settings.checkpoint, config)
    checkPoint = Checkpoint(base_colbert, DocTokenizer(base_colbert.tokenizer, config), config)

    Searcher(config, checkPoint, IndexScorer())
end
