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
    checkPoint = Checkpoint(base_colbert, DocTokenizer(base_colbert.tokenizer, config), QueryTokenizer(base_colbert.tokenizer, config), config)

    Searcher(config, checkPoint, IndexScorer())
end

function encode_query(searcher::Searcher, query::String)
    queries = [query]
    bsize = 128
    Q = queryFromText(searcher.checkpoint, queries, bsize)
    Q
end

function search(searcher::Searcher, query::String, k::Int)
    dense_search(encode_query(query), k) 
end

function dense_search(searcher::Searcher, Q::Matrix{Float64}, k::Int)
    @info "This will implement dense search."
end


