struct Indexer
    config::ColBERTConfig
end

function index(indexer::Indexer)
    index_path = indexer.config.indexing_settings.index_path 
    if isdir(index_path)
        @info "Index at $(index_path) already exists! Skipping indexing."
        return
    end

    config = indexer.config
    checkpoint = config.resource_settings.checkpoint

    # loading the models
    @info "Loading ColBERT layers from HuggingFace."
    base_colbert = BaseColBERT(checkpoint, config)
    checkPoint = Checkpoint(base_colbert, DocTokenizer(base_colbert.tokenizer, config), QueryTokenizer(base_colbert.tokenizer, config), config)

    # creating the encoder, saver and indexer
    encoder = CollectionEncoder(config, checkPoint)
    saver = IndexSaver(config=config)
    collection_indexer = CollectionIndexer(config, encoder, saver)

    # building the index
    @info "Building the index."
    setup(collection_indexer)
    train(collection_indexer)
    index(collection_indexer)
    finalize(collection_indexer)
end
