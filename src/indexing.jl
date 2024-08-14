struct Indexer
    config::ColBERTConfig
    checkpoint::Checkpoint
    collection::Vector{String}
end

function Indexer(config::ColBERTConfig)
    base_colbert = BaseColBERT(config)
    checkpoint = Checkpoint(base_colbert, config)
    collection = readlines(config.collection)

    @info "Loaded ColBERT layers from the $(checkpoint) HuggingFace checkpoint."
    @info "Loaded $(length(collection)) documents from $(config.collection)."

    Indexer(config, checkpoint, collection)
end

# function index(indexer::Indexer)
#     if isdir(indexer.config.index_path)
#         @info "Index at $(indexer.config.index_path) already exists! Skipping indexing."
#         return
#     end
#
#     config = indexer.config
#     checkpoint = config.checkpoint
#
#     # loading the models
#     @info "Loading ColBERT layers from HuggingFace."
#     base_colbert = BaseColBERT(checkpoint, config)
#     checkPoint = Checkpoint(base_colbert, DocTokenizer(base_colbert.tokenizer, config),
#         QueryTokenizer(base_colbert.tokenizer, config), config)
#
#     # creating the encoder, saver and indexer
#     encoder = CollectionEncoder(config, checkPoint)
#     saver = IndexSaver(config = config)
#     collection_indexer = CollectionIndexer(config, encoder, saver)
#
#     # building the index
#     @info "Building the index."
#     setup(collection_indexer)
#     train(collection_indexer)
#     index(collection_indexer)
#     finalize(collection_indexer)
# end
