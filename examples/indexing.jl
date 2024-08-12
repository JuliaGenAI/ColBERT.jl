using ColBERT
using CUDA
using Test
using Random

# set the global seed
Random.seed!(0)

config = ColBERTConfig(
    use_gpu = true,
    collection = "./cityofaustin",
    doc_maxlen = 300,
    index_path = "./cityofaustin_index/",
    chunksize = 500,
)

indexer = Indexer(config)
checkpoint = indexer.checkpoint
collection = indexer.collection

@time ColBERT.setup(config, checkpoint, collection)
@time ColBERT.train(config)
@time ColBERT.index(config, checkpoint, collection)
@time ColBERT.finalize(config.index_path)
