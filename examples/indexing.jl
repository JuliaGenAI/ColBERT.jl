using ColBERT
using CUDA
using Test
using Random

# set the global seed
Random.seed!(0)

config = ColBERTConfig(
    use_gpu = true,
    collection = "./short_collection",
    doc_maxlen = 300,
    index_path = "./short_collection_index/",
    chunksize = 3
)

indexer = Indexer(config)
index(indexer)
