using ColBERT
using CUDA
using Test
using Random

# set the global seed
Random.seed!(0)

# small toy example first
config = ColBERTConfig(
    use_gpu = true,
    checkpoint = "/home/codetalker7/models/colbertv2.0/",
    collection = "./sample_collection.tsv",
    index_path = "./sample_collection_index/",
    chunksize = 3
)

indexer = Indexer(config)
@time index(indexer)

# then big example
config = ColBERTConfig(
    use_gpu = true,
    collection = "./downloads/lotte/lifestyle/dev/collection.tsv",
    index_path = "./lotte_lifestyle_index/"
)

indexer = Indexer(config)
indexer.collection
@time index(indexer)
