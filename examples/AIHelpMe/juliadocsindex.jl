# load the sources and chunks
using HDF5

doc_passages = String[]
doc_sources = String[]

for file in ["genie__v20240818__textembedding3large-1024-Bool__v1.0.hdf5",
    "JuliaData-text-embedding-3-large-1-Bool__v1.0.hdf5",
    "julialang__v20240819__textembedding3large-1024-Bool__v1.0.hdf5",
    "Makie-text-embedding-3-large-1-Bool__v1.0.hdf5",
    "pack.hdf5", "Plots-text-embedding-3-large-1-Bool__v1.0.hdf5",
    "sciml__v20240716__textembedding3large-1024-Bool__v1.0.hdf5",
    "tidier__v20240716__textembedding3large-1024-Bool__v1.0.hdf5"]
    fid = h5open(file, "r")
    chunks, sources = fid["chunks"], fid["sources"]
    append!(doc_passages, read(chunks)) 
    append!(doc_sources, read(sources)) 
end

## run the small toy example at the bottom of the file first to compile everything!
# index it
using ColBERT, CUDA, Random;
# CUDA.devices()
# device!(5)
Random.seed!(0)
config = ColBERTConfig(
    use_gpu = true,
    checkpoint = "/home/codetalker7/models/colbertv2.0",        # local path to the colbert checkpoint
    collection = doc_passages,                          # local path to the collection
    doc_maxlen = 300,                                           # max length beyond which docs are truncated
    index_path = "./juliadocsindex/",                       # local directory to save the index in
    chunksize = 10000    # number of docs to store in a chunk
);
indexer = Indexer(config)
@time index(indexer)

## trying out some queries
searcher = Searcher("./juliadocsindex/");
query = "What is the effect of setting the `compress` parameter to `true` in the `categorical` function and what caution must be taken when using it?"
@time pids, scores = search(searcher, query, 10)
for doc in doc_passages[pids]
    print(doc, "\n\n")
end

query = "How can you construct an uninitialized CategoricalArray with specific levels and dimensions?"
@time pids, scores = search(searcher, query, 10)
for (doc, src) in zip(doc_passages[pids], doc_sources[pids])
    print("doc: \n", doc, "\n")
    print("src: \n", src, "\n\n")
end

query = "How can one determine or change the order of levels in a `CategoricalArray`?"
@time pids, scores = search(searcher, query, 10)
for (doc, src) in zip(doc_passages[pids], doc_sources[pids])
    print("doc: \n", doc, "\n")
    print("src: \n", src, "\n\n")
end

########### TOY example for compilation
Random.seed!(0)
config = ColBERTConfig(
    use_gpu = true,
    checkpoint = "/home/codetalker7/models/colbertv2.0/",
    collection = "../sample_collection.tsv",
    index_path = "../sample_collection_index/",
    chunksize = 3
)

indexer = Indexer(config)
@time index(indexer)
