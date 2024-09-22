# loading the docs
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

# evals
using ColBERT, CUDA, Random, JSON;
using PromptingTools: distance_longest_common_subsequence
# CUDA.devices()
# device!(5)
Random.seed!(0)

## load the evaluation qa
eval_qa = JSON.parsefile("qa_evals.json")

## get the searcher
searcher = Searcher("./juliadocsindex/");

## for each qs, see if the context is returned
k = 5 
num_hits = 0
for query in eval_qa
    @time pids, _ = search(searcher, query["question"], k)
    if minimum(distance_longest_common_subsequence(query["context"], doc_passages[pids])) < 0.33
        num_hits += 1
    end
end
print("Number of hits: ", num_hits / length(eval_qa))
