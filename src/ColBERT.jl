module ColBERT
using Clustering
using CSV
using Dates
using Flux
using .Iterators
using JLD2
using JSON
using LinearAlgebra
using Logging
using NeuralAttentionlib
using Random
using StatsBase
using TextEncodeBase
using Transformers
const HF = HuggingFace

# utils
include("utils.jl")

# config and other infra
include("infra/config.jl")
export ColBERTConfig

# models, document/query tokenizers
include("local_loading.jl")
include("modelling/tokenization/tokenizer_utils.jl")
include("modelling/tokenization/doc_tokenization.jl")
include("modelling/tokenization/query_tokenization.jl")
include("modelling/embedding_utils.jl")
include("modelling/checkpoint.jl")
export BaseColBERT, Checkpoint

# indexer
include("indexing/codecs/residual.jl")
include("indexing.jl")
include("indexing/collection_indexer.jl")
export Indexer, index

# searcher
include("search/ranking.jl")
include("searching.jl")
export Searcher, search

# loaders and savers
include("loaders.jl")
include("savers.jl")

end
