module ColBERT
using CSV
using Dates
using LinearAlgebra
using Logging
using NeuralAttentionlib
using StatsBase
using TextEncodeBase
using Transformers

# utils
include("utils/utils.jl")

# datasets
include("data/collection.jl")
include("data/queries.jl")
export Collection, Queries

# config and other infra
include("infra/settings.jl")
include("infra/config.jl")
export RunSettings, TokenizerSettings, ResourceSettings,
       DocSettings, QuerySettings, IndexingSettings,
       SearchSettings, ColBERTConfig

# models, document/query tokenizers
include("modelling/tokenization/doc_tokenization.jl")
include("modelling/checkpoint.jl")
export BaseColBERT, Checkpoint, DocTokenizer

# indexer
include("indexing.jl")
include("indexing/collection_encoder.jl")
include("indexing/index_saver.jl")
include("indexing/collection_indexer.jl")

export Indexer, CollectionIndexer

end
