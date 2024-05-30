module ColBERT
using CSV
using Logging
using Transformers

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

# models
include("modelling/checkpoint.jl")
export BaseColBERT, Checkpoint

# indexer
include("indexing.jl")
export Indexer

end
