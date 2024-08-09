"""
    RunSettings([root, experiment, index_root, name, rank, nranks])

Structure holding all the settings necessary to describe the run environment.

# Arguments

  - `root`: The root directory for the run. Default is an `"experiments"` folder in the current working directory.
  - `experiment`: The name of the run. Default is `"default"`.
  - `index_root`: The root directory for storing index. For now, there is no need to specify this as it is determined by the indexing component.
  - `name`: The name of the run. Default is the current date and time.
  - `use_gpu`: Whether to use a GPU or not. Default is `false`.
  - `rank`: The index of the running GPU. Default is `0`. For now, the package only allows this to be `0`.
  - `nranks`: The number of GPUs used in the run. Default is `1`. For now, the package only supports one GPU.

# Returns

A `RunSettings` object.
"""
Base.@kwdef struct RunSettings
    root::String = joinpath(pwd(), "experiments")
    experiment::String = "default"
    index_root::Union{Nothing, String} = nothing
    name::String = Dates.format(now(), "yyyy/mm/dd/HH.MM.SS")
    use_gpu::Bool = false
    rank::Int = 0
    nranks::Int = 1
end

"""
    TokenizerSettings([query_token_id, doc_token_id, query_token, doc_token])

Structure to represent settings for the tokenization of queries and documents.

# Arguments

  - `query_token_id`: Unique identifier for query tokens (defaults to `[unused0]`).
  - `doc_token_id`: Unique identifier for document tokens (defaults to `[unused1]`).
  - `query_token`: Token used to represent a query token (defaults to `[Q]`).
  - `doc_token`: Token used to represent a document token (defaults to `[D]`).

# Returns

A `TokenizerSettings` object.
"""
Base.@kwdef struct TokenizerSettings
    query_token_id::String = "[unused0]"
    doc_token_id::String = "[unused1]"
    query_token::String = "[Q]"
    doc_token::String = "[D]"
end

"""
    ResourceSettings([checkpoint, collection, queries, index_name])

Structure to represent resource settings.

# Arguments

  - `checkpoint`: The path to the HuggingFace checkpoint of the underlying ColBERT model.
  - `collection`: The underlying collection of documents
  - `queries`: The underlying collection of queries.
  - `index_name`: The name of the index.

# Returns

A `ResourceSettings` object.
"""
Base.@kwdef struct ResourceSettings
    checkpoint::Union{Nothing, String} = nothing
    collection::Union{Nothing, Collection} = nothing
    queries::Union{Nothing, String} = nothing
    index_name::Union{Nothing, String} = nothing
end

"""
    DocSettings([dim, doc_maxlen, mask_punctuation])

Structure that defines the settings used for generating document embeddings.

# Arguments

  - `dim`: The dimension of the document embedding space. Default is 128.
  - `doc_maxlen`: The maximum length of a document before it is trimmed to fit. Default is 220.
  - `mask_punctuation`: Whether or not to mask punctuation characters tokens in the document. Default is true.

# Returns

A `DocSettings` object.
"""
Base.@kwdef struct DocSettings
    dim::Int = 128
    doc_maxlen::Int = 220
    mask_punctuation::Bool = true
end

"""
    QuerySettings([query_maxlen, attend_to_mask_tokens, interaction])

A structure representing the query settings used by the ColBERT model.

# Arguments

  - `query_maxlen`: The maximum length of queries after which they are trimmed.
  - `attend_to_mask_tokens`: Whether or not to attend to mask tokens in the query. Default value is false.
  - `interaction`: The type of interaction used to compute the scores for the queries. Default value is "colbert".

# Returns

A `QuerySettings` object.
"""
Base.@kwdef struct QuerySettings
    query_maxlen::Int = 32
    attend_to_mask_tokens::Bool = false
    interaction::String = "colbert"
end

"""
    IndexingSettings([index_path, index_bsize, nbits, kmeans_niters])

Structure containing settings for indexing.

# Arguments

  - `index_path`: Path to save the index files.
  - `index_bsize::Int`: Batch size used for some parts of indexing.
  - `nbits::Int`: Number of bits used to compress residuals.
  - `kmeans_niters::Int`: Number of iterations used for k-means clustering.

# Returns

An `IndexingSettings` object.
"""
Base.@kwdef struct IndexingSettings
    index_path::Union{Nothing, String} = nothing
    index_bsize::Int = 64
    nbits::Int = 1
    kmeans_niters = 4
end

Base.@kwdef struct SearchSettings
    nprobe::Int = 2
    ncandidates::Int = 8192
end
