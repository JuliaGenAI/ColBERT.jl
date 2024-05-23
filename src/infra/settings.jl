Base.@kwdef struct RunSettings
    root::String = joinpath(pwd(), "experiments")
    experiment::String = "default"
    index_root::Union{Nothing, String} = nothing
    name::String = Dates.format(now(), "yyyy/mm/dd/HH.MM.SS")
    rank::Int = 0
    nranks::Int = 1
end

Base.@kwdef struct TokenizerSettings
    query_token_id::String = "[unused0]"
    doc_token_id::String = "[unused1]"
    query_token::String = "[Q]"
    doc_token::String = "[D]"
end

Base.@kwdef struct ResourceSettings
    checkpoint::Union{Nothing, String} = nothing
    collection::Union{Nothing, String} = nothing
    queries::Union{Nothing, String} = nothing
    index_name::Union{Nothing, String} = nothing
end

Base.@kwdef struct DocSettings
    dim::Int = 128
    doc_maxlen::Int = 220
    mask_punctuation::Bool = true
end

Base.@kwdef struct QuerySettings
    query_maxlen::Int = 32
    attend_to_mask_tokens::Bool = false
    interaction::String = "colbert"
end

Base.@kwdef struct IndexingSettings
    index_path::Union{Nothing, String} = nothing
    index_bsize::Int = 64
    nbits::Int = 1
    kmeans_niters = 4
end

Base.@kwdef struct SearchSettings
    ncells::Union{Nothing, Int} = nothing
    centroid_score_threshold::Union{Nothing, Float64} = nothing
    ndocs::Union{Nothing, Int} = nothing
end
