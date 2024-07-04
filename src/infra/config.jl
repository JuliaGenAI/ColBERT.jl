"""
    ColBERTConfig(run_settings::RunSettings, tokenizer_settings::TokenizerSettings, resource_settings::ResourceSettings, doc_settings::DocSettings, query_settings::QuerySettings, indexing_settings::IndexingSettings, search_settings::SearchSettings)

Structure containing config for running and training various components.

# Arguments

- `run_settings`: Sets the [`RunSettings`](@ref).
- `tokenizer_settings`: Sets the [`TokenizerSettings`](@ref).
- `resource_settings`: Sets the [`ResourceSettings`](@ref). 
- `doc_settings`: Sets the [`DocSettings`](@ref). 
- `query_settings`: Sets the [`QuerySettings`](@ref).
- `indexing_settings`: Sets the [`IndexingSettings`](@ref). 
- `search_settings`: Sets the [`SearchSettings`](@ref).
"""
Base.@kwdef struct ColBERTConfig
    run_settings::RunSettings
    tokenizer_settings::TokenizerSettings
    resource_settings::ResourceSettings
    doc_settings::DocSettings
    query_settings::QuerySettings
    indexing_settings::IndexingSettings
    search_settings::SearchSettings
end
