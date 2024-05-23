Base.@kwdef struct ColBERTConfig
    run_settings::RunSettings
    tokenizer_settings::TokenizerSettings
    resource_settings::ResourceSettings
    doc_settings::DocSettings
    query_settings::QuerySettings
    indexing_settings::IndexingSettings
    search_settings::SearchSettings
end
