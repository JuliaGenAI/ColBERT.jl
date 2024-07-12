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

# Returns

A [`ColBERTConfig`](@ref) object.

# Examples

The relevant files for this example can be found in the `examples/` folder of the project root.

```julia-repl
julia> dataroot = "downloads/lotte"

julia> dataset = "lifestyle"

julia> datasplit = "dev"

julia> path = joinpath(dataroot, dataset, datasplit, "short_collection.tsv")

julia> collection = Collection(path)

julia> length(collection.data)

julia> nbits = 2   # encode each dimension with 2 bits

julia> doc_maxlen = 300   # truncate passages at 300 tokens

julia> checkpoint = "colbert-ir/colbertv2.0"                       # the HF checkpoint

julia> index_root = "experiments/notebook/indexes"

julia> index_name = "short_\$(dataset).\$(datasplit).\$(nbits)bits"

julia> index_path = joinpath(index_root, index_name)

julia> config = ColBERTConfig(
            RunSettings(
                experiment="notebook",
            ),
            TokenizerSettings(),
            ResourceSettings(
                checkpoint=checkpoint,
                collection=collection,
                index_name=index_name,
            ),
            DocSettings(
                doc_maxlen=doc_maxlen,
            ),
            QuerySettings(),
            IndexingSettings(
                index_path=index_path,
                index_bsize=3,
                nbits=nbits,
                kmeans_niters=20,
            ),
            SearchSettings(),
        );
```
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
