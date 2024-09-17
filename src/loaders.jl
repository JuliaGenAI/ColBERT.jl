"""
    load_codec(index_path::String)

Load compression/decompression information from the index path.

# Arguments

  - `index_path`: The path of the index.
"""
function load_codec(index_path::String)
    centroids_path = joinpath(index_path, "centroids.jld2")
    avg_residual_path = joinpath(index_path, "avg_residual.jld2")
    bucket_cutoffs_path = joinpath(index_path, "bucket_cutoffs.jld2")
    bucket_weights_path = joinpath(index_path, "bucket_weights.jld2")
    missing_files = findall(!isfile,
        [centroids_path, avg_residual_path,
            bucket_cutoffs_path, bucket_weights_path])
    isempty(missing_files) || error("$(missing_files) are missing!")
    @info "Loading codec from $(centroids_path), $(avg_residual_path), " *
          "$(bucket_cutoffs_path) and $(bucket_weights_path)."

    centroids = JLD2.load_object(centroids_path)
    avg_residual = JLD2.load_object(avg_residual_path)
    bucket_cutoffs = JLD2.load_object(bucket_cutoffs_path)
    bucket_weights = JLD2.load_object(bucket_weights_path)

    @assert centroids isa Matrix{Float32}
    @assert avg_residual isa Float32
    @assert bucket_cutoffs isa Vector{Float32}
    @assert bucket_weights isa Vector{Float32}

    Dict(
        "centroids" => centroids,
        "avg_residual" => avg_residual,
        "bucket_cutoffs" => bucket_cutoffs,
        "bucket_weights" => bucket_weights
    )
end

"""
    load_config(index_path::String)

Load a [`ColBERTConfig`](@ref) from disk.

# Arguments

  - `index_path`: The path of the directory where the config resides.

# Examples

```julia-repl
julia> using ColBERT;

julia> config = ColBERTConfig(
           use_gpu = true,
           collection = "/home/codetalker7/documents",
           index_path = "./local_index"
       );

julia> ColBERT.save(config);

julia> ColBERT.load_config("./local_index")
ColBERTConfig(true, 0, 1, "[unused0]", "[unused1]", "[Q]", "[D]", "colbert-ir/colbertv2.0", "/home/codetalker7/documents", 128, 220, true, 32, false, "./local_index", 64, 2, 20, 2, 8192)
```
"""
function load_config(index_path::String)
    config_dict = JSON.parsefile(joinpath(index_path, "config.json"))
    key_vals = collect(zip(Symbol.(keys(config_dict)), values(config_dict)))
    eval(:(ColBERTConfig($([Expr(:kw, :($key), :($val))
                            for (key, val) in key_vals]...))))
end

function load_doclens(index_path::String)
    isfile(joinpath(index_path, "plan.json")) || error("plan.json not found!")
    plan_metadata = JSON.parsefile(joinpath(index_path, "plan.json"))
    doclens = Vector{Int}()
    for chunk_idx in 1:plan_metadata["num_chunks"]
        doclens_file = joinpath(index_path, "doclens.$(chunk_idx).jld2")
        chunk_doclens = JLD2.load_object(doclens_file)
        append!(doclens, chunk_doclens)
    end
    @assert(isequal(sum(doclens), plan_metadata["num_embeddings"]),
        "sum(doclens): $(sum(doclens)), num_embeddings: "*
        "$(plan_metadata["num_embeddings"])")
    doclens
end

function load_compressed_embs(index_path::String)
    config = load_config(index_path)
    plan_metadata = JSON.parsefile(joinpath(index_path, "plan.json"))
    @assert config.dim % 8==0 "dim: $(config.dim)"

    codes = zeros(UInt32, plan_metadata["num_embeddings"])
    residuals = zeros(UInt8, div(config.dim, 8) * config.nbits,
        plan_metadata["num_embeddings"])
    codes_offset = 1
    for chunk_idx in 1:plan_metadata["num_chunks"]
        chunk_codes = JLD2.load_object(joinpath(
            index_path, "$(chunk_idx).codes.jld2"))
        chunk_residuals = JLD2.load_object(joinpath(
            index_path, "$(chunk_idx).residuals.jld2"))

        codes_endpos = codes_offset + length(chunk_codes) - 1
        codes[codes_offset:codes_endpos] = chunk_codes
        residuals[:, codes_offset:codes_endpos] = chunk_residuals

        codes_offset = codes_offset + length(chunk_codes)
    end
    codes, residuals
end

function load_chunk_metadata_property(index_path::String, property::String)
    plan_metadata = JSON.parsefile(joinpath(index_path, "plan.json"))
    plan_metadata["num_chunks"] > 0 || return []
    vector = nothing
    for chunk_idx in 1:plan_metadata["num_chunks"]
        chunk_metadata = JSON.parsefile(joinpath(
            index_path, "$(chunk_idx).metadata.json"))
        if isnothing(vector)
            vector = [chunk_metadata[property]]
        else
            push!(vector, chunk_metadata[property])
        end
    end
    vector
end

function load_codes(index_path::String)
    plan_metadata = JSON.parsefile(joinpath(index_path, "plan.json"))
    codes = Vector{UInt32}()
    for chunk_idx in 1:(plan_metadata["num_chunks"])
        chunk_codes = JLD2.load_object(joinpath(
            index_path, "$(chunk_idx).codes.jld2"))
        append!(codes, chunk_codes)
    end
    codes
end
