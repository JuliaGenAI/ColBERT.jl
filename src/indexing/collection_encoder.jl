using ..ColBERT: ColBERTConfig

"""
    CollectionEncoder(config::ColBERTConfig, checkpoint::Checkpoint)

Structure to represent an encoder used to encode document passages to their corresponding embeddings.

# Arguments

- `config`: The underlying [`ColBERTConfig`](@ref). 
- `checkpoint`: The [`Checkpoint`](@ref) used by the model.

# Returns

A [`CollectionEncoder`](@ref).

"""
struct CollectionEncoder
    config::ColBERTConfig
    checkpoint::Checkpoint
end

function encode_passages(encoder::CollectionEncoder, passages::Vector{String})
    @info "Encoding $(length(passages)) passages."

    if length(passages) == 0
        error("The list of passages to encode is empty!")
    end

    embs, doclens = Vector{Matrix{Float64}}(), Vector{Int}()
    # batching here to avoid storing intermediate embeddings on GPU
    # batching also occurs inside docFromText to do batch packing optimizations
    for passages_batch in batch(passages, encoder.config.indexing_settings.index_bsize * 50)
        embs_, doclens_ = docFromText(encoder.checkpoint, passages_batch, encoder.config.indexing_settings.index_bsize)
        push!(embs, embs_)
        append!(doclens, vec(doclens_))
    end
    embs = cat(embs..., dims=2)
    embs, doclens
end
