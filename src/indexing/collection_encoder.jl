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

"""
    encode_passages(encoder::CollectionEncoder, passages::Vector{String})

Encode a list of passages using `encoder`. 

The given `passages` are run through the underlying BERT model and the linear layer to generate the embeddings, after doing relevant document-specific preprocessing. See [`docFromText`](@ref) for more details.

# Arguments

- `encoder`: The encoder used to encode the passages.
- `passages`: A list of strings representing the passages to be encoded.

# Returns

A tuple `embs, doclens` where:

- `embs::AbstractMatrix{Float32}`: The full embedding matrix. Of shape `(D, N)`, where `D` is the embedding dimension and `N` is the total number of embeddings across all the passages. 
- `doclens::AbstractVector{Int}`: A vector of document lengths for each passage, i.e the total number of attended tokens for each document passage. 
"""
function encode_passages(encoder::CollectionEncoder, passages::Vector{String})
    @info "Encoding $(length(passages)) passages."

    if length(passages) == 0
        error("The list of passages to encode is empty!")
    end

    embs, doclens = Vector{AbstractMatrix{Float32}}(), Vector{Int}()
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
