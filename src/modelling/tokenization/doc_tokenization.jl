using ...ColBERT: ColBERTConfig

struct DocTokenizer
    D_marker_token_id::Int
    config::ColBERTConfig
end

function DocTokenizer(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, config::ColBERTConfig)
    D_marker_token_id = TextEncodeBase.lookup(tokenizer.vocab, config.tokenizer_settings.doc_token_id)
    DocTokenizer(D_marker_token_id, config)
end

function tensorize(doc_tokenizer::DocTokenizer, tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, batch_text::Vector{String}, bsize::Union{Missing, Int})
    # placeholder for [D] marker token
    batch_text = [". " * doc for doc in batch_text]
    vocabsize = length(tokenizer.vocab.list)

    # getting the integer ids and masks
    encoded_text = Transformers.TextEncoders.encode(tokenizer, batch_text)
    ids, mask = encoded_text.token, encoded_text.attention_mask
    integer_ids = reinterpret(Int32, ids)
    integer_mask = NeuralAttentionlib.getmask(mask, ids)[1, :, :]

    # adding the [D] marker token ID
    integer_ids[2, :] .= doc_tokenizer.D_marker_token_id

    if ismissing(bsize)
        integer_ids, integer_mask
    else
        # we sort passages by length to do batch packing for more efficient use of the GPU
        integer_ids, integer_mask, reverse_indices = _sort_by_length(integer_ids, integer_mask, bsize)
        batches = _split_into_batches(integer_ids, integer_mask, bsize)

        batches, reverse_indices
    end
end

"""
    _sort_by_length(ids::AbstractMatrix, mask::AbstractMatrix, bsize::Int)

Sort sentences by number of attended tokens, if the number of sentences is larger than bsize. If the number of passages (first dimension of `ids`) is atmost
than `bsize`, the `ids`, `mask`, and a list `Vector(1:size(ids)[1])` is returned as a three-tuple. Otherwise,
the passages are first sorted by the number of attended tokens (figured out from `mask`), and then the the sorted arrays
`ids` and `mask` are returned, along with a reversed list of indices, i.e a mapping from passages to their indice in the sorted list.
"""
function _sort_by_length(integer_ids::AbstractMatrix, integer_mask::AbstractMatrix, bsize::Int)
    batch_size = size(integer_ids)[2]
    if batch_size <= bsize
        # if the number of passages fits the batch size, do nothing
        integer_ids, integer_mask, Vector(1:batch_size)
    end

    lengths = vec(sum(integer_mask; dims = 1))              # number of attended tokens in each passage
    indices = sortperm(lengths)                     # get the indices which will sort lengths
    reverse_indices = sortperm(indices)             # invert the indices list

    integer_ids[:, indices], integer_mask[:, indices], reverse_indices
end

"""
    _split_into_batches(integer_ids::AbstractArray, integer_mask::AbstractMatrix, bsize::Int)::Vector{Tuple{AbstractArray, AbstractMatrix, Int}}

Split the given `integer_ids` and `integer_mask` into batches of size `bsize`.
"""
function _split_into_batches(integer_ids::AbstractArray, integer_mask::AbstractMatrix, bsize::Int)
    batch_size = size(integer_ids)[2]
    batches = Vector{Tuple{AbstractArray, AbstractMatrix}}()
    for offset in 1:bsize:batch_size
        push!(batches, (integer_ids[:, offset:min(batch_size, offset + bsize - 1)], integer_mask[:, offset:min(batch_size, offset + bsize - 1)]))
    end
    batches
end

# tokenizer = base_colbert.tokenizer
# batch_text = [
#     "hello world",
#     "thank you!",
#     "a",
#     "this is some longer text, so length should be longer",
# ]
# bsize = 2
