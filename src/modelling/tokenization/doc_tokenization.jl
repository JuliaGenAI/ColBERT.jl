using ...ColBERT: ColBERTConfig

"""
    DocTokenizer(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, config::ColBERTConfig)

Construct a `DocTokenizer` from a given tokenizer and configuration. The resulting structure supports functions to perform CoLBERT-style document operations on document texts.  

# Arguments

- `tokenizer`: A tokenizer that has been trained on the BERT vocabulary. Fetched from HuggingFace.
- `config`: The underlying [`ColBERTConfig`](@ref).

# Returns

A `DocTokenizer` object.
"""
struct DocTokenizer
    D_marker_token_id::Int
    config::ColBERTConfig
end

function DocTokenizer(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, config::ColBERTConfig)
    D_marker_token_id = TextEncodeBase.lookup(tokenizer.vocab, config.tokenizer_settings.doc_token_id)
    DocTokenizer(D_marker_token_id, config)
end

"""
    tensorize(doc_tokenizer::DocTokenizer, tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, batch_text::Vector{String}, bsize::Union{Missing, Int})

Convert a collection of documents to tensors in the ColBERT format. 

This function adds the document marker token at the beginning of each document and then converts the text data into integer IDs and masks using the `tokenizer`. The returned objects are determined by the `bsize` argument. More specifically:

- If `bsize` is missing, then a tuple `integer_ids, integer_mask` is returned, where `integer_ids` is an `Array` of token IDs for the modified documents, and `integer_mask` is an `Array` of attention masks for each document.
- If `bsize` is not missing, then more optimizing operations are performed on the documents. First, the arrays of token IDs and attention masks are sorted by document lengths (this is for more efficient use of GPUs on the batches; see [`_sort_by_length`](@ref)), and a list `reverse_indices` is computed, which remembers the original order of the documents (to reorder them later). The arrays of token IDs and attention masks are then batched into batches of size `bsize` (see [`_split_into_batches`](@ref)). Finally, the batches along with the list of `reverse_indices` are returned.

# Arguments

- `doc_tokenizer`: An instance of the `DocTokenizer` type. This object contains information about the document marker token ID.
- `tokenizer`: The tokenizer which is used to convert text data into integer IDs.
- `batch_text`: A document texts that will be converted into tensors of token IDs.
- `bsize`: The size of the batches to split the `batch_text` into. Can also be `missing`. 

# Returns

If `bsize` is `missing`, then a tuple is returned, which contains: 

- `integer_ids`: An `Array` of integer IDs representing the token IDs of the documents in the input collection.
- `integer_mask`: An `Array` of bits representing the attention mask for each document. 

If `bsize` is not `missing`, then a tuple containing the following is returned:

- `batches::`: Batches of sorted integer IDs and masks. 
- `reverse_indices::Vector{Int}`: A vector containing the indices of the documents in their original order.
"""
function tensorize(doc_tokenizer::DocTokenizer, tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, batch_text::Vector{String}, bsize::Union{Missing, Int})
    # placeholder for [D] marker token
    batch_text = [". " * doc for doc in batch_text]

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
    _sort_by_length(integer_ids::AbstractMatrix, integer_mask::AbstractMatrix, bsize::Int)

Sort sentences by number of attended tokens, if the number of sentences is larger than `bsize`. 

# Arguments

- `integer_ids`: The token IDs of documents to be sorted.
- `integer_mask`: The attention masks of the documents to be sorted (attention masks are just bits).
- `bsize`: The size of batches to be considered.

# Returns

Depending upon `bsize`, the following are returned:

- If the number of documents (second dimension of `integer_ids`) is atmost `bsize`, then the `integer_ids` and `integer_mask` are returned unchanged. 
- If the number of documents is larger than `bsize`, then the passages are first sorted by the number of attended tokens (figured out from the `integer_mask`), and then the sorted arrays `integer_ids`, `integer_mask` are returned, along with a list of `reverse_indices`, i.e a mapping from the documents to their indices in the original order.
"""
function _sort_by_length(integer_ids::AbstractMatrix, integer_mask::AbstractMatrix, bsize::Int)
    batch_size = size(integer_ids)[2]
    if batch_size <= bsize
        # if the number of passages fits the batch size, do nothing
        integer_ids, integer_mask, Vector(1:batch_size)
    end

    lengths = vec(sum(integer_mask; dims = 1))              # number of attended tokens in each passage
    indices = sortperm(lengths)                             # get the indices which will sort lengths
    reverse_indices = sortperm(indices)                     # invert the indices list

    integer_ids[:, indices], integer_mask[:, indices], reverse_indices
end

"""
    _split_into_batches(integer_ids::AbstractArray, integer_mask::AbstractMatrix, bsize::Int)

Split the given `integer_ids` and `integer_mask` into batches of size `bsize`.

# Arguments

- `integer_ids`: The array of token IDs to batch. 
- `integer_mask`: The array of attention masks to batch.

# Returns

Batches of token IDs and attention masks, with each batch having size `bsize` (with the possibility of the last batch being smaller).
"""
function _split_into_batches(integer_ids::AbstractArray, integer_mask::AbstractMatrix, bsize::Int)
    batch_size = size(integer_ids)[2]
    batches = Vector{Tuple{AbstractArray, AbstractMatrix}}()
    for offset in 1:bsize:batch_size
        push!(batches, (integer_ids[:, offset:min(batch_size, offset + bsize - 1)], integer_mask[:, offset:min(batch_size, offset + bsize - 1)]))
    end
    batches
end
