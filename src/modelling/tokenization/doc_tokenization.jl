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

- `integer_ids`: An `Array` of integer IDs representing the token IDs of the documents in the input collection. It has shape `(L, N)`, where `L` is the length of the largest document in `batch_text` (i.e the document with the largest number of tokens), and `N` is the number of documents in the batch.
- `integer_mask`: An `Array` of bits representing the attention mask for each document. It has shape `(L, N)`, the same as `integer_ids`. 

If `bsize` is not `missing`, then a tuple containing the following is returned:

- `batches`: A `Vector` of tuples of arrays of token IDs and masks, sorted in the order of document lengths. Each array in each tuple has shape `(L, N)`, where `L` is the length of the largest document in `batch_text`, and `N` is the number of documents in the batch being considered.
- `reverse_indices`: A `Vector` containing the indices of the documents in their original order.

# Examples

```julia-repl
julia> base_colbert = BaseColBERT("colbert-ir/colbertv2.0", config); 

julia> tokenizer = base_colbert.tokenizer;

julia> doc_tokenizer = DocTokenizer(tokenizer, config);

julia> batch_text = [
    "hello world",
    "thank you!",
    "a",
    "this is some longer text, so length should be longer",
];

julia> integer_ids, integer_mask = tensorize(doc_tokenizer, tokenizer, batch_text, missing);    # no batching

julia> integer_ids
14×4 reinterpret(Int32, ::Matrix{PrimitiveOneHot.OneHot{0x0000773a}}):
  102   102   102   102
    3     3     3     3
 7593  4068  1038  2024
 2089  2018   103  2004
  103  1000     1  2071
    1   103     1  2937
    1     1     1  3794
    1     1     1  1011
    1     1     1  2062
    1     1     1  3092
    1     1     1  2324
    1     1     1  2023
    1     1     1  2937
    1     1     1   103

julia> integer_mask
14×4 Matrix{Bool}:
 1  1  1  1
 1  1  1  1
 1  1  1  1
 1  1  1  1
 1  1  0  1
 0  1  0  1
 0  0  0  1
 0  0  0  1
 0  0  0  1
 0  0  0  1
 0  0  0  1
 0  0  0  1
 0  0  0  1
 0  0  0  1

julia> batch_text = [
    "hello world",
    "thank you!",
    "a",
    "this is some longer text, so length should be longer",
    "this is an even longer document. this is some longer text, so length should be longer",
]; 

julia> batches, reverse_indices = tensorize(doc_tokenizer, tokenizer, batch_text, 3)
2-element Vector{Tuple{AbstractArray, AbstractMatrix}}:
 (Int32[102 102 102; 3 3 3; … ; 1 1 1; 1 1 1], Bool[1 1 1; 1 1 1; … ; 0 0 0; 0 0 0])
 (Int32[102 102; 3 3; … ; 1 2937; 1 103], Bool[1 1; 1 1; … ; 0 1; 0 1])

julia> batches[1][1]                # this time they are sorted by length 
21×3 Matrix{Int32}:
  102   102   102
    3     3     3
 1038  7593  4068
  103  2089  2018
    1   103  1000
    1     1   103
    1     1     1
    1     1     1
    1     1     1
    1     1     1
    1     1     1
    1     1     1
    1     1     1
    1     1     1
    1     1     1
    1     1     1
    1     1     1
    1     1     1
    1     1     1
    1     1     1
    1     1     1

julia> reverse_indices              # the original order 
5-element Vector{Int64}:
 2
 3
 1
 4
 5

```
"""
function tensorize(doc_tokenizer::DocTokenizer, tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, batch_text::Vector{String}, bsize::Union{Missing, Int})
    # placeholder for [D] marker token
    batch_text = [". " * doc for doc in batch_text]

    # getting the integer ids and masks
    encoded_text = Transformers.TextEncoders.encode(tokenizer, batch_text)
    ids, mask = encoded_text.token, encoded_text.attention_mask
    integer_ids = reinterpret(Int32, ids)
    integer_mask = NeuralAttentionlib.getmask(mask, ids)[1, :, :]
    @assert isequal(size(integer_ids), size(integer_mask))
    @assert integer_ids isa AbstractMatrix{Int32}
    @assert integer_mask isa AbstractMatrix{Bool}

    # adding the [D] marker token ID
    integer_ids[2, :] .= doc_tokenizer.D_marker_token_id

    if ismissing(bsize)
        integer_ids, integer_mask
    else
        # we sort passages by length to do batch packing for more efficient use of the GPU
        integer_ids, integer_mask, reverse_indices = _sort_by_length(integer_ids, integer_mask, bsize)
        @assert length(reverse_indices) == length(batch_text)
        @assert integer_ids isa AbstractMatrix{Int32}
        @assert integer_mask isa AbstractMatrix{Bool}
        @assert reverse_indices isa Vector{Int64}

        batches = _split_into_batches(integer_ids, integer_mask, bsize)
        @assert batches isa Vector{Tuple{AbstractMatrix{Int32}, AbstractMatrix{Bool}}}

        batches, reverse_indices
    end
end

"""
    _sort_by_length(integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool}, bsize::Int)

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
function _sort_by_length(integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool}, bsize::Int)
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
    _split_into_batches(integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool}, bsize::Int)

Split the given `integer_ids` and `integer_mask` into batches of size `bsize`.

# Arguments

- `integer_ids`: The array of token IDs to batch. 
- `integer_mask`: The array of attention masks to batch.

# Returns

Batches of token IDs and attention masks, with each batch having size `bsize` (with the possibility of the last batch being smaller).
"""
function _split_into_batches(integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool}, bsize::Int)
    batch_size = size(integer_ids)[2]
    batches = Vector{Tuple{AbstractMatrix{Int32}, AbstractMatrix{Bool}}}()
    for offset in 1:bsize:batch_size
        push!(batches, (integer_ids[:, offset:min(batch_size, offset + bsize - 1)], integer_mask[:, offset:min(batch_size, offset + bsize - 1)]))
    end
    batches
end
