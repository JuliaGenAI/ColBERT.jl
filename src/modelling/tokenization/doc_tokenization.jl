"""
    tensorize_docs(config::ColBERTConfig,
        tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder,
        batch_text::Vector{String}, bsize::Union{Missing, Int})

Convert a collection of documents to tensors in the ColBERT format. 

This function adds the document marker token at the beginning of each document
and then converts the text data into integer IDs and masks using the `tokenizer`.
Some optimizing operations are performed on the documents. First, the arrays of
token IDs and attention masks are sorted by document lengths (this is for more
efficient use of GPUs on the batches; see [`_sort_by_length`](@ref)), and a list
`reverse_indices` is computed, which remembers the original order of the documents
(to reorder them later). The arrays of token IDs and attention masks are then
batched into batches of size `bsize` (see [`_split_into_batches`](@ref)).
Finally, the batches along with the list of `reverse_indices` are returned.

# Arguments

- `config`: The `ColBERTConfig` to be used to fetch the document marker token ID.
- `tokenizer`: The tokenizer which is used to convert text data into integer IDs.
- `batch_text`: A document texts that will be converted into tensors of token IDs.
- `bsize`: The size of the batches to split the `batch_text` into. 

# Returns

A tuple containing the following is returned:

- `batches`: A `Vector` of tuples of arrays of token IDs and masks, sorted in the order 
    of document lengths. Each array in each tuple has shape `(L, N)`, where `L` is the length
    of the largest document in `batch_text`, and `N` is the number of documents in the batch
    being considered.
- `reverse_indices`: A `Vector` containing the indices of the documents in their original order.

# Examples

```julia-repl
julia> using ColBERT, Transformers;

julia> config = ColBERTConfig();

julia> tokenizer = Transformers.load_tokenizer(config.checkpoint); 

julia> batch_text = [
    "hello world",
    "thank you!",
    "a",
    "this is some longer text, so length should be longer",
    "this is an even longer document. this is some longer text, so length should be longer",
]; 

julia> batches, reverse_indices = ColBERT.tensorize_docs(config, tokenizer, batch_text, 3)
(Tuple{AbstractMatrix{Int32}, AbstractMatrix{Bool}}[([102 102 102; 3 3 3; … ; 1 1 1; 1 1 1], [1 1 1; 1 1 1; … ; 0 0 0; 0 0 0]), ([102 102; 3 3; … ; 1 2937; 1 103], [1 1; 1 1; … ; 0 1; 0 1])], [2, 3, 1, 4, 5])

julia> batches[1][1]                # sorted by length 
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
function tensorize_docs(config::ColBERTConfig,
        tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder,
        batch_text::Vector{String})
    # placeholder for [D] marker token
    batch_text = [". " * doc for doc in batch_text]

    # getting the integer ids and masks
    encoded_text = Transformers.TextEncoders.encode(tokenizer, batch_text)
    ids, mask = encoded_text.token, encoded_text.attention_mask
    integer_ids = reinterpret(Int32, ids)
    integer_mask = NeuralAttentionlib.getmask(mask, ids)[1, :, :]

    # adding the [D] marker token ID
    D_marker_token_id = TextEncodeBase.lookup(
        tokenizer.vocab, config.doc_token_id)
    integer_ids[2, :] .= D_marker_token_id
    
    @assert isequal(size(integer_ids), size(integer_mask)) "size(integer_ids): $(size(integer_ids)), size(integer_mask): $(integer_mask)"
    @assert isequal(size(integer_ids)[2], length(batch_text))
    @assert integer_ids isa AbstractMatrix{Int32} "$(typeof(integer_ids))"
    @assert integer_mask isa AbstractMatrix{Bool} "$(typeof(integer_mask))"

    integer_ids, integer_mask
end
