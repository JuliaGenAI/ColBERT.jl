"""
    tensorize_docs(config::ColBERTConfig,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        batch_text::Vector{String})

Convert a collection of documents to tensors in the ColBERT format.

This function adds the document marker token at the beginning of each document
and then converts the text data into integer IDs and masks using the `tokenizer`.

# Arguments

- `config`: The `ColBERTConfig` to be used to fetch the document marker token ID.
- `tokenizer`: The tokenizer which is used to convert text data into integer IDs.
- `batch_text`: A document texts that will be converted into tensors of token IDs.

# Returns

A tuple containing the following is returned:

- `integer_ids`: A `Matrix` of token IDs of shape `(L, N)`, where `L` is the length
    of the largest document in `batch_text`, and `N` is the number of documents in the batch
    being considered.
- `integer_mask`: A `Matrix` of attention masks, of the same shape as `integer_ids`.

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

julia> integer_ids, integer_mask = ColBERT.tensorize_docs(config, tokenizer, batch_text)
(Int32[102 102 … 102 102; 3 3 … 3 3; … ; 1 1 … 1 2937; 1 1 … 1 103], Bool[1 1 … 1 1; 1 1 … 1 1; … ; 0 0 … 0 1; 0 0 … 0 1])

julia> integer_ids
21×5 reinterpret(Int32, ::Matrix{PrimitiveOneHot.OneHot{0x0000773a}}):
  102   102   102   102   102
    3     3     3     3     3
 7593  4068  1038  2024  2024
 2089  2018   103  2004  2004
  103  1000     1  2071  2020
    1   103     1  2937  2131
    1     1     1  3794  2937
    1     1     1  1011  6255
    1     1     1  2062  1013
    1     1     1  3092  2024
    1     1     1  2324  2004
    1     1     1  2023  2071
    1     1     1  2937  2937
    1     1     1   103  3794
    1     1     1     1  1011
    1     1     1     1  2062
    1     1     1     1  3092
    1     1     1     1  2324
    1     1     1     1  2023
    1     1     1     1  2937
    1     1     1     1   103

julia> integer_mask
21×5 Matrix{Bool}:
 1  1  1  1  1
 1  1  1  1  1
 1  1  1  1  1
 1  1  1  1  1
 1  1  0  1  1
 0  1  0  1  1
 0  0  0  1  1
 0  0  0  1  1
 0  0  0  1  1
 0  0  0  1  1
 0  0  0  1  1
 0  0  0  1  1
 0  0  0  1  1
 0  0  0  1  1
 0  0  0  0  1
 0  0  0  0  1
 0  0  0  0  1
 0  0  0  0  1
 0  0  0  0  1
 0  0  0  0  1
 0  0  0  0  1

```
"""
function tensorize_docs(config::ColBERTConfig,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
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
