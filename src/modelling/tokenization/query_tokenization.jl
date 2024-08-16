"""
    tensorize_queries(config::ColBERTConfig,
        tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder,
        batch_text::Vector{String})

Convert a collection of queries to tensors of token IDs and attention masks. 

This function adds the query marker token at the beginning of each query text
and then converts the text data into integer IDs and masks using the `tokenizer`.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) to be used to figure out the query marker token ID.
  - `tokenizer`: The tokenizer which is used to convert text data into integer IDs.
  - `batch_text`: A document texts that will be converted into tensors of token IDs.

# Returns

A tuple `integer_ids`, `integer_mask` containing the token IDs and the attention mask. Each
of these two matrices has shape `(L, N)`, where `L` is the maximum query length specified
by the `config` (see [`ColBERTConfig`](@ref)), and `N` is the number of queries in
`batch_text`.

# Examples

In this example, we first fetch the tokenizer from HuggingFace, and then configure the
tokenizer to truncate or pad each sequence to the maximum query length specified by the
config. Note that, at the time of writing this package, configuring tokenizers in
[`Transformers.jl`](https://github.com/chengchingwen/Transformers.jl) doesn't have a
clean interface; so, we have to manually configure the tokenizer.

```julia-repl
julia> using ColBERT, Transformers, TextEncodeBase;

julia> config = ColBERTConfig();

julia> tokenizer = Transformers.load_tokenizer(config.checkpoint);

julia> process = tokenizer.process;

julia> truncpad_pipe = Pipeline{:token}(
           TextEncodeBase.trunc_or_pad(config.query_maxlen, "[PAD]", :tail, :tail),
           :token);

julia> process = process[1:4] |> truncpad_pipe |> process[6:end];

julia> tokenizer = Transformers.TextEncoders.BertTextEncoder(
           tokenizer.tokenizer, tokenizer.vocab, process; startsym = tokenizer.startsym,
           endsym = tokenizer.endsym, padsym = tokenizer.padsym, trunc = tokenizer.trunc);

julia> queries = [
    "what are white spots on raspberries?",
    "what do rabbits eat?"
];

julia> integer_ids, integer_mask = ColBERT.tensorize_queries(config, tokenizer, queries);

julia> 32×2 reinterpret(Int32, ::Matrix{OneHot{0x0000773a}}):
   102    102
     2      2
  2055   2055
  2025   2080
  2318  20404
  7517   4522
  2007   1030
 20711    103
  2362    104
 20969    104
  1030    104
   103    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104
   104    104

julia> integer_mask
32×2 Matrix{Bool}:
 1  1
 1  1
 1  1
 1  1
 1  1
 1  1
 1  1
 1  1
 1  0
 1  0
 1  0
 1  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0
 0  0

```
"""
function tensorize_queries(config::ColBERTConfig,
        tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder,
        batch_text::Vector{String})
    # placeholder for [Q] marker token
    batch_text = [". " * query for query in batch_text]

    # getting the integer ids and masks
    encoded_text = Transformers.TextEncoders.encode(tokenizer, batch_text)
    ids, mask = encoded_text.token, encoded_text.attention_mask
    integer_ids = reinterpret(Int32, ids)
    integer_mask = NeuralAttentionlib.getmask(mask, ids)[1, :, :]
    @assert isequal(size(integer_ids), size(integer_mask)) "size(integer_ids): $(size(integer_ids)), size(integer_mask): $(size(integer_mask))"
    @assert isequal(
        size(integer_ids)[1], config.query_maxlen) "size(integer_ids): $(size(integer_ids)), query_maxlen: $(query_tokenizer.config.query_maxlen)"
    @assert integer_ids isa AbstractMatrix{Int32} "$(typeof(integer_ids))"
    @assert integer_mask isa AbstractMatrix{Bool} "$(typeof(integer_mask))"

    # adding the [Q] marker token ID and [MASK] augmentation
    Q_marker_token_id = TextEncodeBase.lookup(
        tokenizer.vocab, config.query_token_id)
    mask_token_id = TextEncodeBase.lookup(tokenizer.vocab, "[MASK]")
    integer_ids[2, :] .= Q_marker_token_id
    integer_ids[integer_ids .== 1] .= mask_token_id

    if config.attend_to_mask_tokens
        integer_mask[integer_ids .== mask_token_id] .= 1
        @assert isequal(sum(integer_mask), prod(size(integer_mask))) "sum(integer_mask): $(sum(integer_mask)), prod(size(integer_mask)): $(prod(size(integer_mask)))"
    end

    integer_ids, integer_mask
end
