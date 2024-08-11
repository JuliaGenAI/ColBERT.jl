"""
    tensorize_queries(config::ColBERTConfig,
        tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder,
        batch_text::Vector{String}, bsize::Union{Missing, Int})

Convert a collection of queries to tensors in the ColBERT format.

This function adds the query marker token at the beginning of each query text
and then converts the text data into integer IDs and masks using the `tokenizer`.
The returned tensors are batched into sizes given by the `bsize` argument.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) to be used to figure out the query marker token ID.
  - `tokenizer`: The tokenizer which is used to convert text data into integer IDs.
  - `batch_text`: A document texts that will be converted into tensors of token IDs.
  - `bsize`: The size of the batches to split the `batch_text` into.

# Returns

`batches`, A `Vector` of tuples of arrays of token IDs and masks corresponding to
the query texts. Each array in each tuple has shape `(L, N)`, where `L` is the
maximum query length specified by the config (see [`ColBERTConfig`](@ref)), and `N`
is the number of queries in the batch being considered.

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

julia> queries = ["what are white spots on raspberries?"];

julia> batches = ColBERT.tensorize_queries(config, tokenizer, queries, 128);

julia> integer_ids, integer_mask = batches[1][1], batches[1][2];

julia> integer_ids
32×1 Matrix{Int32}:
   102
     2
  2055
  2025
  2318
  7517
  2007
 20711
  2362
 20969
  1030
   103
     ⋮
   104
   104
   104
   104
   104
   104
   104
   104
   104
   104
   104

julia> integer_mask
32×1 Matrix{Bool}:
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 ⋮
 0
 0
 0
 0
 0
 0
 0
 0
 0
 0
 0
```
"""
function tensorize_queries(config::ColBERTConfig,
        tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder,
        batch_text::Vector{String}, bsize::Union{Missing, Int})
    if ismissing(bsize)
        error("Currently bsize cannot be missing!")
    end

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

    batches = _split_into_batches(integer_ids, integer_mask, bsize)
    @assert batches isa Vector{Tuple{AbstractMatrix{Int32}, AbstractMatrix{Bool}}} "$(typeof(batches))"

    batches
end
