using ...ColBERT: ColBERTConfig

"""
    QueryTokenizer(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, config::ColBERTConfig)

Construct a `QueryTokenizer` from a given tokenizer and configuration. The resulting structure supports functions to perform CoLBERT-style query operations on query texts, including addition of the query marker token (`"[Q]"`) and the `"[MASK]"` token augmentation.

# Arguments

- `tokenizer`: A tokenizer that has been trained on the BERT vocabulary. Fetched from HuggingFace. This tokenizer should be configured to truncate or pad a sequence to the maximum allowed query length given by the config (see [`QuerySettings`](@ref)).
- `config`: The underlying [`ColBERTConfig`](@ref).

# Returns

A `QueryTokenizer` object.
"""
struct QueryTokenizer
    Q_marker_token_id::Int
    mask_token_id::Int
    config::ColBERTConfig
end

function QueryTokenizer(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, config::ColBERTConfig)
    Q_marker_token_id = TextEncodeBase.lookup(tokenizer.vocab, config.tokenizer_settings.query_token_id)
    mask_token_id = TextEncodeBase.lookup(tokenizer.vocab, "[MASK]")
    QueryTokenizer(Q_marker_token_id, mask_token_id, config)
end

"""
    tensorize(query_tokenizer::DocTokenizer, tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, batch_text::Vector{String}, bsize::Union{Missing, Int})

Convert a collection of queries to tensors in the ColBERT format. 

This function adds the query marker token at the beginning of each query text and then converts the text data into integer IDs and masks using the `tokenizer`. The returned tensors are batched into sizes given by the `bsize` argument. 

# Arguments

- `query_tokenizer`: An instance of the [`QueryTokenizer`](@ref) type. This object contains information about the query marker token ID and the mask token ID.
- `tokenizer`: The tokenizer which is used to convert text data into integer IDs.
- `batch_text`: A document texts that will be converted into tensors of token IDs.
- `bsize`: The size of the batches to split the `batch_text` into.
# Returns

`batches`, A `Vector` of tuples of arrays of token IDs and masks corresponding to the query texts. Each array in each tuple has shape `(L, N)`, where `L` is the maximum query length specified by the config (see [`QuerySettings`](@ref)), and `N` is the number of queries in the batch being considered.

# Examples

In this example, we first fetch the tokenizer from HuggingFace, and then configure the tokenizer to truncate or pad each sequence to the maximum query length specified by the config. Note that, at the time of writing this package, configuring tokenizers in [`Transformers.jl`](https://github.com/chengchingwen/Transformers.jl) doesn't have a clean interface; so, we have to manually configure the tokenizer. The `config` used is the same as in the example for [`ColBERTConfig`](@ref).

```julia-repl
julia> base_colbert = BaseColBERT("colbert-ir/colbertv2.0", config); 

julia> tokenizer = base_colbert.tokenizer;

julia> process = tokenizer.process;

julia> truncpad_pipe = Pipeline{:token}(TextEncodeBase.trunc_or_pad(config.query_settings.query_maxlen, "[PAD]", :tail, :tail), :token);

julia> process = process[1:4] |> truncpad_pipe |> process[6:end];

julia> tokenizer = Transformers.TextEncoders.BertTextEncoder(tokenizer.tokenizer, tokenizer.vocab, process; startsym = tokenizer.startsym, endsym = tokenizer.endsym, padsym = tokenizer.padsym, trunc = tokenizer.trunc);

julia> query_tokenizer = QueryTokenizer(tokenizer, config);

julia> queries = ["what are white spots on raspberries?"];

julia> batches = tensorize(query_tokenizer, tokenizer, queries, 128);

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
function tensorize(query_tokenizer::QueryTokenizer, tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, batch_text::Vector{String}, bsize::Union{Missing, Int})
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
    @assert isequal(size(integer_ids), size(integer_mask))
    @assert isequal(size(integer_ids)[1], query_tokenizer.config.query_settings.query_maxlen)

    # adding the [Q] marker token ID and [MASK] augmentation
    integer_ids[2, :] .= query_tokenizer.Q_marker_token_id 
    integer_ids[integer_ids .== 1] .= query_tokenizer.mask_token_id

    if query_tokenizer.config.query_settings.attend_to_mask_tokens 
        integer_mask[integer_ids .== query_tokenizer.mask_token_id] .= 1
        @assert isequal(sum(integer_mask), prod(size(integer_mask)))
    end

    batches = _split_into_batches(integer_ids, integer_mask, bsize)
    batches
end
