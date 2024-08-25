"""
using TextEncodeBase: tokenize
    tensorize_queries(query_token::String, attend_to_mask_tokens::Bool,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
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
julia> using ColBERT: tensorize_queries, load_hgf_pretrained_local;

julia> using Transformers, Transformers.TextEncoders, TextEncodeBase;

julia> tokenizer = load_hgf_pretrained_local("/home/codetalker7/models/colbertv2.0/:tokenizer");

# configure the tokenizers maxlen and padding/truncation
julia> query_maxlen = 32;

julia> process = tokenizer.process;

julia> truncpad_pipe = Pipeline{:token}(
    TextEncodeBase.trunc_or_pad(query_maxlen - 1, "[PAD]", :tail, :tail),
    :token);

julia> process = process[1:4] |> truncpad_pipe |> process[6:end];

julia> tokenizer = TextEncoders.BertTextEncoder(
    tokenizer.tokenizer, tokenizer.vocab, process; startsym = tokenizer.startsym,
    endsym = tokenizer.endsym, padsym = tokenizer.padsym, trunc = tokenizer.trunc);

julia> batch_text = [
    "what are white spots on raspberries?",
    "what do rabbits eat?",
    "this is a really long query. I'm deliberately making this long"*
    "so that you can actually see that this is really truncated at 32 tokens"*
    "and that the other two queries are padded to get 32 tokens."*
    "this makes this a nice query as an example."
];

julia> integer_ids, bitmask = tensorize_queries(
    "[unused0]", false, tokenizer, batch_text);
(Int32[102 102 102; 2 2 2; … ; 104 104 8792; 104 104 2095], Bool[1 1 1; 1 1 1; … ; 0 0 1; 0 0 1])

julia> integer_ids
32×3 Matrix{Int32}:
   102    102    102
     2      2      2
  2055   2055   2024
  2025   2080   2004
  2318  20404   1038
  7517   4522   2429
  2007   1030   2147
 20711    103  23033
  2362    104   1013
 20969    104   1046
  1030    104   1006
   103    104   1050
   104    104   9970
   104    104   2438
   104    104   2024
   104    104   2147
   104    104   6500
   104    104   2009
   104    104   2018
   104    104   2065
   104    104   2942
   104    104   2157
   104    104   2009
   104    104   2024
   104    104   2004
   104    104   2429
   104    104  25450
   104    104   2013
   104    104   3591
   104    104  19205
   104    104   8792
   104    104   2095

julia> bitmask 
32×3 Matrix{Bool}:
 1  1  1
 1  1  1
 1  1  1
 1  1  1
 1  1  1
 1  1  1
 1  1  1
 1  1  1
 1  0  1
 1  0  1
 1  0  1
 1  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1
 0  0  1

julia> TextEncoders.decode(tokenizer, integer_ids)
32×3 Matrix{String}:
 "[CLS]"      "[CLS]"      "[CLS]"
 "[unused0]"  "[unused0]"  "[unused0]"
 "what"       "what"       "this"
 "are"        "do"         "is"
 "white"      "rabbits"    "a"
 "spots"      "eat"        "really"
 "on"         "?"          "long"
 "ras"        "[SEP]"      "query"
 "##p"        "[MASK]"     "."
 "##berries"  "[MASK]"     "i"
 "?"          "[MASK]"     "'"
 "[SEP]"      "[MASK]"     "m"
 "[MASK]"     "[MASK]"     "deliberately"
 "[MASK]"     "[MASK]"     "making"
 "[MASK]"     "[MASK]"     "this"
 "[MASK]"     "[MASK]"     "long"
 "[MASK]"     "[MASK]"     "##so"
 "[MASK]"     "[MASK]"     "that"
 "[MASK]"     "[MASK]"     "you"
 "[MASK]"     "[MASK]"     "can"
 "[MASK]"     "[MASK]"     "actually"
 "[MASK]"     "[MASK]"     "see"
 "[MASK]"     "[MASK]"     "that"
 "[MASK]"     "[MASK]"     "this"
 "[MASK]"     "[MASK]"     "is"
 "[MASK]"     "[MASK]"     "really"
 "[MASK]"     "[MASK]"     "truncated"
 "[MASK]"     "[MASK]"     "at"
 "[MASK]"     "[MASK]"     "32"
 "[MASK]"     "[MASK]"     "token"
 "[MASK]"     "[MASK]"     "##san"
 "[MASK]"     "[MASK]"     "##d"
```
"""
function tensorize_queries(query_token::String, attend_to_mask_tokens::Bool,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        batch_text::Vector{String})
    # we assume that tokenizer is configured to have maxlen: query_maxlen - 1
    integer_ids, bitmask = _integer_ids_and_mask(tokenizer, batch_text)

    # adding the [Q] marker token ID and [MASK] augmentation
    Q_marker_token_id = TextEncodeBase.lookup(
        tokenizer.vocab, query_token) |> Int32
    mask_token_id = TextEncodeBase.lookup(tokenizer.vocab, "[MASK]") |> Int32
    pad_token_id = TextEncodeBase.lookup(
        tokenizer.vocab, tokenizer.config.padsym) |> Int32
    integer_ids = _add_marker_row(integer_ids, Q_marker_token_id)
    bitmask = _add_marker_row(bitmask, true)
    integer_ids[integer_ids .== pad_token_id] .= mask_token_id

    if attend_to_mask_tokens
        bitmask[integer_ids .== mask_token_id] .= true
        @assert isequal(sum(bitmask), prod(size(bitmask)))
        "sum(integer_mask): $(sum(bitmask)), prod(size(integer_mask)): $(prod(size(bitmask)))"
    end

    integer_ids, bitmask
end
