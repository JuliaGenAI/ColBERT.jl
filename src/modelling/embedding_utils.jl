"""
    mask_skiplist(tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        integer_ids::AbstractMatrix{Int32}, skiplist::Union{Missing, Vector{Int64}})

Create a mask for the given `integer_ids`, based on the provided `skiplist`.
If the `skiplist` is not missing, then any token IDs in the list will be filtered out along with the padding token.
Otherwise, all tokens are included in the mask.

# Arguments

  - `tokenizer`: The underlying tokenizer.
  - `integer_ids`: An `Array` of token IDs for the documents.
  - `skiplist`: A list of token IDs to skip in the mask.

# Returns

An array of booleans indicating whether the corresponding token ID
is included in the mask or not. The array has the same shape as
`integer_ids`, i.e `(L, N)`, where `L` is the maximum length of
any document in `integer_ids` and `N` is the number of documents.

# Examples

In this example, we'll mask out all punctuations as well as the pad symbol
of a tokenizer.

```julia-repl
julia> using ColBERT: mask_skiplist;

julia> using TextEncodeBase

julia> tokenizer = load_hgf_pretrained_local("/home/codetalker7/models/colbertv2.0/:tokenizer");

julia> punctuations_and_padsym = [string.(collect("!\"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"));
                                   tokenizer.padsym];

julia> skiplist = [lookup(tokenizer.vocab, sym)
                    for sym in punctuations_and_padsym]
33-element Vector{Int64}:
 1000
 1001
 1002
 1003
 1004
 1005
 1006
 1007
 1008
 1009
 1010
 1011
 1012
 1013
 1014
 1025
 1026
 1027
 1028
 1029
 1030
 1031
 1032
 1033
 1034
 1035
 1036
 1037
 1064
 1065
 1066
 1067
    1

julia>  batch_text = [
    "no punctuation text",
    "this, batch,! of text contains puncts! but is larger so that? the other text contains pad symbol;"
];

julia> integer_ids, _ = tensorize_docs("[unused1]", tokenizer, batch_text)

julia> integer_ids
27×2 Matrix{Int32}:
   102    102
     3      3
  2054   2024
 26137   1011
  6594  14109
 14506   1011
  3794   1000
   103   1998
     1   3794
     1   3398
     1  26137
     1  16650
     1   1000
     1   2022
     1   2004
     1   3470
     1   2062
     1   2009
     1   1030
     1   1997
     1   2061
     1   3794
     1   3398
     1  11688
     1   6455
     1   1026
     1    103

julia> decode(tokenizer, integer_ids)
27×2 Matrix{String}:
 " [CLS]"      " [CLS]"
 " [unused1]"  " [unused1]"
 " no"         " this"
 " pun"        " ,"
 "ct"          " batch"
 "uation"      " ,"
 " text"       " !"
 " [SEP]"      " of"
 " [PAD]"      " text"
 " [PAD]"      " contains"
 " [PAD]"      " pun"
 " [PAD]"      "cts"
 " [PAD]"      " !"
 " [PAD]"      " but"
 " [PAD]"      " is"
 " [PAD]"      " larger"
 " [PAD]"      " so"
 " [PAD]"      " that"
 " [PAD]"      " ?"
 " [PAD]"      " the"
 " [PAD]"      " other"
 " [PAD]"      " text"
 " [PAD]"      " contains"
 " [PAD]"      " pad"
 " [PAD]"      " symbol"
 " [PAD]"      " ;"
 " [PAD]"      " [SEP]"

julia> mask_skiplist(integer_ids, skiplist)
27×2 BitMatrix:
 1  1
 1  1
 1  1
 1  0
 1  1
 1  0
 1  0
 1  1
 0  1
 0  1
 0  1
 0  1
 0  0
 0  1
 0  1
 0  1
 0  1
 0  1
 0  0
 0  1
 0  1
 0  1
 0  1
 0  1
 0  1
 0  0
 0  1
```
"""
function mask_skiplist!(mask::AbstractMatrix{Bool},
        integer_ids::AbstractMatrix{Int32}, skiplist::Vector{Int64})
    for token_id in skiplist
        mask .= mask .& (integer_ids .!= token_id)
    end
end

function _clear_masked_embeddings!(D::AbstractArray{Float32, 3},
        integer_ids::AbstractMatrix{Int32}, skiplist::Vector{Int})
    @assert isequal(size(D)[2:end], size(integer_ids))
    "size(D): $(size(D)), size(integer_ids): $(size(integer_ids))"

    # set everything to true
    mask = similar(integer_ids, Bool)                       # respects the device as well
    mask .= true
    mask_skiplist!(mask, integer_ids, skiplist)             # (doc_maxlen, current_batch_size)
    mask = reshape(mask, (1, size(mask)...))                # (1, doc_maxlen, current_batch_size)

    @assert isequal(size(mask)[2:end], size(D)[2:end])
    "size(mask): $(size(mask)), size(D): $(size(D))"
    @assert mask isa AbstractArray{Bool} "$(typeof(mask))"

    D .= D .* mask                                          # clear embeddings of masked tokens
    mask
end

function _flatten_embeddings(D::AbstractArray{Float32, 3})
    reshape(D, size(D, 1), prod(size(D)[2:end]))
end

function _remove_masked_tokens(
        D::AbstractMatrix{Float32}, mask::AbstractMatrix{Bool})
    D[:, reshape(mask, prod(size(mask)))]
end

