"""
    tensorize_docs(doc_token_id::String,
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
julia> using ColBERT: tensorize_docs, load_hgf_pretrained_local;

julia> using Transformers, Transformers.TextEncoders, TextEncodeBase;

julia> tokenizer = load_hgf_pretrained_local("/home/codetalker7/models/colbertv2.0/:tokenizer")

# configure the tokenizers maxlen and padding/truncation
julia> doc_maxlen = 20;

julia> process = tokenizer.process
Pipelines:
  target[token] := TextEncodeBase.nestedcall(string_getvalue, source)
  target[token] := Transformers.TextEncoders.grouping_sentence(target.token)
  target[(token, segment)] := SequenceTemplate{String}([CLS]:<type=1> Input[1]:<type=1> [SEP]:<type=1> (Input[2]:<type=2> [SEP]:<type=2>)...)(target.token)
  target[attention_mask] := (NeuralAttentionlib.LengthMask ∘ Transformers.TextEncoders.getlengths(512))(target.token)
  target[token] := TextEncodeBase.trunc_and_pad(512, [PAD], tail, tail)(target.token)
  target[token] := TextEncodeBase.nested2batch(target.token)
  target[segment] := TextEncodeBase.trunc_and_pad(512, 1, tail, tail)(target.segment)
  target[segment] := TextEncodeBase.nested2batch(target.segment)
  target[sequence_mask] := identity(target.attention_mask)
  target := (target.token, target.segment, target.attention_mask, target.sequence_mask)

julia> truncpad_pipe = Pipeline{:token}(
           TextEncodeBase.trunc_or_pad(doc_maxlen - 1, "[PAD]", :tail, :tail),
           :token);

julia> process = process[1:4] |> truncpad_pipe |> process[6:end];

julia> tokenizer = TextEncoders.BertTextEncoder(
           tokenizer.tokenizer, tokenizer.vocab, process; startsym = tokenizer.startsym,
           endsym = tokenizer.endsym, padsym = tokenizer.padsym, trunc = tokenizer.trunc);

julia> batch_text = [
    "hello world",
    "thank you!",
    "a",
    "this is some longer text, so length should be longer",
    "this is an even longer document. this is some longer text, so length should be longer",
];

julia> integer_ids, integer_mask = tensorize_docs(
    "[unused1]", tokenizer, batch_text)
(Int32[102 102 … 102 102; 3 3 … 3 3; … ; 1 1 … 1 2023; 1 1 … 1 2937], Bool[1 1 … 1 1; 1 1 … 1 1; … ; 0 0 … 0 1; 0 0 … 0 1])

julia> integer_ids
20×5 Matrix{Int32}:
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

julia> integer_mask
20×5 Matrix{Bool}:
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

julia> TextEncoders.decode(tokenizer, integer_ids)
20×5 Matrix{String}:
 "[CLS]"      "[CLS]"      "[CLS]"      "[CLS]"      "[CLS]"
 "[unused1]"  "[unused1]"  "[unused1]"  "[unused1]"  "[unused1]"
 "hello"      "thank"      "a"          "this"       "this"
 "world"      "you"        "[SEP]"      "is"         "is"
 "[SEP]"      "!"          "[PAD]"      "some"       "an"
 "[PAD]"      "[SEP]"      "[PAD]"      "longer"     "even"
 "[PAD]"      "[PAD]"      "[PAD]"      "text"       "longer"
 "[PAD]"      "[PAD]"      "[PAD]"      ","          "document"
 "[PAD]"      "[PAD]"      "[PAD]"      "so"         "."
 "[PAD]"      "[PAD]"      "[PAD]"      "length"     "this"
 "[PAD]"      "[PAD]"      "[PAD]"      "should"     "is"
 "[PAD]"      "[PAD]"      "[PAD]"      "be"         "some"
 "[PAD]"      "[PAD]"      "[PAD]"      "longer"     "longer"
 "[PAD]"      "[PAD]"      "[PAD]"      "[SEP]"      "text"
 "[PAD]"      "[PAD]"      "[PAD]"      "[PAD]"      ","
 "[PAD]"      "[PAD]"      "[PAD]"      "[PAD]"      "so"
 "[PAD]"      "[PAD]"      "[PAD]"      "[PAD]"      "length"
 "[PAD]"      "[PAD]"      "[PAD]"      "[PAD]"      "should"
 "[PAD]"      "[PAD]"      "[PAD]"      "[PAD]"      "be"
 "[PAD]"      "[PAD]"      "[PAD]"      "[PAD]"      "longer"
```
"""
function tensorize_docs(doc_token_id::String,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        batch_text::Vector{String})
    # we assume that tokenizer is configured to have maxlen: doc_maxlen - 1
    # getting the integer ids and masks
    encoded_text = TextEncoders.encode(tokenizer, batch_text)
    ids, mask = encoded_text.token, encoded_text.attention_mask
    integer_ids = reinterpret(Int32, ids)
    integer_mask = NeuralAttentionlib.getmask(mask, ids)[1, :, :]

    # adding the [D] marker token ID as the second token
    # first one is always the "[CLS]" token
    D_marker_token_id = lookup(tokenizer.vocab, doc_token_id) |> Int32
    integer_ids = [integer_ids[begin:1, :];
                   fill(D_marker_token_id, (1, length(batch_text)));
                   integer_ids[2:end, :]]
    integer_mask = [integer_mask[begin:1, :];
                    fill(true, (1, length(batch_text))); integer_mask[2:end, :]]

    @assert isequal(size(integer_ids), size(integer_mask))
    "size(integer_ids): $(size(integer_ids)), size(integer_mask): $(integer_mask)"
    @assert isequal(size(integer_ids, 2), length(batch_text))
    "size(integer_ids): $(size(integer_ids)), length(batch_text): $(length(batch_text))"
    @assert integer_ids isa Matrix{Int32} "$(typeof(integer_ids))"
    @assert integer_mask isa Matrix{Bool} "$(typeof(integer_mask))"

    integer_ids, integer_mask
end
