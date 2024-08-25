"""
    _integer_ids_and_mask(
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        batch_text::AbstractVector{String})

Run `batch_text` through `tokenizer` to get matrices of tokens and attention mask.

# Arguments

  - `tokenizer`: The tokenizer to be used to tokenize the texts.
  - `batch_text`: The list of texts to tokenize.

# Returns

A tuple `integer_ids, bitmask`, where `integer_ids` is a Matrix containing token IDs
and `bitmask` is the attention mask.

# Examples

```julia-repl
julia> using ColBERT: _integer_ids_and_mask, load_hgf_pretrained_local;

julia> tokenizer = load_hgf_pretrained_local("/home/codetalker7/models/colbertv2.0/:tokenizer");

julia> batch_text = [
    "hello world",
    "thank you!",
    "a",
    "this is some longer text, so length should be longer",
    "this is an even longer document. this is some longer text, so length should be longer",
];

julia> integer_ids, bitmask = _integer_ids_and_mask(tokenizer, batch_text);

julia> integer_ids
20×5 Matrix{Int32}:
  102   102   102   102   102
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

julia> bitmask
20×5 BitMatrix:
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
function _integer_ids_and_mask(
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        batch_text::AbstractVector{String})
    encoded_text = TextEncoders.encode(tokenizer, batch_text)
    ids, length_mask = encoded_text.token, encoded_text.attention_mask
    integer_ids = reinterpret(Int32, ids) |> Matrix{Int32}
    bitmask = length_mask .* trues(1, size(integer_ids)...)     # (1, max_len, batch_size)
    bitmask = reshape(bitmask, size(bitmask)[2:end]...)         # (max_len, batch_size)

    @assert isequal(size(integer_ids), size(bitmask))
    "size(integer_ids): $(size(integer_ids)), size(bitmask): $(bitmask)"
    @assert isequal(size(integer_ids, 2), length(batch_text))
    "size(integer_ids): $(size(integer_ids)), length(batch_text): $(length(batch_text))"
    @assert integer_ids isa Matrix{Int32} "$(typeof(integer_ids))"
    @assert bitmask isa BitMatrix "$(typeof(bitmask))"
    "typeof(bitmask): $(typeof(bitmask))"

    integer_ids, bitmask
end

"""
    _add_marker_row(data::AbstractMatrix{T}, marker::T) where {T}

Add row containing `marker` as the second row of `data`.

# Arguments

  - `data`: The matrix in which the row is to be added.
  - `marker`: The marker to be added.

# Returns

A matrix equal to `data`, with the second row being filled with `marker`.

# Examples

```julia-repl
julia> using ColBERT: _add_marker_row; 

julia> x = ones(Float32, 5, 5); 
5×5 Matrix{Float32}:
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0

julia> _add_marker_row(x, zero(Float32))
6×5 Matrix{Float32}:
 1.0  1.0  1.0  1.0  1.0
 0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
```

"""
function _add_marker_row(data::AbstractMatrix{T}, marker::T) where {T}
    [data[begin:1, :]; fill(marker, (1, size(data, 2))); data[2:end, :]]
end
