"""
    BaseColBERT(;
        bert::HuggingFace.HGFBertModel, linear::Layers.Dense,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder)

A struct representing the BERT model, linear layer, and the tokenizer used to compute
embeddings for documents and queries.

# Arguments

  - `bert`: The pre-trained BERT model used to generate the embeddings.
  - `linear`: The linear layer used to project the embeddings to a specific dimension.
  - `tokenizer`: The tokenizer to used by the BERT model.

# Returns

A [`BaseColBERT`](@ref) object.

# Examples

```julia-repl
julia> using ColBERT, CUDA;

julia> base_colbert = BaseColBERT("/home/codetalker7/models/colbertv2.0/");

julia> base_colbert.bert
HGFBertModel(
  Chain(
    CompositeEmbedding(
      token = Embed(768, 30522),        # 23_440_896 parameters
      position = ApplyEmbed(.+, FixedLenPositionEmbed(768, 512)),  # 393_216 parameters
      segment = ApplyEmbed(.+, Embed(768, 2), Transformers.HuggingFace.bert_ones_like),  # 1_536 parameters
    ),
    DropoutLayer<nothing>(
      LayerNorm(768, ϵ = 1.0e-12),      # 1_536 parameters
    ),
  ),
  Transformer<12>(
    PostNormTransformerBlock(
      DropoutLayer<nothing>(
        SelfAttention(
          MultiheadQKVAttenOp(head = 12, p = nothing),
          Fork<3>(Dense(W = (768, 768), b = true)),  # 1_771_776 parameters
          Dense(W = (768, 768), b = true),  # 590_592 parameters
        ),
      ),
      LayerNorm(768, ϵ = 1.0e-12),      # 1_536 parameters
      DropoutLayer<nothing>(
        Chain(
          Dense(σ = NNlib.gelu, W = (768, 3072), b = true),  # 2_362_368 parameters
          Dense(W = (3072, 768), b = true),  # 2_360_064 parameters
        ),
      ),
      LayerNorm(768, ϵ = 1.0e-12),      # 1_536 parameters
    ),
  ),                  # Total: 192 arrays, 85_054_464 parameters, 40.422 KiB.
  Branch{(:pooled,) = (:hidden_state,)}(
    BertPooler(Dense(σ = NNlib.tanh_fast, W = (768, 768), b = true)),  # 590_592 parameters
  ),
)                   # Total: 199 arrays, 109_482_240 parameters, 43.578 KiB.

julia> base_colbert.linear
Dense(W = (768, 128), b = true)  # 98_432 parameters

julia> base_colbert.tokenizer
TrfTextEncoder(
├─ TextTokenizer(MatchTokenization(WordPieceTokenization(bert_uncased_tokenizer, WordPiece(vocab_size = 30522, unk = [UNK], max_char = 100)), 5 patterns)),
├─ vocab = Vocab{String, SizedArray}(size = 30522, unk = [UNK], unki = 101),
├─ config = @NamedTuple{startsym::String, endsym::String, padsym::String, trunc::Union{Nothing, Int64}}(("[CLS]", "[SEP]", "[PAD]", 512)),
├─ annotate = annotate_strings,
├─ onehot = lookup_first,
├─ decode = nestedcall(remove_conti_prefix),
├─ textprocess = Pipelines(target[token] := join_text(source); target[token] := nestedcall(cleanup ∘ remove_prefix_space, target.token); target := (target.token)),
└─ process = Pipelines:
  ╰─ target[token] := TextEncodeBase.nestedcall(string_getvalue, source)
  ╰─ target[token] := Transformers.TextEncoders.grouping_sentence(target.token)
  ╰─ target[(token, segment)] := SequenceTemplate{String}([CLS]:<type=1> Input[1]:<type=1> [SEP]:<type=1> (Input[2]:<type=2> [SEP]:<type=2>)...)(target.token)
  ╰─ target[attention_mask] := (NeuralAttentionlib.LengthMask ∘ Transformers.TextEncoders.getlengths(512))(target.token)
  ╰─ target[token] := TextEncodeBase.trunc_and_pad(512, [PAD], tail, tail)(target.token)
  ╰─ target[token] := TextEncodeBase.nested2batch(target.token)
  ╰─ target[segment] := TextEncodeBase.trunc_and_pad(512, 1, tail, tail)(target.segment)
  ╰─ target[segment] := TextEncodeBase.nested2batch(target.segment)
  ╰─ target[sequence_mask] := identity(target.attention_mask)
  ╰─ target := (target.token, target.segment, target.attention_mask, target.sequence_mask)
```
"""
struct BaseColBERT
    bert::HF.HGFBertModel
    linear::Layers.Dense
    tokenizer::TextEncoders.AbstractTransformerTextEncoder
end

function BaseColBERT(modelpath::AbstractString)
    tokenizer, bert_model, linear = load_hgf_pretrained_local(modelpath)
    bert_model = bert_model |> Flux.gpu
    linear = linear |> Flux.gpu
    BaseColBERT(bert_model, linear, tokenizer)
end

"""
    Checkpoint(model::BaseColBERT, config::ColBERTConfig)

A wrapper for [`BaseColBERT`](@ref), containing information for generating embeddings
for docs and queries.

If the `config` is set to mask punctuations, then the `skiplist` property of the created
[`Checkpoint`](@ref) will be set to a list of token IDs of punctuations. Otherwise, it will be empty.

# Arguments

  - `model`: The [`BaseColBERT`](@ref) to be wrapped.
  - `config`: The underlying [`ColBERTConfig`](@ref).

# Returns

The created [`Checkpoint`](@ref).

# Examples

Continuing from the example for [`BaseColBERT`](@ref):

```julia-repl
julia> checkpoint = Checkpoint(base_colbert, config)

julia> checkpoint.skiplist              # by default, all punctuations
32-element Vector{Int64}:
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
    ⋮
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
```
"""
struct Checkpoint
    model::BaseColBERT
    skiplist::Vector{Int64}
end

function Checkpoint(model::BaseColBERT, config::ColBERTConfig)
    if config.mask_punctuation
        punctuation_list = string.(collect("!\"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"))
        skiplist = [TextEncodeBase.lookup(model.tokenizer.vocab, punct)
                    for punct in punctuation_list]
    else
        skiplist = Vector{Int64}()
    end
    Checkpoint(model, skiplist)
end

"""
    doc(
        config::ColBERTConfig, checkpoint::Checkpoint, integer_ids::AbstractMatrix{Int32},
        integer_mask::AbstractMatrix{Bool})

Compute the hidden state of the BERT and linear layers of ColBERT for documents.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) being used.
  - `checkpoint`: The [`Checkpoint`](@ref) containing the layers to compute the embeddings.
  - `integer_ids`: An array of token IDs to be fed into the BERT model.
  - `integer_mask`: An array of corresponding attention masks. Should have the same shape as `integer_ids`.

# Returns

A tuple `D, mask`, where:

  - `D` is an array containing the normalized embeddings for each token in each document.
    It has shape `(D, L, N)`, where `D` is the embedding dimension (`128` for the linear layer
    of ColBERT), and `(L, N)` is the shape of `integer_ids`, i.e `L` is the maximum length of
    any document and `N` is the total number of documents.
  - `mask` is an array containing attention masks for all documents, after masking out any
    tokens in the `skiplist` of `checkpoint`. It has shape `(1, L, N)`, where `(L, N)`
    is the same as described above.

# Examples

Continuing from the example in [`tensorize_docs`](@ref) and [`Checkpoint`](@ref):

```julia-repl
julia> integer_ids, integer_mask = batches[1]

julia> D, mask = ColBERT.doc(config, checkpoint, integer_ids, integer_mask);

julia> typeof(D), size(D)
(CuArray{Float32, 3, CUDA.DeviceMemory}, (128, 21, 3))

julia> mask
1×21×3 CuArray{Bool, 3, CUDA.DeviceMemory}:
[:, :, 1] =
 1  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0

[:, :, 2] =
 1  1  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0

[:, :, 3] =
 1  1  1  1  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
```
"""
function doc(bert::HF.HGFBertModel, linear::Layers.Dense,
        integer_ids::AbstractMatrix{Int32}, bitmask::AbstractMatrix{Bool})
    linear(bert((token = integer_ids,
        attention_mask = NeuralAttentionlib.GenericSequenceMask(bitmask))).hidden_state)
end

function _doc_embeddings_and_doclens(
        bert::HF.HGFBertModel, linear::Layers.Dense, skiplist::Vector{Int},
        integer_ids::AbstractMatrix{Int32}, bitmask::AbstractMatrix{Bool})
    D = doc(bert, linear, integer_ids, bitmask)                 # (dim, doc_maxlen, current_batch_size)
    mask = _clear_masked_embeddings!(D, integer_ids, skiplist)  # (1, doc_maxlen, current_batch_size)

    # normalize each embedding in D; along dims = 1
    _normalize_array!(D, dims = 1)

    # get the doclens by unsqueezing the mask
    mask = reshape(mask, size(mask)[2:end])                     # (doc_maxlen, current_batch_size)
    doclens = vec(sum(mask, dims = 1))

    # flatten out embeddings, i.e get embeddings for each token in each passage
    D = _flatten_embeddings(D)                                  # (dim, total_num_embeddings)

    # remove embeddings for masked tokens
    D = _remove_masked_tokens(D, mask)                          # (dim, total_num_masked_embeddings)

    @assert ndims(D)==2 "ndims(D): $(ndims(D))"
    @assert size(D, 2)==sum(doclens) "size(D): $(size(D)), sum(doclens): $(sum(doclens))"
    @assert D isa AbstractMatrix{Float32} "$(typeof(D))"
    @assert doclens isa AbstractVector{Int64} "$(typeof(doclens))"

    D, doclens
end

function _query_embeddings(
        bert::HF.HGFBertModel, linear::Layers.Dense, skiplist::Vector{Int},
        integer_ids::AbstractMatrix{Int32}, bitmask::AbstractMatrix{Bool})
    Q = doc(bert, linear, integer_ids, bitmask)                 # (dim, query_maxlen, current_batch_size)

    # skiplist only contains the pad symbol by default
    _ = _clear_masked_embeddings!(Q, integer_ids, skiplist)

    # normalize each embedding in Q; along dims = 1
    _normalize_array!(Q, dims = 1)

    @assert ndims(Q)==3 "ndims(Q): $(ndims(Q))"
    @assert(isequal(size(Q)[2:end], size(integer_ids)),
        "size(Q): $(size(Q)), size(integer_ids): $(size(integer_ids))")
    @assert Q isa AbstractArray{Float32} "$(typeof(Q))"

    Q
end

"""
    encode_passages(
        config::ColBERTConfig, checkpoint::Checkpoint, passages::Vector{String})

Encode a list of passages using `checkpoint`.

The given `passages` are run through the underlying BERT model and the linear layer to
generate the embeddings, after doing relevant document-specific preprocessing.
See [`docFromText`](@ref) for more details.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) to be used.
  - `checkpoint`: The [`Checkpoint`](@ref) used to encode the passages.
  - `passages`: A list of strings representing the passages to be encoded.

# Returns

A tuple `embs, doclens` where:

  - `embs::AbstractMatrix{Float32}`: The full embedding matrix. Of shape `(D, N)`,
    where `D` is the embedding dimension and `N` is the total number of embeddings
    across all the passages.
  - `doclens::AbstractVector{Int}`: A vector of document lengths for each passage,
    i.e the total number of attended tokens for each document passage.

# Examples

```julia-repl
julia> using ColBERT: load_hgf_pretrained_local, ColBERTConfig, encode_passages;

julia> using CUDA, Flux, Transformers, TextEncodeBase;

julia> config = ColBERTConfig();

julia> dim = config.dim
128

julia> index_bsize = 128;                       # this is the batch size to be fed in the transformer

julia> doc_maxlen = config.doc_maxlen
300

julia> doc_token = config.doc_token_id
"[unused1]"

julia> tokenizer, bert, linear = load_hgf_pretrained_local("/home/codetalker7/models/colbertv2.0/");

julia> process = tokenizer.process;

julia> truncpad_pipe = Pipeline{:token}(
           TextEncodeBase.trunc_and_pad(doc_maxlen - 1, "[PAD]", :tail, :tail),
           :token);

julia> process = process[1:4] |> truncpad_pipe |> process[6:end];

julia> tokenizer = TextEncoders.BertTextEncoder(
           tokenizer.tokenizer, tokenizer.vocab, process; startsym = tokenizer.startsym,
           endsym = tokenizer.endsym, padsym = tokenizer.padsym, trunc = tokenizer.trunc);

julia> bert = bert |> Flux.gpu;

julia> linear = linear |> Flux.gpu;

julia> passages = readlines("./downloads/lotte/lifestyle/dev/collection.tsv")[1:1000];

julia> punctuations_and_padsym = [string.(collect("!\"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"));
                                   tokenizer.padsym];

julia> skiplist = [lookup(tokenizer.vocab, sym)
                    for sym in punctuations_and_padsym];

julia> @time embs, doclens = encode_passages(
    bert, linear, tokenizer, passages, dim, index_bsize, doc_token, skiplist)      # second run stats
[ Info: Encoding 1000 passages.
 25.247094 seconds (29.65 M allocations: 1.189 GiB, 37.26% gc time, 0.00% compilation time)
(Float32[-0.08001435 -0.10785186 … -0.08651956 -0.12118215; 0.07319974 0.06629379 … 0.0929825 0.13665271; … ; -0.037957724 -0.039623592 … 0.031274226 0.063107446; 0.15484622 0.16779025 … 0.11533891 0.11508792], [279, 117, 251, 105, 133, 170, 181, 115, 190, 132  …  76, 204, 199, 244, 256, 125, 251, 261, 262, 263])

```
"""
function encode_passages(bert::HF.HGFBertModel, linear::Layers.Dense,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        passages::Vector{String}, dim::Int, index_bsize::Int,
        doc_token::String, skiplist::Vector{Int})
    @info "Encoding $(length(passages)) passages."
    length(passages) == 0 && return rand(Float32, dim, 0), rand(Int, 0)

    # batching here to avoid storing intermediate embeddings on GPU
    embs, doclens = Vector{AbstractMatrix{Float32}}(), Vector{Int}()
    for passage_offset in 1:index_bsize:length(passages)
        passage_end_offset = min(
            length(passages), passage_offset + index_bsize - 1)

        # get the token IDs and attention mask
        integer_ids, bitmask = tensorize_docs(
            doc_token, tokenizer, passages[passage_offset:passage_end_offset])

        integer_ids = integer_ids |> Flux.gpu
        bitmask = bitmask |> Flux.gpu

        # run the tokens and attention mask through the transformer
        # and mask the skiplist tokens
        D, doclens_ = _doc_embeddings_and_doclens(
            bert, linear, skiplist, integer_ids, bitmask)

        push!(embs, Flux.cpu(D))
        append!(doclens, Flux.cpu(doclens_))
    end
    embs = cat(embs..., dims = 2)
    embs, doclens
end

"""
    encode_query(searcher::Searcher, query::String)

Encode a search query to a matrix of embeddings using the provided `searcher`. The encoded query can then be used to search the collection.

# Arguments

  - `searcher`: A Searcher object that contains information about the collection and the index.
  - `query`: The search query to encode.

# Returns

An array containing the embeddings for each token in the query. Also see [queryFromText](@ref) to see the size of the array.

# Examples

Here's an example using the `config` and `checkpoint` from the example for [`Checkpoint`](@ref).

```julia-repl
julia> using ColBERT: load_hgf_pretrained_local, ColBERTConfig, encode_queries;

julia> using CUDA, Flux, Transformers, TextEncodeBase;

julia> config = ColBERTConfig();

julia> dim = config.dim
128

julia> index_bsize = 128;                       # this is the batch size to be fed in the transformer

julia> query_maxlen = config.query_maxlen
300

julia> query_token = config.query_token_id
"[unused1]"

julia> tokenizer, bert, linear = load_hgf_pretrained_local("/home/codetalker7/models/colbertv2.0/");

julia> process = tokenizer.process;

julia> truncpad_pipe = Pipeline{:token}(
           TextEncodeBase.trunc_or_pad(query_maxlen - 1, "[PAD]", :tail, :tail),
           :token);

julia> process = process[1:4] |> truncpad_pipe |> process[6:end];

julia> tokenizer = TextEncoders.BertTextEncoder(
           tokenizer.tokenizer, tokenizer.vocab, process; startsym = tokenizer.startsym,
           endsym = tokenizer.endsym, padsym = tokenizer.padsym, trunc = tokenizer.trunc);

julia> bert = bert |> Flux.gpu;

julia> linear = linear |> Flux.gpu;

julia> skiplist = [lookup(tokenizer.vocab, tokenizer.padsym)]
1-element Vector{Int64}:
 1

julia> attend_to_mask_tokens = config.attend_to_mask_tokens

julia> queries = [
    "what are white spots on raspberries?",
    "here is another query!",
];

julia> @time encode_queries(bert, linear, tokenizer, queries, dim, index_bsize,
    query_token, attend_to_mask_tokens, skiplist);
[ Info: Encoding 2 queries.
  0.029858 seconds (27.58 k allocations: 781.727 KiB, 0.00% compilation time)
```
"""
function encode_queries(bert::HF.HGFBertModel, linear::Layers.Dense,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        queries::Vector{String}, dim::Int,
        index_bsize::Int, query_token::String, attend_to_mask_tokens::Bool,
        skiplist::Vector{Int})
    # we assume that tokenizer is configured to truncate or pad to query_maxlen - 1
    @info "Encoding $(length(queries)) queries."
    length(queries) == 0 && return rand(Float32, dim, 0)

    # batching here to avoid storing intermediate embeddings on GPU
    embs = Vector{AbstractArray{Float32, 3}}()
    for query_offset in 1:index_bsize:length(queries)
        query_end_offset = min(
            length(queries), query_offset + index_bsize - 1)

        # get the token IDs and attention mask
        integer_ids, bitmask = tensorize_queries(
            query_token, attend_to_mask_tokens, tokenizer,
            queries[query_offset:query_end_offset])                 # (query_maxlen, current_batch_size)

        integer_ids = integer_ids |> Flux.gpu
        bitmask = bitmask |> Flux.gpu

        # run the tokens and attention mask through the transformer
        Q = _query_embeddings(
            bert, linear, skiplist, integer_ids, bitmask)           # (dim, query_maxlen, current_batch_size)

        push!(embs, Flux.cpu(Q))
    end
    embs = cat(embs..., dims = 3)
end
