"""
    doc(bert::HF.HGFBertModel, linear::Layers.Dense,
        integer_ids::AbstractMatrix{Int32}, bitmask::AbstractMatrix{Bool})

Compute the hidden state of the BERT and linear layers of ColBERT for documents.

# Arguments

  - `bert`: The pre-trained BERT component of the ColBERT model. 
  - `linear`: The pre-trained linear component of the ColBERT model. 
  - `integer_ids`: An array of token IDs to be fed into the BERT model.
  - `integer_mask`: An array of corresponding attention masks. Should have the same shape as `integer_ids`.

# Returns

An array `D` containing the normalized embeddings for each token in each document.
It has shape `(D, L, N)`, where `D` is the embedding dimension (`128` for the linear layer
of ColBERT), and `(L, N)` is the shape of `integer_ids`, i.e `L` is the maximum length of
any document and `N` is the total number of documents.
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
    encode_passages(bert::HF.HGFBertModel, linear::Layers.Dense,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        passages::Vector{String}, dim::Int, index_bsize::Int,
        doc_token::String, skiplist::Vector{Int})

Encode a list of document passages.

The given `passages` are run through the underlying BERT model and the linear layer to
generate the embeddings, after doing relevant document-specific preprocessing.

# Arguments

  - `bert`: The pre-trained BERT component of the ColBERT model. 
  - `linear`: The pre-trained linear component of the ColBERT model. 
  - `tokenizer`: The tokenizer to be used. 
  - `passages`: A list of strings representing the passages to be encoded.
  - `dim`: The embedding dimension. 
  - `index_bsize`: The batch size to be used for running the transformer. 
  - `doc_token`: The document token. 
  - `skiplist`: A list of tokens to skip. 

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
    encode_queries(bert::HF.HGFBertModel, linear::Layers.Dense,
        tokenizer::TextEncoders.AbstractTransformerTextEncoder,
        queries::Vector{String}, dim::Int,
        index_bsize::Int, query_token::String, attend_to_mask_tokens::Bool,
        skiplist::Vector{Int})

Encode a list of query passages.

# Arguments

  - `bert`: The pre-trained BERT component of the ColBERT model. 
  - `linear`: The pre-trained linear component of the ColBERT model. 
  - `tokenizer`: The tokenizer to be used. 
  - `queries`: A list of strings representing the queries to be encoded.
  - `dim`: The embedding dimension. 
  - `index_bsize`: The batch size to be used for running the transformer. 
  - `query_token`: The query token. 
- `attend_to_mask_tokens`: Whether to attend to `"[MASK]"` tokens. 
  - `skiplist`: A list of tokens to skip. 

# Returns

An array containing the embeddings for each token in the query.

# Examples

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
