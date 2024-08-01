"""
    BaseColBERT(; bert::Transformers.HuggingFace.HGFBertModel, linear::Transformers.Layers.Dense, tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder)

A struct representing the BERT model, linear layer, and the tokenizer used to compute embeddings for documents and queries.

# Arguments
- `bert`: The pre-trained BERT model used to generate the embeddings.
- `linear`: The linear layer used to project the embeddings to a specific dimension.
- `tokenizer`: The tokenizer to used by the BERT model.

# Returns

A [`BaseColBERT`](@ref) object.

# Examples

The `config` in the below example is taken from the example in [`ColBERTConfig`](@ref).

```julia-repl
julia> base_colbert = BaseColBERT(checkpoint, config); 

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
  ),                  # Total: 192 arrays, 85_054_464 parameters, 324.477 MiB.
  Branch{(:pooled,) = (:hidden_state,)}(
    BertPooler(Dense(σ = NNlib.tanh_fast, W = (768, 768), b = true)),  # 590_592 parameters
  ),
)                   # Total: 199 arrays, 109_482_240 parameters, 417.664 MiB.

julia> base_colbert.linear
Dense(W = (768, 128), b = true)  # 98_432 parameters

julia> base_colbert.tokenizer
BertTextEncoder(
├─ TextTokenizer(MatchTokenization(WordPieceTokenization(bert_uncased_tokenizer, WordPiece(vocab_size = 30522, unk = [UNK], max_char = 100)), 5 patterns)),
├─ vocab = Vocab{String, SizedArray}(size = 30522, unk = [UNK], unki = 101),
├─ startsym = [CLS],
├─ endsym = [SEP],
├─ padsym = [PAD],
├─ trunc = 512,
└─ process = Pipelines:
  ╰─ target[token] := TextEncodeBase.nestedcall(string_getvalue, source)
  ╰─ target[token] := Transformers.TextEncoders.grouping_sentence(target.token)
  ╰─ target[(token, segment)] := SequenceTemplate{String}([CLS]:<type=1> Input[1]:<type=1> [SEP]:<type=1> (Input[2]:<type=2> [SEP]:<type=2>)...)(target.token)
  ╰─ target[attention_mask] := (NeuralAttentionlib.LengthMask ∘ Transformers.TextEncoders.getlengths(512))(target.token)
  ╰─ target[token] := TextEncodeBase.trunc_and_pad(512, [PAD], tail, tail)(target.token)
  ╰─ target[token] := TextEncodeBase.nested2batch(target.token)
  ╰─ target[segment] := TextEncodeBase.trunc_and_pad(512, 1, tail, tail)(target.segment)
  ╰─ target[segment] := TextEncodeBase.nested2batch(target.segment)
  ╰─ target := (target.token, target.segment, target.attention_mask)

```
"""
struct BaseColBERT
    bert::Transformers.HuggingFace.HGFBertModel
    linear::Transformers.Layers.Dense
    tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder
end

function BaseColBERT(checkpoint::String, config::ColBERTConfig)
    # since Transformers.jl doesn't support local loading
    # we manually load the linear layer
    bert_config = Transformers.load_config(checkpoint)
    bert_state_dict = HuggingFace.load_state_dict(checkpoint)
    bert_model = HuggingFace.load_model(:bert, checkpoint, :model, bert_state_dict; config = bert_config)
    linear = HuggingFace._load_dense(bert_state_dict, "linear", bert_config.hidden_size, config.doc_settings.dim, bert_config.initializer_range, true)
    tokenizer = Transformers.load_tokenizer(checkpoint)
    
    if config.run_settings.use_gpu
        bert_model = bert_model |> Flux.gpu 
        linear = linear |> Flux.gpu
    end

    BaseColBERT(bert_model, linear, tokenizer)
end

"""
    Checkpoint(model::BaseColBERT, doc_tokenizer::DocTokenizer, config::ColBERTConfig)

A wrapper for [`BaseColBERT`](@ref), which includes a [`ColBERTConfig`](@ref) and tokenization-specific functions via the [`DocTokenizer`](@ref) and [`QueryTokenizer`] types. 

If the config's [`DocSettings`](@ref) are configured to mask punctuations, then the `skiplist` property of the created [`Checkpoint`](@ref) will be set to a list of token IDs of punctuations.

# Arguments
- `model`: The [`BaseColBERT`](@ref) to be wrapped.
- `doc_tokenizer`: A [`DocTokenizer`](@ref) used for functions related to document tokenization. 
- `query_tokenizer`: A [`QueryTokenizer`](@ref) used for functions related to query tokenization. 
- `config`: The underlying [`ColBERTConfig`](@ref). 

# Returns
The created [`Checkpoint`](@ref).

# Examples

Continuing from the example for [`BaseColBERT`](@ref):

```julia-repl
julia> checkPoint = Checkpoint(base_colbert, DocTokenizer(base_colbert.tokenizer, config), QueryTokenizer(base_colbert.tokenizer, config), config)

julia> checkPoint.skiplist              # by default, all punctuations
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
    doc_tokenizer::DocTokenizer
    query_tokenizer::QueryTokenizer
    config::ColBERTConfig
    skiplist::Union{Missing, Vector{Int64}}
end

function Checkpoint(model::BaseColBERT, doc_tokenizer::DocTokenizer, query_tokenizer::QueryTokenizer, config::ColBERTConfig)
    if config.doc_settings.mask_punctuation
        punctuation_list = string.(collect("!\"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"))
        skiplist = [TextEncodeBase.lookup(model.tokenizer.vocab, punct) for punct in punctuation_list]
    else
        skiplist = missing
    end
    Checkpoint(model, doc_tokenizer, query_tokenizer, config, skiplist)
end

"""
    mask_skiplist(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, integer_ids::AbstractMatrix{Int32}, skiplist::Union{Missing, Vector{Int64}})

Create a mask for the given `integer_ids`, based on the provided `skiplist`. 
If the `skiplist` is not missing, then any token IDs in the list will be filtered out along with the padding token.
Otherwise, all tokens are included in the mask.

# Arguments

- `tokenizer`: The underlying tokenizer. 
- `integer_ids`: An `Array` of token IDs for the documents. 
- `skiplist`: A list of token IDs to skip in the mask. 

# Returns
An array of booleans indicating whether the corresponding token ID is included in the mask or not. The array has the same shape as `integer_ids`, i.e `(L, N)`, where `L` is the maximum length of any document in `integer_ids` and `N` is the number of documents.

# Examples

Continuing with the example for [`tensorize`](@ref) and the `skiplist` from the example in [`Checkpoint`](@ref). 

```julia-repl
julia>  mask_skiplist(checkPoint.model.tokenizer, integer_ids, checkPoint.skiplist)
14×4 BitMatrix:
 1  1  1  1
 1  1  1  1
 1  1  1  1
 1  1  1  1
 1  0  0  1
 0  1  0  1
 0  0  0  1
 0  0  0  0
 0  0  0  1
 0  0  0  1
 0  0  0  1
 0  0  0  1
 0  0  0  1
 0  0  0  1

```
"""
function mask_skiplist(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, integer_ids::AbstractMatrix{Int32}, skiplist::Union{Missing, Vector{Int64}})
    filter = integer_ids .!= TextEncodeBase.lookup(tokenizer.vocab, tokenizer.padsym)
    for token_id in skiplist
        filter = filter .& (integer_ids .!= token_id)
    end
    filter
end

"""
    doc(checkpoint::Checkpoint, integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool})

Compute the hidden state of the BERT and linear layers of ColBERT for documents. 

# Arguments

- `checkpoint`: The [`Checkpoint`](@ref) containing the layers to compute the embeddings. 
- `integer_ids`: An array of token IDs to be fed into the BERT model. 
- `integer_mask`: An array of corresponding attention masks. Should have the same shape as `integer_ids`. 

# Returns

A tuple `D, mask`, where:

- `D` is an array containing the normalized embeddings for each token in each document. It has shape `(D, L, N)`, where `D` is the embedding dimension (`128` for the linear layer of ColBERT), and `(L, N)` is the shape of `integer_ids`, i.e `L` is the maximum length of any document and `N` is the total number of documents.
- `mask` is an array containing attention masks for all documents, after masking out any tokens in the `skiplist` of `checkpoint`. It has shape `(1, L, N)`, where `(L, N)` is the same as described above.

# Examples

Continuing from the example in [`tensorize`](@ref) and [`Checkpoint`](@ref):

```julia-repl
julia> D, mask = doc(checkPoint, integer_ids, integer_mask);

julia> mask
1×14×4 BitArray{3}:
[:, :, 1] =
 1  1  1  1  1  0  0  0  0  0  0  0  0  0

[:, :, 2] =
 1  1  1  1  0  1  0  0  0  0  0  0  0  0

[:, :, 3] =
 1  1  1  1  0  0  0  0  0  0  0  0  0  0

[:, :, 4] =
 1  1  1  1  1  1  1  0  1  1  1  1  1  1

```
"""
function doc(checkpoint::Checkpoint, integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool})
    use_gpu = checkpoint.config.run_settings.use_gpu

    if use_gpu 
        integer_ids = integer_ids |> Flux.gpu
        integer_mask = integer_mask|> Flux.gpu
    end

    D = checkpoint.model.bert((token=integer_ids, attention_mask=NeuralAttentionlib.GenericSequenceMask(integer_mask))).hidden_state
    D = checkpoint.model.linear(D)

    mask = mask_skiplist(checkpoint.model.tokenizer, integer_ids, checkpoint.skiplist)
    mask = reshape(mask, (1, size(mask)...))                                        # equivalent of unsqueeze
    @assert isequal(size(mask)[2:end], size(D)[2:end]) "size(mask): $(size(mask)), size(D): $(size(D))"
    @assert mask isa AbstractArray{Bool} "$(typeof(mask))" 

    D = D .* mask                                                                   # clear out embeddings of masked tokens

    if !use_gpu
        # doing this because normalize gives exact results
        D = mapslices(v -> iszero(v) ? v : normalize(v), D, dims = 1)                 # normalize each embedding
    else
        # TODO: try to do some tests to see the gap between this and LinearAlgebra.normalize
        # mapreduce doesn't give exact normalization
        norms = map(sqrt, mapreduce(abs2, +, D, dims = 1))
        norms[norms .== 0] .= 1                                                         # avoid division by 0
        D = D ./ norms
    end

    D, mask
end

"""
    docFromText(checkpoint::Checkpoint, docs::Vector{String}, bsize::Union{Missing, Int})

Get ColBERT embeddings for `docs` using `checkpoint`.

This function also applies ColBERT-style document pre-processing for each document in `docs`.

# Arguments

- `checkpoint`: A [`Checkpoint`](@ref) to be used to compute embeddings.  
- `docs`: A list of documents to get the embeddings for. 
- `bsize`: A batch size for processing documents in batches. 

# Returns

A tuple `embs, doclens`, where `embs` is an array of embeddings and `doclens` is a `Vector` of document lengths. The array `embs` has shape `(D, N)`, where `D` is the embedding dimension (`128` for ColBERT's linear layer) and `N` is the total number of embeddings across all documents in `docs`. 

# Examples

Continuing from the example in [`Checkpoint`](@ref):

```julia-repl
julia> docs = [
    "hello world",
    "thank you!",
    "a",
    "this is some longer text, so length should be longer",
];

julia> embs, doclens = docFromText(checkPoint, docs, config.indexing_settings.index_bsize)
(Float32[0.07590997 0.00056472444 … -0.09958261 -0.03259005; 0.08413661 -0.016337946 … -0.061889287 -0.017708546; … ; -0.11584533 0.016651645 … 0.0073241345 0.09233974; 0.043868616 0.084660925 … -0.0294838 -0.08536169], [5 5 4 13])

julia> embs
128×27 Matrix{Float32}:
  0.07591       0.000564724  …  -0.0811892   -0.0995826   -0.0325901
  0.0841366    -0.0163379       -0.0118506   -0.0618893   -0.0177085
 -0.0301104    -0.0128125        0.0138397   -0.0573847    0.177861
  0.0375673     0.216562        -0.110819     0.00425483  -0.00131543
  0.0252677     0.151702        -0.0272065    0.0350983   -0.0381015
  0.00608629   -0.0415363    …   0.122848     0.0747104    0.0836627
 -0.185256     -0.106582         0.0352982   -0.0405874   -0.064156
 -0.0816655    -0.142809         0.0565001   -0.134649     0.00380807
  0.00471224    0.00444499       0.0112827    0.0253297    0.0665076
 -0.121564     -0.189994         0.0151938   -0.119054    -0.0980481
  0.157599      0.0919844    …   0.0330667    0.0205288    0.0184296
  0.0132481    -0.0430333        0.0404867    0.0575921    0.101702
  0.0695787     0.0281928       -0.0378472   -0.053183    -0.123457
 -0.0933986    -0.0390347        0.0279156    0.0309749    0.00298161
  0.0458561     0.0729707        0.103661     0.00905471   0.127777
  0.00452597    0.05959      …   0.148845     0.0569492    0.293592
  ⋮                          ⋱                ⋮
  0.0510929    -0.138272        -0.00646483  -0.0171806   -0.0618908
  0.128495      0.181198        -0.00408871   0.0274591    0.0343185
 -0.0961544    -0.0223997        0.0117907   -0.0813832    0.038232
  0.0285498     0.0556695    …  -0.0139291   -0.14533     -0.0176019
  0.011212     -0.164717         0.071643    -0.0662124    0.164667
 -0.00178153    0.0600864        0.120243     0.0490749    0.0562548
 -0.0261783     0.0343851        0.0469064    0.040038    -0.0536367
 -0.0696538    -0.020624         0.0441996    0.0842775    0.0567261
 -0.0940356    -0.106123     …   0.00334512   0.00795235  -0.0439883
  0.0567849    -0.0312434       -0.113022     0.0616158   -0.0738149
 -0.0143086     0.105833        -0.142671    -0.0430241   -0.0831739
  0.044704      0.0783603       -0.0413787    0.0315282   -0.171445
  0.129225      0.112544         0.120684     0.107231     0.119762
  0.000207455  -0.124472     …  -0.0930788   -0.0519733    0.0837618
 -0.115845      0.0166516        0.0577464    0.00732413   0.0923397
  0.0438686     0.0846609       -0.0967041   -0.0294838   -0.0853617

julia> doclens
4-element Vector{Int64}:
  5
  5
  4
 13

```
"""
function docFromText(checkpoint::Checkpoint, docs::Vector{String}, bsize::Union{Missing, Int})
    if ismissing(bsize)
        # integer_ids, integer_mask = tensorize(checkpoint.doc_tokenizer, checkpoint.model.tokenizer, docs, bsize)
        # doc(checkpoint, integer_ids, integer_mask)
        error("Currently bsize cannot be missing!")
    else
        text_batches, reverse_indices = tensorize(checkpoint.doc_tokenizer, checkpoint.model.tokenizer, docs, bsize)
        batches = [doc(checkpoint, integer_ids, integer_mask) for (integer_ids, integer_mask) in text_batches]

        # aggregate all embeddings
        D, mask = Vector{AbstractArray{Float32}}(), Vector{BitArray}()
        for (_D, _mask) in batches
            push!(D, _D)
            push!(mask, _mask)
        end

        # concat embeddings and masks, and put them in the original order
        D, mask = cat(D..., dims = 3)[:, :, reverse_indices], cat(mask..., dims = 3)[:, :, reverse_indices]
        mask = reshape(mask, size(mask)[2:end])

        # get doclens, i.e number of attended tokens for each passage
        doclens = vec(sum(mask, dims = 1))

        # flatten out embeddings, i.e get embeddings for each token in each passage
        D = reshape(D, size(D)[1], prod(size(D)[2:end]))

        # remove embeddings for masked tokens
        D = D[:, reshape(mask, prod(size(mask)))]

        @assert ndims(D) == 2 "ndims(D): $(ndims(D))"
        @assert size(D)[2] == sum(doclens) "size(D): $(size(D)), sum(doclens): $(sum(doclens))"
        @assert D isa AbstractMatrix{Float32} "$(typeof(D))"
        @assert doclens isa Vector{Int64} "$(typeof(doclens))"

        D, doclens
    end
end

"""
    query(checkpoint::Checkpoint, integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool})

Compute the hidden state of the BERT and linear layers of ColBERT for queries. 

# Arguments

- `checkpoint`: The [`Checkpoint`](@ref) containing the layers to compute the embeddings. 
- `integer_ids`: An array of token IDs to be fed into the BERT model. 
- `integer_mask`: An array of corresponding attention masks. Should have the same shape as `integer_ids`. 

# Returns

`Q`, where `Q` is an array containing the normalized embeddings for each token in the query matrix. It has shape `(D, L, N)`, where `D` is the embedding dimension (`128` for the linear layer of ColBERT), and `(L, N)` is the shape of `integer_ids`, i.e `L` is the maximum length of any query and `N` is the total number of queries.

# Examples

Continuing from the queries example for [`tensorize`](@ref) and [`Checkpoint`](@ref):

```julia-repl
julia> query(checkPoint, integer_ids, integer_mask)
128×32×1 Array{Float32, 3}:
[:, :, 1] =
  0.0158567    0.169676     0.092745     0.0798617   …   0.115938     0.112977     0.107919
  0.220185     0.0304873    0.165348     0.150315        0.0168762    0.0178042    0.0200357
 -0.00790007  -0.0192251   -0.0852364   -0.0799609      -0.0777439   -0.0776733   -0.0830504
 -0.109909    -0.170906    -0.0138702   -0.0409767      -0.126037    -0.126829    -0.13149
 -0.0231786    0.0532214    0.0607473    0.0279048       0.117017     0.114073     0.108536
  0.0620549    0.0465075    0.0821693    0.0606439   …   0.0150612    0.0133353    0.0126583
 -0.0290509    0.143255     0.0306142    0.042658       -0.164401    -0.161857    -0.160327
  0.0921477    0.0588331    0.250449     0.234636        0.0664076    0.0659837    0.0711357
  0.0279402   -0.0278357    0.144855     0.147958        0.154552     0.155525     0.163634
 -0.0768143   -0.00587305   0.00543038   0.00443374     -0.11757     -0.112495    -0.11112
  ⋮                                                  ⋱                ⋮
 -0.0859686    0.0623054    0.0974813    0.126841        0.0182795    0.0230549    0.031103
  0.0392043    0.0162653    0.0926306    0.104053        0.0491495    0.0484318    0.0438132
 -0.0340363   -0.0278066   -0.0181035   -0.0282369   …  -0.0617945   -0.0631367   -0.0675882
  0.013123     0.0565132   -0.0349061   -0.0464192       0.0724731    0.0780166    0.074623
 -0.117425     0.162483     0.11039      0.136364       -0.00538225  -0.00685449  -0.0019436
 -0.0401158   -0.0045094    0.0539569    0.0689953      -0.00518063  -0.00600252  -0.00771469
  0.0893983    0.0695061   -0.0499409   -0.035411        0.0960932    0.0961893    0.103431
 -0.116265    -0.106331    -0.179832    -0.149728    …  -0.0197172   -0.022061    -0.018135
 -0.0443452   -0.192203    -0.0187912   -0.0247794      -0.0699095   -0.0684749   -0.0662904
  0.100019    -0.0618588    0.106134     0.0989047      -0.0556761   -0.0556784   -0.059571

```
"""
function query(checkpoint::Checkpoint, integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool})
    Q = checkpoint.model.bert((token=integer_ids, attention_mask=NeuralAttentionlib.GenericSequenceMask(integer_mask))).hidden_state
    Q = checkpoint.model.linear(Q)

    # only skip the pad symbol, i.e an empty skiplist
    mask = mask_skiplist(checkpoint.model.tokenizer, integer_ids, Vector{Int}())
    mask = reshape(mask, (1, size(mask)...))                                        # equivalent of unsqueeze
    @assert isequal(size(mask)[2:end], size(Q)[2:end]) "size(mask): $(size(mask)), size(Q): $(size(Q))"

    Q = Q .* mask
    Q = mapslices(v -> iszero(v) ? v : normalize(v), Q, dims = 1)                   # normalize each embedding

    @assert ndims(Q) == 3 "ndims(Q): $(ndims(Q))"
    @assert isequal(size(Q)[2:end], size(integer_ids)) "size(Q): $(size(Q)), size(integer_ids): $(size(integer_ids))"
    @assert Q isa AbstractArray{Float32} "$(typeof(Q))" 

    Q
end

"""
    queryFromText(checkpoint::Checkpoint, queries::Vector{String}, bsize::Union{Missing, Int})

Get ColBERT embeddings for `queries` using `checkpoint`.

This function also applies ColBERT-style query pre-processing for each query in `queries`.

# Arguments

- `checkpoint`: A [`Checkpoint`](@ref) to be used to compute embeddings.  
- `queries`: A list of queries to get the embeddings for. 
- `bsize`: A batch size for processing queries in batches. 

# Returns

`embs`, where `embs` is an array of embeddings. The array `embs` has shape `(D, L, N)`, where `D` is the embedding dimension (`128` for ColBERT's linear layer), `L` is the maximum length of any query in the batch, and `N` is the total number of queries in `queries`. 

# Examples

Continuing from the example in [`Checkpoint`](@ref):

```julia-repl
julia> queries = ["what are white spots on raspberries?"]; 

julia> queryFromText(checkPoint, queries, 128)
128×32×1 Array{Float32, 3}:
[:, :, 1] =
  0.0158567    0.169676     0.092745     0.0798617   …   0.115806     0.115938     0.112977     0.107919
  0.220185     0.0304873    0.165348     0.150315        0.0165188    0.0168762    0.0178042    0.0200357
 -0.00790007  -0.0192251   -0.0852364   -0.0799609      -0.0737461   -0.0777439   -0.0776733   -0.0830504
 -0.109909    -0.170906    -0.0138702   -0.0409767      -0.118738    -0.126037    -0.126829    -0.13149
 -0.0231786    0.0532214    0.0607473    0.0279048       0.111831     0.117017     0.114073     0.108536
  0.0620549    0.0465075    0.0821693    0.0606439   …   0.0148605    0.0150612    0.0133353    0.0126583
 -0.0290509    0.143255     0.0306142    0.042658       -0.169493    -0.164401    -0.161857    -0.160327
  0.0921477    0.0588331    0.250449     0.234636        0.0642578    0.0664076    0.0659837    0.0711357
  0.0279402   -0.0278357    0.144855     0.147958        0.157629     0.154552     0.155525     0.163634
 -0.0768143   -0.00587305   0.00543038   0.00443374     -0.123969    -0.11757     -0.112495    -0.11112
 -0.0184338    0.00668557  -0.191863    -0.161345    …  -0.10374     -0.107664    -0.107267    -0.114564
  ⋮                                                  ⋱                             ⋮
 -0.0859686    0.0623054    0.0974813    0.126841        0.0191363    0.0182795    0.0230549    0.031103
  0.0392043    0.0162653    0.0926306    0.104053        0.0553615    0.0491495    0.0484318    0.0438132
 -0.0340363   -0.0278066   -0.0181035   -0.0282369   …  -0.0562518   -0.0617945   -0.0631367   -0.0675882
  0.013123     0.0565132   -0.0349061   -0.0464192       0.0698766    0.0724731    0.0780166    0.074623
 -0.117425     0.162483     0.11039      0.136364       -0.0050836   -0.00538225  -0.00685449  -0.0019436
 -0.0401158   -0.0045094    0.0539569    0.0689953      -0.00322497  -0.00518063  -0.00600252  -0.00771469
  0.0893983    0.0695061   -0.0499409   -0.035411        0.0964842    0.0960932    0.0961893    0.103431
 -0.116265    -0.106331    -0.179832    -0.149728    …  -0.0275017   -0.0197172   -0.022061    -0.018135
 -0.0443452   -0.192203    -0.0187912   -0.0247794      -0.0735711   -0.0699095   -0.0684749   -0.0662904
  0.100019    -0.0618588    0.106134     0.0989047      -0.0553564   -0.0556761   -0.0556784   -0.059571

```
"""
function queryFromText(checkpoint::Checkpoint, queries::Vector{String}, bsize::Union{Missing, Int})
    if ismissing(bsize)
        error("Currently bsize cannot be missing!")
    end

    # configure the tokenizer to truncate or pad to query_maxlen
    tokenizer = checkpoint.model.tokenizer 
    process = tokenizer.process
    truncpad_pipe = Pipeline{:token}(TextEncodeBase.trunc_or_pad(checkpoint.config.query_settings.query_maxlen, "[PAD]", :tail, :tail), :token)
    process = process[1:4] |> truncpad_pipe |> process[6:end]
    tokenizer = Transformers.TextEncoders.BertTextEncoder(tokenizer.tokenizer, tokenizer.vocab, process; startsym = tokenizer.startsym, endsym = tokenizer.endsym, padsym = tokenizer.padsym, trunc = tokenizer.trunc)

    # get ids and masks, embeddings and returning the concatenated tensors
    batches = tensorize(checkpoint.query_tokenizer, tokenizer, queries, bsize)
    batches = [query(checkpoint, integer_ids, integer_mask) for (integer_ids, integer_mask) in batches]
    Q = cat(batches..., dims=3)

    @assert ndims(Q) == 3 "ndims(Q): $(ndims(Q))"
    @assert Q isa AbstractArray{Float32} "$(typeof(Q))"

    Q
end
