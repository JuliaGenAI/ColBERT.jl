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

"""
    query(
        config::ColBERTConfig, checkpoint::Checkpoint, integer_ids::AbstractMatrix{Int32},
        integer_mask::AbstractMatrix{Bool})

Compute the hidden state of the BERT and linear layers of ColBERT for queries.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) to be used.
  - `checkpoint`: The [`Checkpoint`](@ref) containing the layers to compute the embeddings.
  - `integer_ids`: An array of token IDs to be fed into the BERT model.
  - `integer_mask`: An array of corresponding attention masks. Should have the same shape as `integer_ids`.

# Returns

`Q`, where `Q` is an array containing the normalized embeddings for each token in the query matrix.
It has shape `(D, L, N)`, where `D` is the embedding dimension (`128` for the linear layer of ColBERT),
and `(L, N)` is the shape of `integer_ids`, i.e `L` is the maximum length of any query and `N` is
the total number of queries.

# Examples

Continuing from the queries example for [`tensorize_queries`](@ref) and [`Checkpoint`](@ref):

```julia-repl
julia> ColBERT.query(config, checkpoint, integer_ids, integer_mask)[:, :, 1]
128×32×1 CuArray{Float32, 3, CUDA.DeviceMemory}:
[:, :, 1] =
  0.0158568    0.169676     0.092745      0.0798617     0.153079    …   0.117006     0.115806     0.115938      0.112977      0.107919
  0.220185     0.0304873    0.165348      0.150315     -0.0116249       0.0173332    0.0165187    0.0168762     0.0178042     0.0200356
 -0.00790017  -0.0192251   -0.0852365    -0.0799609    -0.0465292      -0.0693319   -0.0737462   -0.0777439    -0.0776733    -0.0830504
 -0.109909    -0.170906    -0.0138701    -0.0409766    -0.177391       -0.113141    -0.118738    -0.126037     -0.126829     -0.13149
 -0.0231787    0.0532214    0.0607473     0.0279048     0.0634681       0.112296     0.111831     0.117017      0.114073      0.108536
  0.0620549    0.0465075    0.0821693     0.0606439     0.0592031   …   0.0167847    0.0148605    0.0150612     0.0133353     0.0126583
 -0.0290508    0.143255     0.0306142     0.0426579     0.129972       -0.17502     -0.169493    -0.164401     -0.161857     -0.160327
  0.0921475    0.058833     0.250449      0.234636      0.0412965       0.0590262    0.0642577    0.0664076     0.0659837     0.0711358
  0.0279402   -0.0278357    0.144855      0.147958     -0.0268559       0.161106     0.157629     0.154552      0.155525      0.163634
 -0.0768143   -0.00587302   0.00543038    0.00443376   -0.0134111      -0.126912    -0.123969    -0.11757      -0.112495     -0.11112
 -0.0184337    0.00668561  -0.191863     -0.161345      0.0222466   …  -0.103246    -0.10374     -0.107664     -0.107267     -0.114564
  0.0112104    0.0214651   -0.0923963    -0.0823052     0.0600248       0.103589     0.103387     0.106261      0.105065      0.10409
  0.110971     0.272576     0.148319      0.143233      0.239578        0.11224      0.107913     0.109914      0.112652      0.108365
 -0.131066     0.0376254   -0.0164237    -0.000193318   0.00344707     -0.0893371   -0.0919217   -0.0969305    -0.0935498    -0.096145
 -0.0402605    0.0350559    0.0162864     0.0269105     0.00968855     -0.0623393   -0.0670097   -0.070679     -0.0655848    -0.0564059
  0.0799973    0.0482302    0.0712078     0.0792903     0.0108783   …   0.00820444   0.00854873   0.00889943    0.00932721    0.00751066
 -0.137565    -0.0369116   -0.065728     -0.0664102    -0.0238012       0.029041     0.0292468    0.0297059     0.0278639     0.0257616
  0.0479746   -0.102338    -0.0557072    -0.0833976    -0.0979401      -0.057629    -0.053911    -0.0566325    -0.0568765    -0.0581378
  0.0656851    0.0195639    0.0288789     0.0559219     0.0315515       0.0472323    0.054771     0.0596156     0.0541802     0.0525933
  0.0668634   -0.00400549   0.0297102     0.0505045    -0.00082792      0.0414113    0.0400276    0.0361149     0.0325914     0.0260693
 -0.0691096    0.0348577   -0.000312685   0.0232462    -0.00250495  …  -0.141874    -0.142026    -0.132163     -0.129679     -0.131122
 -0.0273036    0.0653352    0.0332689     0.017918      0.0875479       0.0500921    0.0471914    0.0469949     0.0434268     0.0442646
 -0.0981665   -0.0296463   -0.0114686    -0.0348033    -0.0468719      -0.0772672   -0.0805913   -0.0809244    -0.0823798    -0.081472
  ⋮                                                                 ⋱                                           ⋮
  0.0506199    0.00290888   0.047947      0.063503     -0.0072114       0.0360347    0.0326486    0.033966      0.0327732     0.0261081
 -0.0288586   -0.150171    -0.0699125    -0.108002     -0.142865       -0.0775934   -0.072192    -0.0697569    -0.0715358    -0.0683193
 -0.0646991    0.0724608   -0.00767811   -0.0184348     0.0524162       0.0457267    0.0532778    0.0649795     0.0697126     0.0808413
  0.0445508    0.0296366    0.0325647     0.0521935     0.0436496       0.129031     0.126605     0.12324       0.120497      0.117703
 -0.127301    -0.0224252   -0.00579415   -0.00877803   -0.0140665   …  -0.080026    -0.080839    -0.0823464    -0.0803394    -0.0856279
  0.0304881    0.0396951    0.0798097     0.0736797     0.0800866       0.0426674    0.0411406    0.0460205     0.0460111     0.0532082
  0.0488798    0.252244     0.0866849     0.098552      0.251561       -0.0236942   -0.035116    -0.0395483    -0.0463498    -0.0494207
 -0.0296798   -0.0494761    0.00688248    0.0264166    -0.0352487      -0.0476357   -0.0435388   -0.0404835    -0.0410673    -0.0367272
  0.023548    -0.00147361   0.0629259     0.106951      0.0406627       0.00627022   0.00403014  -0.000107777  -0.000898423   0.00296315
 -0.0574151   -0.0875744   -0.103787     -0.114166     -0.103979    …  -0.0708782   -0.0700138   -0.0687795    -0.070967     -0.0636385
  0.0280373    0.149767    -0.0899733    -0.0732524     0.162316        0.022177     0.0183834    0.0201251     0.0197228     0.0219051
 -0.0617143   -0.0573989   -0.0973785    -0.0805046    -0.0525925       0.0997715    0.102691     0.107432      0.108591      0.109502
 -0.0859687    0.0623054    0.0974813     0.126841      0.0595557       0.0187937    0.0191363    0.0182794     0.0230548     0.031103
  0.0392044    0.0162653    0.0926306     0.104054      0.0509464       0.0559883    0.0553617    0.0491496     0.0484319     0.0438133
 -0.0340362   -0.0278067   -0.0181035    -0.0282369    -0.0490531   …  -0.0564175   -0.0562518   -0.0617946    -0.0631367    -0.0675882
  0.0131229    0.0565131   -0.0349061    -0.0464192     0.0456515       0.0676478    0.0698765    0.0724731     0.0780165     0.0746229
 -0.117425     0.162483     0.11039       0.136364      0.135339       -0.00432259  -0.00508357  -0.00538224   -0.00685447   -0.00194357
 -0.0401157   -0.00450943   0.0539568     0.0689953    -0.00295334     -0.00671544  -0.00322498  -0.00518066   -0.00600254   -0.0077147
  0.0893984    0.0695061   -0.049941     -0.035411      0.0767663       0.0913505    0.0964841    0.0960931     0.0961892     0.103431
 -0.116265    -0.106331    -0.179832     -0.149728     -0.0913282   …  -0.0287848   -0.0275017   -0.0197172    -0.0220611    -0.018135
 -0.0443452   -0.192203    -0.0187912    -0.0247794    -0.180245       -0.0780865   -0.073571    -0.0699094    -0.0684748    -0.0662903
  0.100019    -0.0618588    0.106134      0.0989047    -0.0885639      -0.0547317   -0.0553563   -0.055676     -0.0556784    -0.0595709
```
"""
function query(
        config::ColBERTConfig, checkpoint::Checkpoint, integer_ids::AbstractMatrix{Int32},
        integer_mask::AbstractMatrix{Bool})
    integer_ids = integer_ids |> Flux.gpu
    integer_mask = integer_mask |> Flux.gpu

    Q = checkpoint.model.bert((token = integer_ids,
        attention_mask = NeuralAttentionlib.GenericSequenceMask(integer_mask))).hidden_state
    Q = checkpoint.model.linear(Q)

    # only skip the pad symbol, i.e an empty skiplist
    mask = mask_skiplist(
        checkpoint.model.tokenizer, integer_ids, Vector{Int64}())
    mask = reshape(mask, (1, size(mask)...))                                        # equivalent of unsqueeze
    @assert isequal(size(mask)[2:end], size(Q)[2:end])
    "size(mask): $(size(mask)), size(Q): $(size(Q))"
    @assert mask isa AbstractArray{Bool} "$(typeof(mask))"

    Q = Q .* mask

    if !config.use_gpu
        # doing this because normalize gives exact results
        Q = mapslices(v -> iszero(v) ? v : normalize(v), Q, dims = 1)                 # normalize each embedding
    else
        # TODO: try to do some tests to see the gap between this and LinearAlgebra.normalize
        # mapreduce doesn't give exact normalization
        norms = map(sqrt, mapreduce(abs2, +, Q, dims = 1))
        norms[norms .== 0] .= 1                                                         # avoid division by 0
        @assert isequal(size(norms)[2:end], size(Q)[2:end])
        @assert size(norms)[1] == 1

        Q = Q ./ norms
    end

    @assert ndims(Q)==3 "ndims(Q): $(ndims(Q))"
    @assert isequal(size(Q)[2:end], size(integer_ids))
    "size(Q): $(size(Q)), size(integer_ids): $(size(integer_ids))"
    @assert Q isa AbstractArray{Float32} "$(typeof(Q))"

    Q
end

"""
    queryFromText(config::ColBERTConfig,
        checkpoint::Checkpoint, queries::Vector{String}, bsize::Union{Missing, Int})

Get ColBERT embeddings for `queries` using `checkpoint`.

This function also applies ColBERT-style query pre-processing for each query in `queries`.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) to be used.
  - `checkpoint`: A [`Checkpoint`](@ref) to be used to compute embeddings.
  - `queries`: A list of queries to get the embeddings for.
  - `bsize`: A batch size for processing queries in batches.

# Returns

`embs`, where `embs` is an array of embeddings. The array `embs` has shape `(D, L, N)`,
where `D` is the embedding dimension (`128` for ColBERT's linear layer), `L` is the
maximum length of any query in the batch, and `N` is the total number of queries in `queries`.

# Examples

Continuing from the example in [`Checkpoint`](@ref):

```julia-repl
julia> queries = ["what are white spots on raspberries?"];

julia> ColBERT.queryFromText(config, checkpoint, queries, 128)
128×32×1 Array{Float32, 3}:
[:, :, 1] =
  0.0158568    0.169676     0.092745      0.0798617     0.153079    …   0.117734     0.117006     0.115806     0.115938      0.112977      0.107919
  0.220185     0.0304873    0.165348      0.150315     -0.0116249       0.0181126    0.0173332    0.0165187    0.0168762     0.0178042     0.0200356
 -0.00790017  -0.0192251   -0.0852365    -0.0799609    -0.0465292      -0.0672796   -0.0693319   -0.0737462   -0.0777439    -0.0776733    -0.0830504
 -0.109909    -0.170906    -0.0138701    -0.0409766    -0.177391       -0.10489     -0.113141    -0.118738    -0.126037     -0.126829     -0.13149
 -0.0231787    0.0532214    0.0607473     0.0279048     0.0634681       0.113961     0.112296     0.111831     0.117017      0.114073      0.108536
  0.0620549    0.0465075    0.0821693     0.0606439     0.0592031   …   0.0174852    0.0167847    0.0148605    0.0150612     0.0133353     0.0126583
 -0.0290508    0.143255     0.0306142     0.0426579     0.129972       -0.175238    -0.17502     -0.169493    -0.164401     -0.161857     -0.160327
  0.0921475    0.058833     0.250449      0.234636      0.0412965       0.0555153    0.0590262    0.0642577    0.0664076     0.0659837     0.0711358
  0.0279402   -0.0278357    0.144855      0.147958     -0.0268559       0.162062     0.161106     0.157629     0.154552      0.155525      0.163634
 -0.0768143   -0.00587302   0.00543038    0.00443376   -0.0134111      -0.128129    -0.126912    -0.123969    -0.11757      -0.112495     -0.11112
 -0.0184337    0.00668561  -0.191863     -0.161345      0.0222466   …  -0.102283    -0.103246    -0.10374     -0.107664     -0.107267     -0.114564
  0.0112104    0.0214651   -0.0923963    -0.0823052     0.0600248       0.103233     0.103589     0.103387     0.106261      0.105065      0.10409
  0.110971     0.272576     0.148319      0.143233      0.239578        0.109759     0.11224      0.107913     0.109914      0.112652      0.108365
 -0.131066     0.0376254   -0.0164237    -0.000193318   0.00344707     -0.0862689   -0.0893371   -0.0919217   -0.0969305    -0.0935498    -0.096145
 -0.0402605    0.0350559    0.0162864     0.0269105     0.00968855     -0.0587467   -0.0623393   -0.0670097   -0.070679     -0.0655848    -0.0564059
  0.0799973    0.0482302    0.0712078     0.0792903     0.0108783   …   0.00501423   0.00820444   0.00854873   0.00889943    0.00932721    0.00751066
 -0.137565    -0.0369116   -0.065728     -0.0664102    -0.0238012       0.0250844    0.029041     0.0292468    0.0297059     0.0278639     0.0257616
  0.0479746   -0.102338    -0.0557072    -0.0833976    -0.0979401      -0.0583169   -0.057629    -0.053911    -0.0566325    -0.0568765    -0.0581378
  0.0656851    0.0195639    0.0288789     0.0559219     0.0315515       0.03907      0.0472323    0.054771     0.0596156     0.0541802     0.0525933
  0.0668634   -0.00400549   0.0297102     0.0505045    -0.00082792      0.0399623    0.0414113    0.0400276    0.0361149     0.0325914     0.0260693
 -0.0691096    0.0348577   -0.000312685   0.0232462    -0.00250495  …  -0.146082    -0.141874    -0.142026    -0.132163     -0.129679     -0.131122
 -0.0273036    0.0653352    0.0332689     0.017918      0.0875479       0.0535029    0.0500921    0.0471914    0.0469949     0.0434268     0.0442646
 -0.0981665   -0.0296463   -0.0114686    -0.0348033    -0.0468719      -0.0741133   -0.0772672   -0.0805913   -0.0809244    -0.0823798    -0.081472
 -0.0262739    0.109895     0.0117273     0.0222689     0.100869        0.0119844    0.0132486    0.012956     0.0175875     0.013171      0.0195091
  0.0861164    0.0799029    0.00381147    0.0170927     0.103322        0.0238912    0.0209658    0.0226638    0.0209905     0.0230679     0.0221191
  0.125112     0.0880232    0.0351989     0.022897      0.0862715   …  -0.0219898   -0.0238914   -0.0207844   -0.0229276    -0.0238033    -0.0236367
  ⋮                                                                 ⋱                                                        ⋮
 -0.158838     0.0415251   -0.0584126    -0.0373528     0.0819274      -0.212757    -0.214835    -0.213414    -0.212899     -0.215478     -0.210674
 -0.039636    -0.0837763   -0.0837142    -0.0597521    -0.0868467       0.0309127    0.0339911    0.03399      0.0313526     0.0316408     0.0309661
  0.0755214    0.0960326    0.0858578     0.0614626     0.111979    …   0.102411     0.101302     0.108277     0.109034      0.107593      0.111863
  0.0506199    0.00290888   0.047947      0.063503     -0.0072114       0.0388324    0.0360347    0.0326486    0.033966      0.0327732     0.0261081
 -0.0288586   -0.150171    -0.0699125    -0.108002     -0.142865       -0.0811611   -0.0775934   -0.072192    -0.0697569    -0.0715358    -0.0683193
 -0.0646991    0.0724608   -0.00767811   -0.0184348     0.0524162       0.046386     0.0457267    0.0532778    0.0649795     0.0697126     0.0808413
  0.0445508    0.0296366    0.0325647     0.0521935     0.0436496       0.125633     0.129031     0.126605     0.12324       0.120497      0.117703
 -0.127301    -0.0224252   -0.00579415   -0.00877803   -0.0140665   …  -0.0826691   -0.080026    -0.080839    -0.0823464    -0.0803394    -0.0856279
  0.0304881    0.0396951    0.0798097     0.0736797     0.0800866       0.0448139    0.0426674    0.0411406    0.0460205     0.0460111     0.0532082
  0.0488798    0.252244     0.0866849     0.098552      0.251561       -0.0212669   -0.0236942   -0.035116    -0.0395483    -0.0463498    -0.0494207
 -0.0296798   -0.0494761    0.00688248    0.0264166    -0.0352487      -0.0486577   -0.0476357   -0.0435388   -0.0404835    -0.0410673    -0.0367272
  0.023548    -0.00147361   0.0629259     0.106951      0.0406627       0.00599323   0.00627022   0.00403014  -0.000107777  -0.000898423   0.00296315
 -0.0574151   -0.0875744   -0.103787     -0.114166     -0.103979    …  -0.0697383   -0.0708782   -0.0700138   -0.0687795    -0.070967     -0.0636385
  0.0280373    0.149767    -0.0899733    -0.0732524     0.162316        0.0233808    0.022177     0.0183834    0.0201251     0.0197228     0.0219051
 -0.0617143   -0.0573989   -0.0973785    -0.0805046    -0.0525925       0.0936075    0.0997715    0.102691     0.107432      0.108591      0.109502
 -0.0859687    0.0623054    0.0974813     0.126841      0.0595557       0.0244905    0.0187937    0.0191363    0.0182794     0.0230548     0.031103
  0.0392044    0.0162653    0.0926306     0.104054      0.0509464       0.0516558    0.0559883    0.0553617    0.0491496     0.0484319     0.0438133
 -0.0340362   -0.0278067   -0.0181035    -0.0282369    -0.0490531   …  -0.0528032   -0.0564175   -0.0562518   -0.0617946    -0.0631367    -0.0675882
  0.0131229    0.0565131   -0.0349061    -0.0464192     0.0456515       0.0670016    0.0676478    0.0698765    0.0724731     0.0780165     0.0746229
 -0.117425     0.162483     0.11039       0.136364      0.135339       -0.00589512  -0.00432259  -0.00508357  -0.00538224   -0.00685447   -0.00194357
 -0.0401157   -0.00450943   0.0539568     0.0689953    -0.00295334     -0.0122461   -0.00671544  -0.00322498  -0.00518066   -0.00600254   -0.0077147
  0.0893984    0.0695061   -0.049941     -0.035411      0.0767663       0.0880484    0.0913505    0.0964841    0.0960931     0.0961892     0.103431
 -0.116265    -0.106331    -0.179832     -0.149728     -0.0913282   …  -0.0318565   -0.0287848   -0.0275017   -0.0197172    -0.0220611    -0.018135
 -0.0443452   -0.192203    -0.0187912    -0.0247794    -0.180245       -0.0800835   -0.0780865   -0.073571    -0.0699094    -0.0684748    -0.0662903
  0.100019    -0.0618588    0.106134      0.0989047    -0.0885639      -0.0577217   -0.0547317   -0.0553563   -0.055676     -0.0556784    -0.0595709
```
"""
function queryFromText(config::ColBERTConfig,
        checkpoint::Checkpoint, queries::Vector{String}, bsize::Union{
            Missing, Int})
    if ismissing(bsize)
        error("Currently bsize cannot be missing!")
    end

    # configure the tokenizer to truncate or pad to query_maxlen
    tokenizer = checkpoint.model.tokenizer
    process = tokenizer.process
    truncpad_pipe = Pipeline{:token}(
        TextEncodeBase.trunc_or_pad(
            config.query_maxlen, "[PAD]", :tail, :tail),
        :token)
    process = process[1:4] |> truncpad_pipe |> process[6:end]
    tokenizer = Transformers.TextEncoders.BertTextEncoder(
        tokenizer.tokenizer, tokenizer.vocab, process; startsym = tokenizer.startsym,
        endsym = tokenizer.endsym, padsym = tokenizer.padsym, trunc = tokenizer.trunc)

    # get ids and masks, embeddings and returning the concatenated tensors
    integer_ids, integer_mask = tensorize_queries(config, tokenizer, queries)

    # aggregate all embeddings
    Q = Vector{AbstractArray{Float32}}()
    for query_offset in 1:bsize:length(queries)
        query_end_offset = min(length(queries), query_offset + bsize - 1)
        Q_ = query(
            config, checkpoint, integer_ids[:, query_offset:query_end_offset],
            integer_mask[:, query_offset:query_end_offset])
        push!(Q, Q_)
        Q_ = nothing
    end
    Q = cat(Q..., dims = 3)

    @assert ndims(Q)==3 "ndims(Q): $(ndims(Q))"
    @assert Q isa AbstractArray{Float32} "$(typeof(Q))"

    Flux.cpu(Q)
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
           TextEncodeBase.trunc_or_pad(doc_maxlen - 1, "[PAD]", :tail, :tail),
           :token);

julia> process = process[1:4] |> truncpad_pipe |> process[6:end];

julia> tokenizer = TextEncoders.BertTextEncoder(
           tokenizer.tokenizer, tokenizer.vocab, process; startsym = tokenizer.startsym,
           endsym = tokenizer.endsym, padsym = tokenizer.padsym, trunc = tokenizer.trunc);

julia> bert = bert |> Flux.gpu;

julia> linear = linear |> Flux.gpu;

julia> passages = readlines("./downloads/lotte/lifestyle/dev/collection.tsv")[1:50000];

julia> punctuations_and_padsym = [string.(collect("!\"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"));
                                   tokenizer.padsym];

julia> skiplist = [lookup(tokenizer.vocab, sym)
                    for sym in punctuations_and_padsym];

julia> @time encode_passages(bert, linear, tokenizer, passages, dim, index_bsize, doc_token, skiplist)  

julia> passages = [
    "hello world",
    "thank you!",
    "a",
    "this is some longer text, so length should be longer",
];

julia> @time embs, doclen = encode_passages(bert, linear, tokenizer, passages, dim, index_bsize, doc_token, skiplist) 
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
        D = doc(bert, linear, integer_ids, bitmask)                 # (dim, doc_maxlen, current_batch_size)
        mask = _clear_masked_embeddings!(D, integer_ids, skiplist)  # (1, doc_maxlen, current_batch_size)

        # normalize each embedding in D; along dims = 1
        _normalize_array!(D, dims = 1)

        # get the doclens by unsqueezing the mask
        mask = reshape(mask, size(mask)[2:end])                     # (doc_maxlen, current_batch_size)
        doclens_ = vec(sum(mask, dims = 1))

        # flatten out embeddings, i.e get embeddings for each token in each passage
        D = _flatten_embeddings(D)                                  # (dim, total_num_embeddings)

        # remove embeddings for masked tokens
        D = _remove_masked_tokens(D, mask)                          # (dim, total_num_masked_embeddings)

        @assert ndims(D)==2 "ndims(D): $(ndims(D))"
        @assert size(D, 1) == dim "size(D): $(size(D)), dim: $(dim)"
        @assert size(D, 2)==sum(doclens_) "size(D): $(size(D)), sum(doclens): $(sum(doclens_))"
        @assert D isa AbstractMatrix{Float32} "$(typeof(D))"
        @assert doclens_ isa AbstractVector{Int64} "$(typeof(doclens_))"

        push!(embs, Flux.cpu(D))
        append!(doclens, Flux.cpu(doclens_))
    end
    embs = cat(embs..., dims = 2)
    embs, doclens
end
