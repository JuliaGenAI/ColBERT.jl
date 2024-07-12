using ..ColBERT: DocTokenizer, ColBERTConfig

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
    BaseColBERT(bert_model, linear, tokenizer)
end

"""
    Checkpoint(model::BaseColBERT, doc_tokenizer::DocTokenizer, colbert_config::ColBERTConfig)

A wrapper for [`BaseColBERT`](@ref), which includes a [`ColBERTConfig`](@ref) and tokenization-specific functions via the [`DocTokenizer`](@ref) type. 

If the config's [`DocSettings`](@ref) are configured to mask punctuations, then the `skiplist` property of the created [`Checkpoint`](@ref) will be set to a list of token IDs of punctuations.

# Arguments
- `model`: The [`BaseColBERT`](@ref) to be wrapped.
- `doc_tokenizer`: A [`DocTokenizer`](@ref) used for functions related to document tokenization. 
- `colbert_config`: The underlying [`ColBERTConfig`](@ref). 

# Returns
The created [`Checkpoint`](@ref).

# Examples

Continuing from the example for [`BaseColBERT`](@ref):

```julia-repl
julia> checkPoint = Checkpoint(base_colbert, DocTokenizer(base_colbert.tokenizer, config), config);

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
    colbert_config::ColBERTConfig
    skiplist::Union{Missing, Vector{Int}}
end

function Checkpoint(model::BaseColBERT, doc_tokenizer::DocTokenizer, colbert_config::ColBERTConfig)
    if colbert_config.doc_settings.mask_punctuation
        punctuation_list = string.(collect("!\"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"))
        skiplist = [TextEncodeBase.lookup(model.tokenizer.vocab, punct) for punct in punctuation_list]
    else
        skiplist = missing
    end
    Checkpoint(model, doc_tokenizer, colbert_config, skiplist)
end

"""
    mask_skiplist(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, integer_ids::AbstractArray, skiplist::Union{Missing, Vector{Int}})

Create a mask for the given `integer_ids`, based on the provided `skiplist`. 
If the `skiplist` is not missing, then any token ids in the list will be filtered out.
Otherwise, all tokens are included in the mask.

# Arguments

- `tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder`: The text encoder used to transform the input text into integer ids. 
- `integer_ids::AbstractArray`: An array of integers representing the encoded tokens. 
- `skiplist::Union{Missing, Vector{Int}}`: A list of token ids to skip in the mask. If missing, all tokens are included.

# Returns
An array of booleans indicating whether each token id is included in the mask or not.
"""
function mask_skiplist(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, integer_ids::AbstractArray, skiplist::Union{Missing, Vector{Int}})
    if !ismissing(skiplist)
        filter = token_id -> !(token_id in skiplist) && token_id != TextEncodeBase.lookup(tokenizer.vocab, tokenizer.padsym)
    else
        filter = token_id -> true
    end
    filter.(integer_ids)
end

function doc(checkpoint::Checkpoint, integer_ids::AbstractArray, integer_mask::AbstractArray)
    D = checkpoint.model.bert((token=integer_ids, attention_mask=NeuralAttentionlib.GenericSequenceMask(integer_mask))).hidden_state
    D = checkpoint.model.linear(D)

    mask = mask_skiplist(checkpoint.model.tokenizer, integer_ids, checkpoint.skiplist)
    mask = reshape(mask, (1, size(mask)...))                                        # equivalent of unsqueeze

    D = D .* mask                                                                   # clear out embeddings of masked tokens
    D = mapslices(v -> iszero(v) ? v : normalize(v), D, dims = 1)                   # normalize each embedding
    D, mask
end

function docFromText(checkpoint::Checkpoint, docs::Vector{String}, bsize::Union{Missing, Int})
    if ismissing(bsize)
        integer_ids, integer_mask = tensorize(checkpoint.doc_tokenizer, checkpoint.model.tokenizer, docs, bsize)
        doc(checkpoint, integer_ids, integer_mask)
    else
        text_batches, reverse_indices = tensorize(checkpoint.doc_tokenizer, checkpoint.model.tokenizer, docs, bsize)
        batches = [doc(checkpoint, integer_ids, integer_mask) for (integer_ids, integer_mask) in text_batches]

        # aggregate all embeddings
        D, mask = [], []
        for (_D, _mask) in batches
            push!(D, _D)
            push!(mask, _mask)
        end

        # concat embeddings and masks, and put them in the original order
        D, mask = cat(D..., dims = 3)[:, :, reverse_indices], cat(mask..., dims = 3)[:, :, reverse_indices]
        mask = reshape(mask, size(mask)[2:end])

        # get doclens, i.e number of attended tokens for each passage
        doclens = sum(mask, dims = 1)

        # flatten out embeddings, i.e get embeddings for each token in each passage
        D = reshape(D, size(D)[1], prod(size(D)[2:end]))

        # remove embeddings for masked tokens
        D = D[:, reshape(mask, prod(size(mask)))]

        D, doclens
    end
end
