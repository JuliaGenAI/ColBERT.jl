using ..ColBERT: DocTokenizer, ColBERTConfig

struct BaseColBERT
    bert::Transformers.HuggingFace.HGFBertModel
    linear::Transformers.Layers.Dense
    tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder
end

"""
    BaseColBERT(; bert::Transformers.HuggingFace.HGFBertModel, linear::Transformers.Layers.Dense, tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder)

A struct representing the BERT model, linear layer, and the tokenizer used to compute embeddings for documents and queries.

# Arguments
- `bert`: The pre-trained BERT model used to generate the embeddings.
- `linear`: The linear layer used to project the embeddings to a specific dimension.
- `tokenizer`: The tokenizer to used by the BERT model.

# Returns

A [`BaseColBERT`](@ref) object.
"""
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
