using ..ColBERT: DocTokenizer, ColBERTConfig

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
