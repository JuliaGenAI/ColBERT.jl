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
