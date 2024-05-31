using ...ColBERT: ColBERTConfig

struct DocTokenizer
    D_marker_token_id::Int
    config::ColBERTConfig
end

function DocTokenizer(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, config::ColBERTConfig)
    D_marker_token_id = TextEncodeBase.lookup(tokenizer.vocab, config.tokenizer_settings.doc_token_id)
    DocTokenizer(D_marker_token_id, config)
end
