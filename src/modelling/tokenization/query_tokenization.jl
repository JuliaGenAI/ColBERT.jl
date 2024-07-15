using ...ColBERT: ColBERTConfig

struct QueryTokenizer
    Q_marker_token_id::Int
    config::ColBERTConfig
end

function QueryTokenizer(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, config::ColBERTConfig)
    Q_marker_token_id = TextEncodeBase.lookup(tokenizer.vocab, config.tokenizer_settings.query_token_id)
    QueryTokenizer(Q_marker_token_id, config)
end
