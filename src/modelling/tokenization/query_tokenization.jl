using ...ColBERT: ColBERTConfig

struct QueryTokenizer
    Q_marker_token_id::Int
    config::ColBERTConfig
end
