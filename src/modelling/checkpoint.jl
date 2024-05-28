struct BaseColBERT
    bert
    linear
    tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder
end

struct Checkpoint
    model::BaseColBERT
    doc_tokenizer
    colbert_config
end
