struct BaseColBERT
    bert::Any
    linear::Any
    tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder
end

struct Checkpoint
    model::BaseColBERT
    doc_tokenizer::Any
    colbert_config::Any
end
