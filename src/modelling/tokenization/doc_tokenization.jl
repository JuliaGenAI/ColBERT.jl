using ...ColBERT: ColBERTConfig

struct DocTokenizer
    D_marker_token_id::Int
    config::ColBERTConfig
end

function DocTokenizer(tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, config::ColBERTConfig)
    D_marker_token_id = TextEncodeBase.lookup(tokenizer.vocab, config.tokenizer_settings.doc_token_id)
    DocTokenizer(D_marker_token_id, config)
end

function tensorize(doc_tokenizer::DocTokenizer, tokenizer::Transformers.TextEncoders.AbstractTransformerTextEncoder, batch_text::Vector{String}, bsize::Union{Missing, Int})
    # placeholder for [D] marker token
    batch_text = [". " * doc for doc in batch_text]
    vocabsize = length(tokenizer.vocab.list)

    # getting the integer ids and masks
    encoded_text = Transformers.TextEncoders.encode(tokenizer, batch_text)
    ids, mask = encoded_text.token, encoded_text.attention_mask
    integer_ids = reinterpret(Int32, ids)
    integer_mask = NeuralAttentionlib.getmask(mask, ids)[1, :, :]

    # adding the [D] marker token ID
    integer_ids[2, :] .= doc_tokenizer.D_marker_token_id

    if ismissing(bsize)
        return integer_ids, integer_mask
    else
        integer_ids, integer_mask, reverse_indices = _sort_by_length(integer_ids, integer_mask bsize)
        batches = _split_into_batches(integer_ids, integer_mask, bsize)

        return batches, reverse_indices
    end
end



# tokenizer = base_colbert.tokenizer
# batch_text = [
#     "hello world",
#     "thank you!",
#     "a",
#     "this is some longer text, so length should be longer",
# ]
# bsize = 2
