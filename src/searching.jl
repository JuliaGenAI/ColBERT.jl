using .ColBERT: Checkpoint, ColBERTConfig, Collection, IndexScorer

struct Searcher
    config::ColBERTConfig
    checkpoint::Checkpoint
end
