struct Searcher
    config::ColBERTConfig
    bert::HF.HGFBertModel
    linear::Layers.Dense
    tokenizer::TextEncoders.AbstractTransformerTextEncoder
    centroids::Matrix{Float32}
    bucket_cutoffs::Vector{Float32}
    bucket_weights::Vector{Float32}
    ivf::Vector{Int}
    ivf_lengths::Vector{Int}
    doclens::Vector{Int}
    codes::Vector{UInt32}
    residuals::Matrix{UInt8}
    emb2pid::Vector{Int}
    skiplist::Vector{Int}
end

function _build_emb2pid(doclens::Vector{Int})
    num_embeddings = sum(doclens)
    emb2pid = zeros(Int, num_embeddings)
    offset_doclens = 1
    for (pid, dlength) in enumerate(doclens)
        emb2pid[offset_doclens:(offset_doclens + dlength - 1)] .= pid
        offset_doclens += dlength
    end
    emb2pid
end

function Searcher(index_path::String)
    if !isdir(index_path)
        error("Index at $(index_path) does not exist! Please build the index first and try again.")
    end

    @info "Loading config from $(index_path)."
    config = load_config(index_path)

    @info "Loading ColBERT layers from the $(config.checkpoint) HuggingFace checkpoint."
    tokenizer, bert, linear = load_hgf_pretrained_local(config.checkpoint)
    bert = bert |> Flux.gpu
    linear = linear |> Flux.gpu
    # configuring the tokenizer; with query_maxlen - 1
    process = tokenizer.process;
    truncpad_pipe = Pipeline{:token}(
        TextEncodeBase.trunc_or_pad(config.query_maxlen - 1, "[PAD]", :tail, :tail),
        :token);
    process = process[1:4] |> truncpad_pipe |> process[6:end];
    tokenizer = TextEncoders.BertTextEncoder(
        tokenizer.tokenizer, tokenizer.vocab, process; startsym = tokenizer.startsym,
        endsym = tokenizer.endsym, padsym = tokenizer.padsym, trunc = tokenizer.trunc);

    @info "Loading codec."
    codec = load_codec(index_path)
    ivf = JLD2.load_object(joinpath(index_path, "ivf.jld2"))
    ivf_lengths = JLD2.load_object(joinpath(index_path, "ivf_lengths.jld2"))
    doclens = load_doclens(index_path)
    codes, residuals = load_compressed_embs(index_path)

    @info "Building the emb2pid mapping."
    emb2pid = _build_emb2pid(doclens)

    # by default, only include the pad symbol in the skiplist
    skiplist = [lookup(tokenizer.vocab, tokenizer.padsym)]

    Searcher(
        config,
        bert,
        linear,
        tokenizer,
        codec["centroids"],
        codec["bucket_cutoffs"],
        codec["bucket_weights"],
        ivf,
        ivf_lengths,
        doclens,
        codes,
        residuals,
        emb2pid,
        skiplist
    )
end

function search(searcher::Searcher, query::String, k::Int)
    Q = encode_query(searcher.config, searcher.checkpoint, query)

    if size(Q)[3] > 1
        error("Only one query is supported at the moment!")
    end
    @assert size(Q)[3]==1 "size(Q): $(size(Q))"
    @assert isequal(size(Q)[2], searcher.config.query_maxlen)
    "size(Q): $(size(Q)), query_maxlen: $(searcher.config.query_maxlen)"     # Q: (128, 32, 1)

    Q = reshape(Q, size(Q)[1:end .!= end]...)           # squeeze out the last dimension 
    @assert isequal(length(size(Q)), 2) "size(Q): $(size(Q))"

    pids = retrieve(searcher.ivf, searcher.ivf_lengths, searcher.centroids,
        searcher.emb2pid, searcher.config.nprobe, Q)
    scores = score_pids(
        searcher.config, searcher.centroids, searcher.bucket_weights,
        searcher.doclens, searcher.codes, searcher.residuals, Q, pids)

    indices = sortperm(scores, rev = true)
    pids, scores = pids[indices], scores[indices]
    pids[1:k], scores[1:k]
end
