struct Searcher
    config::ColBERTConfig
    bert::HF.HGFBertModel
    linear::Layers.Dense
    tokenizer::TextEncoders.AbstractTransformerTextEncoder
    centroids::AbstractMatrix{Float32}
    bucket_cutoffs::AbstractVector{Float32}
    bucket_weights::AbstractVector{Float32}
    ivf::Vector{Int}
    ivf_lengths::Vector{Int}
    doclens::Vector{Int}
    codes::Vector{UInt32}
    residuals::Matrix{UInt8}
    emb2pid::Vector{Int}
    skiplist::Vector{Int}
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
    process = tokenizer.process
    truncpad_pipe = Pipeline{:token}(
        TextEncodeBase.trunc_or_pad(
            config.query_maxlen - 1, "[PAD]", :tail, :tail),
        :token)
    process = process[1:4] |> truncpad_pipe |> process[6:end]
    tokenizer = TextEncoders.BertTextEncoder(
        tokenizer.tokenizer, tokenizer.vocab, process; startsym = tokenizer.startsym,
        endsym = tokenizer.endsym, padsym = tokenizer.padsym, trunc = tokenizer.trunc)

    # loading the codec
    @info "Loading codec."
    codec = load_codec(index_path)
    codec["centroids"] = codec["centroids"] |> Flux.gpu
    codec["bucket_cutoffs"] = codec["bucket_cutoffs"] |> Flux.gpu
    codec["bucket_weights"] = codec["bucket_weights"] |> Flux.gpu

    # loading the ivf
    ivf = JLD2.load_object(joinpath(index_path, "ivf.jld2"))
    ivf_lengths = JLD2.load_object(joinpath(index_path, "ivf_lengths.jld2"))

    # loading the doclens and compressed embeddings
    doclens = load_doclens(index_path)
    codes, residuals = load_compressed_embs(index_path)

    # building emb2pid
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

function _build_emb2pid(doclens::Vector{Int})
    num_embeddings = sum(doclens)
    emb2pid = zeros(Int, num_embeddings)
    embs2pid_offsets = cumsum([1; _head(doclens)])
    for (pid, dlength) in enumerate(doclens)
        offset = embs2pid_offsets[pid]
        emb2pid[offset:(offset + dlength - 1)] .= pid
    end
    emb2pid
end

function search(searcher::Searcher, query::String, k::Int)
    Q = encode_queries(searcher.bert, searcher.linear,
        searcher.tokenizer, [query], searcher.config.dim,
        searcher.config.index_bsize, searcher.config.query_token,
        searcher.config.attend_to_mask_tokens, searcher.skiplist)
    @assert size(Q, 3)==1 "size(Q): $(size(Q))"
    @assert(isequal(size(Q, 2), searcher.config.query_maxlen),
        "size(Q): $(size(Q)), query_maxlen: $(searcher.config.query_maxlen)")

    # squeeze out last dim and move to gpu
    Q = reshape(Q, size(Q)[1:end .!= end]...) |> Flux.gpu

    # get candidate pids
    pids = retrieve(searcher.ivf, searcher.ivf_lengths, searcher.centroids,
        searcher.emb2pid, searcher.config.nprobe, Q)

    # get compressed embeddings for the candidate pids
    codes_packed, residuals_packed = _collect_compressed_embs_for_pids(
        searcher.doclens, searcher.codes, searcher.residuals, pids)

    # decompress these embeddings and move to gpu
    D_packed = decompress(searcher.config.dim, searcher.config.nbits,
        Flux.cpu(searcher.centroids), Flux.cpu(searcher.bucket_weights),
        codes_packed, residuals_packed) |> Flux.gpu
    @assert(size(D_packed, 2)==sum(searcher.doclens[pids]),
        "size(D_packed): $(size(D_packed)), num_embs: $(sum(searcher.doclens[pids]))")
    @assert D_packed isa AbstractMatrix{Float32} "$(typeof(D_packed))"

    # get maxsim scores for the candidate pids
    scores = maxsim(Q, D_packed, pids, searcher.doclens)

    # sort scores and candidate pids, and return the top k
    indices = sortperm(scores, rev = true)
    pids, scores = pids[indices], scores[indices]
    pids[1:k], scores[1:k]
end
