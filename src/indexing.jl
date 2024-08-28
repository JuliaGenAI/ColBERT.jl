struct Indexer
    config::ColBERTConfig
    bert::HF.HGFBertModel
    linear::Layers.Dense
    tokenizer::TextEncoders.AbstractTransformerTextEncoder
    collection::Vector{String}
    skiplist::Vector{Int}
end

"""
    Indexer(config::ColBERTConfig)

Type representing an ColBERT indexer.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) used to build the index.

# Returns

An [`Indexer`] wrapping a [`ColBERTConfig`](@ref), a [`Checkpoint`](@ref) and
a collection of documents to index.
"""
function Indexer(config::ColBERTConfig)
    tokenizer, bert, linear = load_hgf_pretrained_local(config.checkpoint)
    bert = bert |> Flux.gpu
    linear = linear |> Flux.gpu
    collection = readlines(config.collection)
    punctuations_and_padsym = [string.(collect("!\"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"));
                               tokenizer.padsym]
    skiplist = config.mask_punctuation ?
               [lookup(tokenizer.vocab, sym) for sym in punctuations_and_padsym] :
               [lookup(tokenizer.vocab, tokenizer.padsym)]

    # configuring the tokenizer; using doc_maxlen
    process = tokenizer.process
    truncpad_pipe = Pipeline{:token}(
        TextEncodeBase.trunc_or_pad(
            config.doc_maxlen - 1, "[PAD]", :tail, :tail),
        :token)
    process = process[1:4] |> truncpad_pipe |> process[6:end]
    tokenizer = TextEncoders.BertTextEncoder(
        tokenizer.tokenizer, tokenizer.vocab, process;
        startsym = tokenizer.startsym, endsym = tokenizer.endsym,
        padsym = tokenizer.padsym, trunc = tokenizer.trunc)

    @info "Loaded ColBERT layers from the $(config.checkpoint) HuggingFace checkpoint."
    @info "Loaded $(length(collection)) documents from $(config.collection)."

    Indexer(config, bert, linear, tokenizer, collection, skiplist)
end

"""
    index(indexer::Indexer)

Build an index given the configuration stored in `indexer`.

# Arguments

  - `indexer`: An `Indexer` which is used to build the index on disk.
"""
function index(indexer::Indexer)
    if isdir(indexer.config.index_path)
        @info "Index at $(indexer.config.index_path) already exists! Skipping indexing."
        return
    end
    isdir(indexer.config.index_path) || mkdir(indexer.config.index_path)

    # sampling passages and getting their embedings
    @info "Sampling PIDs for clustering and generating their embeddings."
    @time avg_doclen_est, sample = _sample_embeddings(
        indexer.bert, indexer.linear, indexer.tokenizer,
        indexer.config.dim, indexer.config.index_bsize,
        indexer.config.doc_token_id, indexer.skiplist, indexer.collection)

    # splitting the sample to a clustering set and a heldout set
    @info "Splitting the sampled embeddings to a heldout set."
    @time sample, sample_heldout = _heldout_split(sample)
    @assert sample isa AbstractMatrix{Float32} "$(typeof(sample))"
    @assert sample_heldout isa AbstractMatrix{Float32} "$(typeof(sample_heldout))"

    # generating the indexing setup
    plan_dict = setup(indexer.collection, avg_doclen_est, size(sample, 2),
        indexer.config.chunksize, indexer.config.nranks)
    @info "Saving the index plan to $(joinpath(indexer.config.index_path, "plan.json"))."
    open(joinpath(indexer.config.index_path, "plan.json"), "w") do io
        JSON.print(io,
            plan_dict,
            4                                                               # indent
        )
    end
    @info "Saving the config to the indexing path."
    ColBERT.save(indexer.config)

    # training/clustering
    @info "Training the clusters."
    @time centroids, bucket_cutoffs, bucket_weights, avg_residual = train(
        sample, sample_heldout, plan_dict["num_partitions"],
        indexer.config.nbits, indexer.config.kmeans_niters)
    save_codec(
        indexer.config.index_path, centroids, bucket_cutoffs,
        bucket_weights, avg_residual)
    sample, sample_heldout = nothing, nothing           # these are big arrays

    # indexing
    @info "Building the index."
    @time index(indexer.config.index_path, indexer.bert, indexer.linear,
        indexer.tokenizer, indexer.collection, indexer.config.dim,
        indexer.config.index_bsize, indexer.config.doc_token_id,
        indexer.skiplist, plan_dict["num_chunks"], plan_dict["chunksize"],
        centroids, bucket_cutoffs, indexer.config.nbits)

    # finalizing
    @info "Running some final checks."
    _check_all_files_are_saved(indexer.config.index_path)
    _collect_embedding_id_offset(indexer.config.index_path)
    _build_ivf(indexer.config.index_path)
end
