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
            4
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

    # check if all relevant files are saved
    _check_all_files_are_saved(indexer.config.index_path)

    # collect embedding offsets and more metadata for chunks
    chunk_emb_counts = load_chunk_metadata_property(
        indexer.config.index_path, "num_embeddings")
    num_embeddings, embeddings_offsets = _collect_embedding_id_offset(chunk_emb_counts)
    @info "Updating chunk metadata and indexing plan"
    plan_dict["num_embeddings"] = num_embeddings
    plan_dict["embeddings_offsets"] = embeddings_offsets
    open(joinpath(indexer.config.index_path, "plan.json"), "w") do io
        JSON.print(io,
            plan_dict,
            4
        )
    end
    save_chunk_metadata_property(
        indexer.config.index_path, "embedding_offset", embeddings_offsets)

    # build and save the ivf
    @info "Building the centroid to embedding IVF."
    codes = load_codes(indexer.config.index_path)
    ivf, ivf_lengths = _build_ivf(codes, plan_dict["num_partitions"])

    @info "Saving the IVF."
    ivf_path = joinpath(indexer.config.index_path, "ivf.jld2")
    ivf_lengths_path = joinpath(indexer.config.index_path, "ivf_lengths.jld2")
    JLD2.save_object(ivf_path, ivf)
    JLD2.save_object(ivf_lengths_path, ivf_lengths)
end
