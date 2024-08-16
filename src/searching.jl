struct Searcher
    config::ColBERTConfig
    checkpoint::Checkpoint
    centroids::Matrix{Float32}
    bucket_cutoffs::Vector{Float32}
    bucket_weights::Vector{Float32}
    ivf::Vector{Int}
    ivf_lengths::Vector{Int}
    doclens::Vector{Int}
    codes::Vector{UInt32}
    residuals::Matrix{UInt8}
    emb2pid::Vector{Int}
end

function Searcher(index_path::String)
    if !isdir(index_path)
        error("Index at $(index_path) does not exist! Please build the index first and try again.")
    end

    # loading the config from the path
    config = load_config(index_path)

    # loading the model and saving it to prevent multiple loads
    base_colbert = BaseColBERT(config)
    checkpoint = Checkpoint(base_colbert, config)
    @info "Loaded ColBERT layers from the $(config.checkpoint) HuggingFace checkpoint."

    plan_metadata = JSON.parsefile(joinpath(index_path, "plan.json"))
    codec = load_codec(index_path)
    ivf = JLD2.load_object(joinpath(index_path, "ivf.jld2"))
    ivf_lengths = JLD2.load_object(joinpath(index_path, "ivf_lengths.jld2"))

    # loading all doclens
    doclens = Vector{Int}()
    for chunk_idx in 1:plan_metadata["num_chunks"]
        doclens_file = joinpath(index_path, "doclens.$(chunk_idx).jld2")
        chunk_doclens = JLD2.load_object(doclens_file)
        append!(doclens, chunk_doclens)
    end

    # loading all compressed embeddings
    num_embeddings = plan_metadata["num_embeddings"]
    dim, nbits = config.dim, config.nbits
    @assert (dim * nbits) % 8==0 "(dim, nbits): $((dim, nbits))"
    codes = zeros(UInt32, num_embeddings)
    residuals = zeros(UInt8, Int((dim / 8) * nbits), num_embeddings)
    codes_offset = 1
    for chunk_idx in 1:plan_metadata["num_chunks"]
        chunk_codes = JLD2.load_object(joinpath(index_path, "$(chunk_idx).codes.jld2"))
        chunk_residuals = JLD2.load_object(joinpath(index_path, "$(chunk_idx).residuals.jld2"))

        codes_endpos = codes_offset + length(chunk_codes) - 1
        codes[codes_offset:codes_endpos] = chunk_codes
        residuals[:, codes_offset:codes_endpos] = chunk_residuals

        codes_offset = codes_offset + length(chunk_codes)
    end

    # the emb2pid mapping
    @info "Building the emb2pid mapping."
    @assert isequal(sum(doclens), plan_metadata["num_embeddings"]) "sum(doclens): $(sum(doclens)), num_embeddings: $(plan_metadata["num_embeddings"])"
    emb2pid = zeros(Int, plan_metadata["num_embeddings"])

    offset_doclens = 1
    for (pid, dlength) in enumerate(doclens)
        emb2pid[offset_doclens:(offset_doclens + dlength - 1)] .= pid
        offset_doclens += dlength
    end
    
    Searcher(
        config,
        checkpoint,
        codec["centroids"],
        codec["bucket_cutoffs"],
        codec["bucket_weights"],
        ivf,
        ivf_lengths,
        doclens,
        codes,
        residuals,
        emb2pid
    )
end

"""
    encode_query(searcher::Searcher, query::String)

Encode a search query to a matrix of embeddings using the provided `searcher`. The encoded query can then be used to search the collection.

# Arguments

  - `searcher`: A Searcher object that contains information about the collection and the index.
  - `query`: The search query to encode.

# Returns

An array containing the embeddings for each token in the query. Also see [queryFromText](@ref) to see the size of the array.

# Examples

Here's an example using the `config` and `checkpoint` from the example for [`Checkpoint`](@ref).

```julia-repl
julia> encode_query(config, checkpoint, "what are white spots on raspberries?")
128×32×1 Array{Float32, 3}:
[:, :, 1] =
  0.0158568    0.169676     0.092745      0.0798617    …   0.115938      0.112977      0.107919
  0.220185     0.0304873    0.165348      0.150315         0.0168762     0.0178042     0.0200356
 -0.00790017  -0.0192251   -0.0852365    -0.0799609       -0.0777439    -0.0776733    -0.0830504
 -0.109909    -0.170906    -0.0138701    -0.0409766       -0.126037     -0.126829     -0.13149
 -0.0231787    0.0532214    0.0607473     0.0279048        0.117017      0.114073      0.108536
  0.0620549    0.0465075    0.0821693     0.0606439    …   0.0150612     0.0133353     0.0126583
 -0.0290508    0.143255     0.0306142     0.0426579       -0.164401     -0.161857     -0.160327
  0.0921475    0.058833     0.250449      0.234636         0.0664076     0.0659837     0.0711358
  0.0279402   -0.0278357    0.144855      0.147958         0.154552      0.155525      0.163634
 -0.0768143   -0.00587302   0.00543038    0.00443376      -0.11757      -0.112495     -0.11112
 -0.0184337    0.00668561  -0.191863     -0.161345     …  -0.107664     -0.107267     -0.114564
  0.0112104    0.0214651   -0.0923963    -0.0823052        0.106261      0.105065      0.10409
  0.110971     0.272576     0.148319      0.143233         0.109914      0.112652      0.108365
 -0.131066     0.0376254   -0.0164237    -0.000193318     -0.0969305    -0.0935498    -0.096145
 -0.0402605    0.0350559    0.0162864     0.0269105       -0.070679     -0.0655848    -0.0564059
  0.0799973    0.0482302    0.0712078     0.0792903    …   0.00889943    0.00932721    0.00751066
 -0.137565    -0.0369116   -0.065728     -0.0664102        0.0297059     0.0278639     0.0257616
  0.0479746   -0.102338    -0.0557072    -0.0833976       -0.0566325    -0.0568765    -0.0581378
  0.0656851    0.0195639    0.0288789     0.0559219        0.0596156     0.0541802     0.0525933
  0.0668634   -0.00400549   0.0297102     0.0505045        0.0361149     0.0325914     0.0260693
 -0.0691096    0.0348577   -0.000312685   0.0232462    …  -0.132163     -0.129679     -0.131122
 -0.0273036    0.0653352    0.0332689     0.017918         0.0469949     0.0434268     0.0442646
 -0.0981665   -0.0296463   -0.0114686    -0.0348033       -0.0809244    -0.0823798    -0.081472
 -0.0262739    0.109895     0.0117273     0.0222689        0.0175875     0.013171      0.0195091
  0.0861164    0.0799029    0.00381147    0.0170927        0.0209905     0.0230679     0.0221191
  ⋮                                                    ⋱                 ⋮
 -0.039636    -0.0837763   -0.0837142    -0.0597521        0.0313526     0.0316408     0.0309661
  0.0755214    0.0960326    0.0858578     0.0614626    …   0.109034      0.107593      0.111863
  0.0506199    0.00290888   0.047947      0.063503         0.033966      0.0327732     0.0261081
 -0.0288586   -0.150171    -0.0699125    -0.108002        -0.0697569    -0.0715358    -0.0683193
 -0.0646991    0.0724608   -0.00767811   -0.0184348        0.0649795     0.0697126     0.0808413
  0.0445508    0.0296366    0.0325647     0.0521935        0.12324       0.120497      0.117703
 -0.127301    -0.0224252   -0.00579415   -0.00877803   …  -0.0823464    -0.0803394    -0.0856279
  0.0304881    0.0396951    0.0798097     0.0736797        0.0460205     0.0460111     0.0532082
  0.0488798    0.252244     0.0866849     0.098552        -0.0395483    -0.0463498    -0.0494207
 -0.0296798   -0.0494761    0.00688248    0.0264166       -0.0404835    -0.0410673    -0.0367272
  0.023548    -0.00147361   0.0629259     0.106951        -0.000107777  -0.000898423   0.00296315
 -0.0574151   -0.0875744   -0.103787     -0.114166     …  -0.0687795    -0.070967     -0.0636385
  0.0280373    0.149767    -0.0899733    -0.0732524        0.0201251     0.0197228     0.0219051
 -0.0617143   -0.0573989   -0.0973785    -0.0805046        0.107432      0.108591      0.109502
 -0.0859687    0.0623054    0.0974813     0.126841         0.0182794     0.0230548     0.031103
  0.0392044    0.0162653    0.0926306     0.104054         0.0491496     0.0484319     0.0438133
 -0.0340362   -0.0278067   -0.0181035    -0.0282369    …  -0.0617946    -0.0631367    -0.0675882
  0.0131229    0.0565131   -0.0349061    -0.0464192        0.0724731     0.0780165     0.0746229
 -0.117425     0.162483     0.11039       0.136364        -0.00538224   -0.00685447   -0.00194357
 -0.0401157   -0.00450943   0.0539568     0.0689953       -0.00518066   -0.00600254   -0.0077147
  0.0893984    0.0695061   -0.049941     -0.035411         0.0960931     0.0961892     0.103431
 -0.116265    -0.106331    -0.179832     -0.149728     …  -0.0197172    -0.0220611    -0.018135
 -0.0443452   -0.192203    -0.0187912    -0.0247794       -0.0699094    -0.0684748    -0.0662903
  0.100019    -0.0618588    0.106134      0.0989047       -0.055676     -0.0556784    -0.0595709
```
"""
function encode_query(config::ColBERTConfig, checkpoint::Checkpoint, query::String)
    queries = [query]
    queryFromText(config, checkpoint, queries, config.index_bsize)
end

function search(searcher::Searcher, query::String, k::Int)
    Q = encode_query(searcher.config, searcher.checkpoint, query)

    if size(Q)[3] > 1
        error("Only one query is supported at the moment!")
    end
    @assert size(Q)[3]==1 "size(Q): $(size(Q))"
    @assert isequal(size(Q)[2], searcher.config.query_maxlen) "size(Q): $(size(Q)), query_maxlen: $(searcher.config.query_maxlen)"     # Q: (128, 32, 1)

    Q = reshape(Q, size(Q)[1:end .!= end]...)           # squeeze out the last dimension 
    @assert isequal(length(size(Q)), 2) "size(Q): $(size(Q))"

    pids = retrieve(searcher.ivf, searcher.ivf_lengths, searcher.centroids, searcher.emb2pid, searcher.config.nprobe, Q)
    scores = score_pids(searcher.config, searcher.centroids, searcher.bucket_weights, searcher.doclens, searcher.codes, searcher.residuals, Q, pids)

    indices = sortperm(scores, rev = true)
    pids, scores = pids[indices], scores[indices]
    pids[1:k], scores[1:k]
end
