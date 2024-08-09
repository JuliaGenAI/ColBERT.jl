struct Searcher
    config::ColBERTConfig
    checkpoint::Checkpoint
    ranker::IndexScorer
end

function Searcher(index_path::String)
    if !isdir(index_path)
        error("Index at $(index_path) does not exist! Please build the index first and try again.")
    end

    # loading the config from the path
    config = JLD2.load(joinpath(index_path, "config.jld2"))["config"]

    # loading the model and saving it to prevent multiple loads
    @info "Loading ColBERT layers from HuggingFace."
    base_colbert = BaseColBERT(config.resource_settings.checkpoint, config)
    checkPoint = Checkpoint(base_colbert, DocTokenizer(base_colbert.tokenizer, config),
        QueryTokenizer(base_colbert.tokenizer, config), config)

    Searcher(config, checkPoint, IndexScorer(index_path))
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

Here's an example using the config given in docs for [`ColBERTConfig`](@ref).

```julia-repl
julia> searcher = Searcher(config);

julia> encode_query(searcher, "what are white spots on raspberries?")
128×32×1 Array{Float32, 3}:
[:, :, 1] =
  0.0158567    0.169676     0.092745     0.0798617   …   0.115938     0.112977     0.107919
  0.220185     0.0304873    0.165348     0.150315        0.0168762    0.0178042    0.0200357
 -0.00790007  -0.0192251   -0.0852364   -0.0799609      -0.0777439   -0.0776733   -0.0830504
 -0.109909    -0.170906    -0.0138702   -0.0409767      -0.126037    -0.126829    -0.13149
 -0.0231786    0.0532214    0.0607473    0.0279048       0.117017     0.114073     0.108536
  0.0620549    0.0465075    0.0821693    0.0606439   …   0.0150612    0.0133353    0.0126583
 -0.0290509    0.143255     0.0306142    0.042658       -0.164401    -0.161857    -0.160327
  0.0921477    0.0588331    0.250449     0.234636        0.0664076    0.0659837    0.0711357
  0.0279402   -0.0278357    0.144855     0.147958        0.154552     0.155525     0.163634
 -0.0768143   -0.00587305   0.00543038   0.00443374     -0.11757     -0.112495    -0.11112
 -0.0184338    0.00668557  -0.191863    -0.161345    …  -0.107664    -0.107267    -0.114564
  0.0112104    0.0214651   -0.0923963   -0.0823051       0.106261     0.105065     0.10409
  ⋮                                                  ⋱                ⋮
 -0.0617142   -0.0573989   -0.0973785   -0.0805046       0.107432     0.108591     0.109501
 -0.0859686    0.0623054    0.0974813    0.126841        0.0182795    0.0230549    0.031103
  0.0392043    0.0162653    0.0926306    0.104053        0.0491495    0.0484318    0.0438132
 -0.0340363   -0.0278066   -0.0181035   -0.0282369   …  -0.0617945   -0.0631367   -0.0675882
  0.013123     0.0565132   -0.0349061   -0.0464192       0.0724731    0.0780166    0.074623
 -0.117425     0.162483     0.11039      0.136364       -0.00538225  -0.00685449  -0.0019436
 -0.0401158   -0.0045094    0.0539569    0.0689953      -0.00518063  -0.00600252  -0.00771469
  0.0893983    0.0695061   -0.0499409   -0.035411        0.0960932    0.0961893    0.103431
 -0.116265    -0.106331    -0.179832    -0.149728    …  -0.0197172   -0.022061    -0.018135
 -0.0443452   -0.192203    -0.0187912   -0.0247794      -0.0699095   -0.0684749   -0.0662904
  0.100019    -0.0618588    0.106134     0.0989047      -0.0556761   -0.0556784   -0.059571
```
"""
function encode_query(searcher::Searcher, query::String)
    queries = [query]
    bsize = 128
    Q = queryFromText(searcher.checkpoint, queries, bsize)
    Q
end

function search(searcher::Searcher, query::String, k::Int)
    dense_search(searcher, encode_query(searcher, query), k)
end

function dense_search(searcher::Searcher, Q::AbstractArray{Float32}, k::Int)
    pids, scores = rank(searcher.ranker, searcher.config, Q)

    pids[1:k], scores[1:k]
end
