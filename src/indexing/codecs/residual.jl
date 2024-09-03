"""
    compress_into_codes(
        centroids::AbstractMatrix{Float32}, embs::AbstractMatrix{Float32})

Compresses a matrix of embeddings into a vector of codes using the given `centroids`,
where the code for each embedding is its nearest centroid ID.

# Arguments

  - `centroids`: The matrix of centroids.
  - `embs`: The matrix of embeddings to be compressed.

# Returns

A `Vector{UInt32}` of codes, where each code corresponds to the nearest centroid ID for the embedding.

# Examples

```julia-repl
julia> using ColBERT: compress_into_codes;

julia> using Flux, CUDA, Random;

julia> Random.seed!(0);

julia> centroids = rand(Float32, 128, 500) |> Flux.gpu;

julia> embs = rand(Float32, 128, 10000) |> Flux.gpu;

julia> codes = zeros(UInt32, size(embs, 2)) |> Flux.gpu;

julia> @time compress_into_codes!(codes, centroids, embs);
  0.003489 seconds (4.51 k allocations: 117.117 KiB)

julia> codes
10000-element CuArray{UInt32, 1, CUDA.DeviceMemory}:
 0x00000194
 0x00000194
 0x0000000b
 0x000001d9
 0x0000011f
 0x00000098
 0x0000014e
 0x00000012
 0x000000a0
 0x00000098
 0x000001a7
 0x00000098
 0x000001a7
 0x00000194
          ⋮
 0x00000199
 0x000001a7
 0x0000014e
 0x000001a7
 0x000001a7
 0x000001a7
 0x000000ec
 0x00000098
 0x000001d9
 0x00000098
 0x000001d9
 0x000001d9
 0x00000012
```
"""
function compress_into_codes!(
        codes::AbstractVector{UInt32}, centroids::AbstractMatrix{Float32},
        embs::AbstractMatrix{Float32};
        bsize::Int = 1000)
    _, n = size(embs)
    length(codes) == n ||
        throw(DimensionMismatch("length(codes) must be equal" *
                                "to the number of embeddings!"))
    for offset in 1:bsize:size(embs, 2)
        offset_end = min(n, offset + bsize - 1)
        dot_products = (embs[:, offset:offset_end])' * centroids    # (num_embs, num_centroids)
        indices = getindex.(argmax(dot_products, dims = 2), 2)
        codes[offset:offset_end] .= indices
    end
end

"""

# Examples

```julia-repl
julia> using ColBERT: _binarize;

julia> using Flux, CUDA, Random;

julia> Random.seed!(0);

julia> nbits = 5;

julia> data = rand(0:2^nbits - 1, 100, 200000) |> Flux.gpu
100×200000 CuArray{Int64, 2, CUDA.DeviceMemory}:
 12  23  11   6   5   2  27   1   0   4  15   8  24  …   4  25  22  18   4   0  15  16   3  25   4  13
  2  11  29   8  31   3  15   1   8   1  22  22  10     25  25   1  12  21  13  27  20  23  24   9  14
 27   4   4  15   4   9  19   4   3  10  27  14   3     10   8  18  19  12   9  29  23   8  15  30  21
  2   7   4   5  25  16  27  23   5  24  26  19   9     22   1  21  12  31  20   4  31  26  21  25   6
 21  18  25   9   9  17   6  20  16  13  14   2   2     28  13  11   9  22   4   2  22  27  24   9  31
  3  26  22   8  24  23  29  19  13   3   2  20  14  …  22  18  18   5  16   5   9   3  21  19  17  23
  3  13   5   9   8  12  24  26   8  10  14   1  21     14  25  18   5   1   4  13   0  14  11  16   8
 22  20  22   6  25   1  29  23   9  21  13  27   6     11  21   4  31  14  14   5  27  17   6  27  19
  9   2   7   2  16   1  23  15   2  17  30  18   4     26   5  20  31  18   8  20  13  23  26  29  25
  0   6  20   8   0  18   9  28   8  30   6   2  21      0   7  25  23  19   2   6  27  13   3   6  22
 17   2   0  13  26   6   7   8  14  20  11   9  17  …  29   4  28  22   1  10  29  20  11  20  30   8
 28   5   0  30   1  26  23   9  29   9  29   2  15     27   8  13  11  27   6  11   7  19   4   7  28
  8   9  16  29  22   8   9  19  30  20   4   0   1      1  25  14  16  17  26  28  31  25   4  22  23
 10   9  31  22  20  15   1   9  26   2   0   1  27     23  21  15  22  29  29   1  24  30  22  17  22
 13   8  23   9   1   6   2  28  18   1  15   5  12     28  27   3   6  22   3  20  24   3   2   2  29
 28  22  19   7  20  28  25  13   3  13  17  31  28  …  18  17  19   6  20  11  31   9  28   9  19   1
 23   1   7  14   6  14   0   9   1   9  12  30  24     23   2  13   9   0  20  17   4  16  22  27  11
  4  19   8  31  14  30   2  13  27  16  29  10  30     29  25  28  31  13  11   8  12  30  13  10   7
 18  26  30   6  31   6  15  11  10  31  21  24  11     19  19  29  17  13   5   3  28  29  31  22  13
 14  29  18  14  25  10  28  28  15   8   5  14   5     10  17  13  23   0  26  25  13  15  26   3   5
  0   4  24  23  20  16  25   9  17  27  15   0  10  …   5  18   2   2  30  17   8  11  27  11  15  27
 15   2  22   8   6   8  16   2   8  24  26  15  30     27  12  28  31  26  18   4  10   5  16  23  16
 20  20  29  24   1   9  18  31  16   3   9  17  31      8   4   4  15  13  16   0  10  31  28   8  29
  2   3   2  23  15  21   6   8  21   7  17  15  17      7  15  19  25   3   2  11  26  16  12  11  27
 13  21  22  20  15   0  22   2  30  14  14  20  26     13  23  14  18   0  24  21  17   8  11  26  22
  ⋮                   ⋮                   ⋮          ⋱           ⋮                   ⋮
  9   7   1   1  28  28  10  16  23  18  26   9   7  …  14   5  12   3   6  25  20   5  13   3  20  10
 28  25  21   8  31   4  25   7  27  26  19   4   9     15  26   2  23  14  16  29  17  11  29  12  18
  4  15  20   2   3  10   6   9  13  22   5  28  21     12  11  12  14  14   9  13  31  12   6   9  21
  9  24   2   4  27  14   4  15  19   2  14  30   3     17   5   6   2  23  15  11   1   0  10   0  28
 20   0  26   8  21   7   1   7  22  10  10   5  31     23   5  20  11  29  12  25  14  13   5  25  15
  2   9  27  28  25   7  27  30  20   5  10   2  28  …  21  19  22  30  24   0  10  19  10  30  22   9
 10   2  31  10  12  13  16  10   5  28  16   4  16      3   1  31  20  19  16  19  30  31  14   5  20
 14   2  20  19  16  25   4   1  15  31  22  17   8     12  19   9  29  30  20  13  19  14  18   7  22
 20   3  27  23   9  21  20  10  14   3   5  26  22     19  19  11   3  22  19  24  12  27  12  28  17
  1  27  27  10   8  29  17  14  19   6   6  12   6     10   6  24  29  26  11   2  25   7   6   1  28
 11  19   5   1   7  19   8  17  27   4   4   7   0  …  13  29   0  15  15   2   2   6  24   0   5  18
 17  31  31  23  24  18   0  31   6  22  20  31  23     16   5   8  17   6  20  23  21  26  15  27  30
  1   6  30  31   8   3  28  31  10  23  23  24  26     12  30  10   3  25  24  12  20   8   7  14  11
 26  22  23  21  24   7   2  19  10  27  21  14   7      7  27   1  29   7  23  30  24  12   9  12  14
 28  26   8  28  10  18  23  28  10  19  31  26  17     18  20  23   8  31  15  18  10  24  28   7  23
  1   7  15  22  23   0  21  19  28  10  15  13   7  …  21  15  16   1  16   9  25  23   1  24  20   5
 21   7  30  30   5   0  27  26   6   7  30   2  16      2  16   6   9   6   4  12   4  12  18  28  17
 11  16   0  20  20  13  18  19  21   7  24   4  26      1  26   7  16   0   2   3   2  22  27  25  15
  9  20  31  24  14  29  28  26  29  31   7  28  12     28   0  12   3  17   7   0  30  25  22  23  20
 19  21  30  16  15  20  31   2   2   8  27  20  29     27  13   2  27   8  17  19  15   9  22   3  27
 13  17   6   4   9   1  18   2  21  27  13  14  12  …  28  21   4   2  11  13  31  13  25  25  29  21
  2  17  19  15  17   1  12   0  11   9  16   1  13     25  21  28  22   7  13   3  20   7   6  26  21
 13   6   5  11  12   2   2   3   4  16  30  14  19     16   5   5  19  17   3  31  26  19   2  11  15
 20  30  21  30  13  26   7   9  11  18   3   0  15      3  14  15   1   9  16   1  16   0   2   2  11
  3  24   6  16  10   3   7  17   0  30   9  14   1     29   4   8   4  17  14  27   0  17  12   4  19

julia> _binarize(data, nbits)
5×100×200000 CuArray{Bool, 3, CUDA.DeviceMemory}:
[:, :, 1] =
 0  0  1  0  1  1  1  0  1  0  1  0  0  0  1  0  1  0  0  …  0  0  0  1  1  1  1  0  0  1  1  1  1  1  1  0  1  0  1
 0  1  1  1  0  1  1  1  0  0  0  0  0  1  0  0  1  0  1     1  1  0  0  1  0  0  1  0  0  0  1  0  1  0  1  0  0  1
 1  0  0  0  1  0  0  1  0  0  0  1  0  0  1  1  1  1  0     0  1  1  0  0  0  0  0  1  0  1  0  0  0  1  0  1  1  0
 1  0  1  0  0  0  0  0  1  0  0  1  1  1  1  1  0  0  0     1  1  0  0  1  0  0  1  1  0  0  1  1  0  1  0  1  0  0
 0  0  1  0  1  0  0  1  0  0  1  1  0  0  0  1  1  0  1     0  0  1  0  0  1  0  1  1  0  1  0  0  1  0  0  0  1  0

[:, :, 2] =
 1  1  0  1  0  0  1  0  0  0  0  1  1  1  0  0  1  1  0  …  0  0  1  1  1  1  0  0  0  1  1  0  0  1  1  1  0  0  0
 1  1  0  1  1  1  0  0  1  1  1  0  0  0  0  1  0  1  1     1  1  1  1  1  1  1  1  1  1  1  0  0  0  0  0  1  1  0
 1  0  1  1  0  0  1  1  0  1  0  1  0  0  0  1  0  0  0     0  0  0  0  0  1  1  1  0  1  1  0  1  1  0  0  1  1  0
 0  1  0  0  0  1  1  0  0  0  0  0  1  1  1  0  0  0  1     0  0  0  1  0  1  0  0  1  0  0  0  0  0  0  0  0  1  1
 1  0  0  0  1  1  0  1  0  0  0  0  0  0  0  1  0  1  1     0  0  0  1  1  1  0  1  1  0  0  1  1  1  1  1  0  1  1

[:, :, 3] =
 1  1  0  0  1  0  1  0  1  0  0  0  0  1  1  1  1  0  0  …  1  0  1  1  1  1  0  1  0  1  0  0  1  0  0  1  1  1  0
 1  0  0  0  0  1  0  1  1  0  0  0  0  1  1  1  1  0  1     1  0  1  1  0  1  1  1  0  1  1  0  1  1  1  1  0  0  1
 0  1  1  1  0  1  1  1  1  1  0  0  0  1  1  0  1  0  1     1  1  0  0  1  1  1  1  0  1  1  0  1  1  1  0  1  1  1
 1  1  0  0  1  0  0  0  0  0  0  0  0  1  0  0  0  1  1     1  0  1  1  0  1  1  0  1  1  1  0  1  1  0  0  0  0  0
 0  1  0  0  1  1  0  1  0  1  0  0  1  1  1  1  0  0  1     1  1  1  1  0  1  1  1  0  0  1  0  1  1  0  1  0  1  0

;;; …

[:, :, 199998] =
 1  0  1  1  0  1  1  0  0  1  0  0  0  0  0  1  0  1  1  …  0  0  0  0  0  1  1  1  0  0  0  1  0  0  1  0  0  0  0
 0  0  1  0  0  1  1  1  1  1  0  0  0  1  1  0  1  0  1     1  1  0  1  0  1  1  0  0  0  1  1  1  1  0  1  1  1  0
 0  0  1  1  0  0  0  1  0  0  1  1  1  1  0  0  1  1  1     1  0  1  1  0  1  1  0  1  0  0  0  1  1  0  1  0  0  1
 1  1  1  0  1  0  1  0  1  0  0  0  0  0  0  1  0  1  1     1  0  1  0  0  1  0  1  1  1  0  1  0  0  1  0  0  0  1
 1  1  0  1  1  1  0  0  1  0  1  0  0  1  0  0  1  0  1     0  1  0  0  0  0  0  0  1  1  1  1  1  1  1  0  0  0  0

[:, :, 199999] =
 0  1  0  1  1  1  0  1  1  0  0  1  0  1  0  1  1  0  0  …  1  1  0  1  1  1  0  0  1  0  0  1  1  1  1  0  1  0  0
 0  0  1  0  0  0  0  1  0  1  1  1  1  0  1  1  1  1  1     0  1  0  0  0  1  1  0  1  0  0  0  1  1  0  1  1  1  0
 1  0  1  0  0  0  0  0  1  1  1  1  1  0  0  0  0  0  1     1  1  1  0  1  0  1  1  1  1  1  0  1  0  1  0  0  0  1
 0  1  1  1  1  0  0  1  1  0  1  0  0  0  0  0  1  1  0     0  0  1  0  0  1  1  1  0  0  1  1  0  0  1  1  1  0  0
 0  0  1  1  0  1  1  1  1  0  1  0  1  1  0  1  1  0  1     0  0  1  0  0  1  0  0  0  1  1  1  1  0  1  1  0  0  0

[:, :, 200000] =
 1  0  1  0  1  1  0  1  1  0  0  0  1  0  1  1  1  1  1  …  0  0  1  0  0  0  1  0  1  1  1  1  0  1  1  1  1  1  1
 0  1  0  1  1  1  0  1  0  1  0  0  1  1  0  0  1  1  0     0  1  0  0  1  1  1  1  1  0  0  1  0  1  0  0  1  1  1
 1  1  1  1  1  1  0  0  0  1  0  1  1  1  1  0  0  1  1     1  1  0  1  0  1  0  1  1  1  0  1  1  0  1  1  1  0  0
 1  1  0  0  1  0  1  0  1  0  1  1  0  0  1  0  1  0  1     0  0  0  1  0  1  1  1  0  0  0  1  0  1  0  0  1  1  0
 0  0  1  0  1  1  0  1  1  1  0  1  1  1  1  0  0  0  0     1  1  1  1  1  1  0  0  1  0  1  0  1  1  1  1  0  0  1
```
"""
function _binarize(data::AbstractMatrix{T}, nbits::Int) where {T <: Integer}
    all(in(0:(1 << nbits - 1)), data) ||
        throw(DomainError("All values in the matrix should be in " *
                          "range [0, 2^nbits - 1]!"))
    data = stack(fill(data, nbits), dims = 1)                       # (nbits, dim, batch_size)
    positionbits = similar(data, nbits)                             # respects device
    copyto!(positionbits, map(Base.Fix1(<<, 1), 0:(nbits - 1)))     # (nbits, 1)
    positionbits = reshape(positionbits, nbits, 1, 1)               # (nbits, 1, 1)
    data .= fld.(data, positionbits)                                # divide by 2^bit for each bit position
    data .= data .& 1                                               # apply mod 1 to binarize
    map(Bool, data)
end

"""

# Examples

```julia-repl
julia> using ColBERT: _binarize, _unbinarize;

julia> using Flux, CUDA, Random;

julia> Random.seed!(0);

julia> nbits = 5;

julia> data = rand(0:2^nbits - 1, 100, 200000) |> Flux.gpu

julia> binarized_data = _binarize(data, nbits);

julia> unbinarized_data = _unbinarize(binarized_data);

julia> isequal(unbinarized_data, data)
true
```
"""
function _unbinarize(data::AbstractArray{Bool, 3})
    nbits = size(data, 1)
    positionbits = similar(data, Int, nbits)                        # respects device
    copyto!(positionbits, map(Base.Fix1(<<, 1), 0:(nbits - 1)))     # (nbits, 1)
    positionbits = reshape(positionbits, nbits, 1, 1)               # (nbits, 1, 1)
    integer_data = sum(data .* positionbits, dims = 1)
    reshape(integer_data, size(integer_data)[2:end])
end

"""

# Examples

```julia-repl
julia> using ColBERT: _bucket_indices;

julia> using Random; Random.seed!(0);

julia> data = rand(50, 50) |> Flux.gpu;
50×50 CuArray{Float32, 2, CUDA.DeviceMemory}:
 0.455238   0.828104   0.735106   0.042069   …  0.916387    0.10078      0.00907127
 0.547642   0.100748   0.993553   0.0275458     0.0954245   0.351846     0.548682
 0.773354   0.908416   0.703694   0.839846      0.613082    0.605597     0.660227
 0.940585   0.932748   0.150822   0.920883      0.754362    0.843869     0.0453409
 0.0296477  0.123079   0.409406   0.672372      0.19912     0.106127     0.945276
 0.746943   0.149248   0.864755   0.116243   …  0.541295    0.224275     0.660706
 0.746801   0.743713   0.64608    0.446445      0.951642    0.583662     0.338174
 0.97667    0.722362   0.692789   0.646206      0.089323    0.305554     0.454803
 0.329335   0.785124   0.254097   0.271299      0.320879    0.000438984  0.161356
 0.672001   0.532197   0.869579   0.182068      0.289906    0.068645     0.142121
 0.0997382  0.523732   0.315933   0.935547   …  0.819027    0.770597     0.654065
 0.230139   0.997278   0.455917   0.566976      0.0180972   0.275211     0.0619634
 0.631256   0.709048   0.810256   0.754144      0.452911    0.358555     0.116042
 0.096652   0.454081   0.715283   0.923417      0.498907    0.781054     0.841858
 0.69801    0.0439444  0.27613    0.617714      0.589872    0.708365     0.0266968
 0.470257   0.654557   0.351769   0.812597   …  0.323819    0.621386     0.63478
 0.114864   0.897316   0.0243141  0.910847      0.232374    0.861399     0.844008
 0.984812   0.491806   0.356395   0.501248      0.651833    0.173494     0.38356
 0.730758   0.970359   0.456407   0.8044        0.0385577   0.306404     0.705577
 0.117333   0.233628   0.332989   0.0857914     0.224095    0.747571     0.387572
 ⋮                                           ⋱
 0.908402   0.609104   0.108874   0.430905   …  0.00564743  0.964602     0.541285
 0.570179   0.10114    0.210174   0.945569      0.149051    0.785343     0.241959
 0.408136   0.221389   0.425872   0.204654      0.238413    0.583185     0.271998
 0.526989   0.0401535  0.686314   0.534208      0.29416     0.488244     0.747676
 0.129952   0.716592   0.352166   0.584363      0.0850619   0.161153     0.243575
 0.0256413  0.0831649  0.179467   0.799997   …  0.229072    0.711857     0.326977
 0.939913   0.21433    0.223666   0.914527      0.425202    0.129862     0.766065
 0.600877   0.516631   0.753827   0.674017      0.665329    0.622929     0.645962
 0.223773   0.257933   0.854171   0.259882      0.298119    0.231662     0.824881
 0.268817   0.468576   0.218589   0.835418      0.802857    0.0159643    0.0330232
 0.408092   0.361884   0.849442   0.527004   …  0.0500168   0.427498     0.70482
 0.740789   0.952265   0.722908   0.0856596     0.507305    0.32629      0.117663
 0.873501   0.587707   0.894573   0.355338      0.345011    0.0693833    0.457268
 0.758824   0.162728   0.608327   0.902837      0.492069    0.716635     0.459272
 0.922832   0.950539   0.51935    0.52672       0.725665    0.36443      0.936056
 0.239929   0.3754     0.247219   0.92438    …  0.0763809   0.737196     0.712317
 0.76676    0.182714   0.866055   0.749239      0.132254    0.755823     0.0869469
 0.378313   0.0392607  0.93354    0.908511      0.733769    0.552135     0.351491
 0.811121   0.891591   0.610976   0.0427439     0.0258436   0.482621     0.193291
 0.109315   0.474986   0.140528   0.776382      0.609791    0.49946      0.116989

julia> bucket_cutoffs = sort(rand(5)) |> Flux.gpu;
5-element CuArray{Float32, 1, CUDA.DeviceMemory}:
 0.42291805
 0.7075339
 0.8812783
 0.89976573
 0.9318977

julia> _bucket_indices(data, bucket_cutoffs)
50×50 CuArray{Int64, 2, CUDA.DeviceMemory}:
 1  2  2  0  1  0  2  0  0  2  0  1  1  0  …  0  0  0  1  1  0  2  2  4  0  4  0  0
 1  0  5  0  1  4  1  2  0  0  5  1  0  0     0  0  1  2  4  2  0  0  0  2  0  0  1
 2  4  1  2  1  0  5  0  1  1  0  0  0  1     2  5  1  1  1  1  1  1  0  5  1  1  1
 5  5  0  4  0  0  1  2  4  0  4  1  0  0     5  5  4  2  1  0  2  0  1  0  2  2  0
 0  0  0  1  0  0  1  1  0  2  0  1  2  0     1  0  2  0  2  0  2  1  1  5  0  0  5
 2  0  2  0  1  0  1  0  2  4  2  2  0  2  …  0  1  0  4  0  5  0  0  0  2  1  0  1
 2  2  1  1  1  0  3  0  2  0  1  1  5  0     2  0  0  0  0  1  0  5  5  1  5  1  0
 5  2  1  1  2  5  0  0  1  3  0  1  0  1     0  0  0  0  0  1  4  0  1  0  0  0  1
 0  2  0  0  1  1  0  5  2  0  2  2  2  2     0  0  5  5  0  0  2  2  0  2  0  0  0
 1  1  2  0  2  4  5  5  1  0  2  2  2  0     0  0  1  1  1  0  0  1  1  2  0  0  0
 0  1  0  5  0  0  2  0  2  0  0  3  0  0  …  1  2  0  5  0  1  2  0  0  0  2  2  1
 0  5  1  1  2  1  0  1  1  0  0  1  1  0     5  0  0  2  2  0  3  1  1  4  0  0  0
 1  2  2  2  2  1  1  5  0  0  0  1  0  5     0  1  1  0  0  0  2  0  2  0  1  0  0
 0  1  2  4  1  2  1  2  0  2  2  0  0  0     0  1  0  1  0  1  3  1  1  1  1  2  2
 1  0  0  1  4  0  2  2  5  4  0  3  0  1     3  0  0  0  0  5  0  1  2  0  1  2  0
 1  1  0  2  0  1  5  3  1  2  5  2  1  2  …  1  1  2  0  0  0  2  1  2  3  0  1  1
 0  3  0  4  0  0  0  0  0  0  0  0  0  1     1  1  1  2  0  1  0  2  3  0  0  2  2
 5  1  0  1  2  0  2  0  0  2  0  0  1  0     1  4  0  2  0  0  0  0  1  0  1  0  0
 2  5  1  2  0  1  0  2  5  1  1  1  5  0     1  1  0  0  2  0  1  0  4  0  0  0  1
 0  0  0  0  0  2  3  1  0  1  1  0  1  2     0  1  1  1  1  0  0  0  5  1  0  2  0
 ⋮              ⋮              ⋮           ⋱           ⋮              ⋮
 4  1  0  1  4  1  2  0  1  0  0  1  0  2  …  0  0  0  0  0  2  0  2  0  1  0  5  1
 1  0  0  5  2  2  5  0  0  3  5  0  1  5     1  2  0  1  2  0  0  0  1  0  0  2  0
 0  0  1  0  0  1  4  0  0  1  0  5  1  5     1  1  2  0  2  0  1  1  2  4  0  1  0
 1  0  1  1  0  0  0  0  1  0  0  0  0  4     0  0  1  0  3  5  0  1  1  1  0  1  2
 0  2  0  1  0  0  2  0  2  1  1  2  1  1     0  0  0  1  1  1  0  0  1  2  0  0  0
 0  0  0  2  5  2  2  0  0  5  5  4  1  0  …  0  0  2  1  5  0  1  0  1  0  0  2  0
 5  0  0  4  0  1  0  0  0  1  2  2  0  0     1  0  0  0  1  1  4  0  5  1  1  0  2
 1  1  2  1  1  1  0  0  0  0  0  2  1  0     0  5  0  1  0  0  1  2  0  0  1  1  1
 0  0  2  0  0  1  1  4  0  2  2  0  5  1     1  1  1  1  5  0  3  2  2  1  0  0  2
 0  1  0  2  2  1  1  0  1  0  1  0  0  2     5  0  1  0  5  0  0  2  2  0  2  0  0
 0  0  2  1  0  1  1  1  1  2  4  0  1  2  …  1  1  1  1  0  0  5  1  0  0  0  1  1
 2  5  2  0  0  0  2  0  2  0  0  0  0  0     4  0  5  5  0  2  0  0  0  0  1  0  0
 2  1  3  0  1  1  0  0  4  0  0  1  1  0     1  1  0  4  1  1  0  2  0  3  0  0  1
 2  0  1  4  1  0  0  1  0  2  1  0  0  0     5  1  0  0  1  1  0  0  2  0  1  2  1
 4  5  1  1  1  1  0  0  0  1  1  0  5  2     5  0  2  2  1  1  1  5  2  1  2  0  5
 0  0  0  4  2  1  0  3  0  3  2  0  1  2  …  0  1  0  2  0  0  2  5  2  0  0  2  2
 2  0  2  2  1  0  0  3  1  1  0  5  2  0     2  0  2  0  5  1  0  0  1  0  0  2  0
 0  0  5  4  1  0  2  2  2  0  1  1  2  5     0  0  0  0  1  0  0  1  0  1  2  1  0
 2  3  1  0  0  2  0  0  5  0  5  0  1  1     0  0  5  2  0  1  0  5  2  1  0  1  0
 0  1  0  2  1  0  2  2  1  0  1  4  1  1     5  1  0  1  4  1  1  1  1  1  1  1  0
```
"""
function _bucket_indices(data::AbstractMatrix{T},
        bucket_cutoffs::AbstractVector{S}) where {T <: Number, S <: Number}
    map(Base.Fix1(searchsortedfirst, bucket_cutoffs), data) .- 1
end

"""

# Examples

```julia-repl
julia> using ColBERT: _packbits;

julia> using Random; Random.seed!(0);

julia> bitsarray = rand(Bool, 2, 128, 200000);

julia> _packbits(bitsarray)
32×200000 Matrix{UInt8}:
 0x2e  0x93  0x5a  0xbd  0xd1  0x89  0x2c  0x39  0x6a  …  0xed  0xdb  0x45  0x95  0xf8  0x64  0x57  0x5b  0x06
 0x3f  0x45  0x0c  0x2a  0x14  0xdb  0x16  0x2b  0x00     0x70  0xba  0x3c  0x40  0x56  0xa6  0xbe  0x33  0x3d
 0xbd  0x61  0xa3  0xa7  0xb4  0xe7  0x1e  0xf8  0xa7     0xf0  0x70  0xaf  0xc0  0xeb  0xa3  0x34  0x6d  0x81
 0x15  0x9d  0x02  0xa5  0x7b  0x84  0xde  0x2f  0x28     0xa7  0xf2  0x51  0xb3  0xe7  0x01  0xbf  0x6f  0x5a
 0xaf  0x76  0x8f  0x55  0x81  0x2f  0xa5  0xcc  0x03     0xe7  0xea  0x17  0xf2  0x07  0x45  0x40  0x40  0xd8
 0xd2  0xd4  0x25  0xcc  0x41  0xc6  0x87  0x7e  0xfd  …  0x5a  0xe6  0xed  0x28  0x26  0x8b  0x39  0x3b  0x4b
 0xb3  0xbe  0x08  0xdb  0x73  0x3d  0x58  0x04  0xda     0x7b  0xf7  0xab  0x1f  0x2d  0x7b  0x71  0x12  0xdf
 0x6f  0x86  0x20  0x90  0xa5  0x0f  0xc7  0xeb  0x79     0x19  0x92  0x74  0x59  0x4b  0xfe  0xe2  0xb9  0xef
 0x4b  0x93  0x7c  0x02  0x4f  0x40  0xad  0xe3  0x4f     0x9c  0x9c  0x69  0xd1  0xf8  0xd9  0x9e  0x00  0x70
 0x77  0x5d  0x05  0xa6  0x2c  0xaa  0x9d  0xf6  0x8d     0xa9  0x4e  0x46  0x70  0xd9  0x47  0x80  0x06  0x7e
 0x6e  0x7e  0x0f  0x3c  0xe7  0xaf  0x12  0xbf  0x0a  …  0x3f  0xaf  0xe8  0x57  0x26  0x4b  0x2c  0x3f  0x01
 0x72  0xb1  0xea  0xde  0x97  0x1d  0xf4  0x4c  0x89     0x47  0x98  0xc5  0xb6  0x47  0xaf  0x95  0xb1  0x74
 0xc6  0x2b  0x51  0x95  0x30  0xab  0xdc  0x29  0x79     0x5c  0x7b  0xc3  0xf4  0x6a  0xa6  0x09  0x39  0x96
 0xeb  0xef  0x6f  0x70  0x8d  0x1f  0xb9  0x95  0x4e     0xd0  0xf5  0x68  0x0a  0x04  0x63  0x5b  0x45  0xf5
 0xef  0xca  0xb7  0xd4  0x31  0x14  0x34  0x96  0x0c     0x1e  0x6a  0xce  0xf2  0xa3  0xa0  0xbe  0x92  0x9c
 0xda  0x91  0x53  0xd1  0x43  0xfa  0x59  0x7a  0x0c  …  0x0f  0x7a  0xa0  0x4a  0x19  0xc6  0xd3  0xbb  0x7a
 0x9a  0x81  0xdb  0xee  0xce  0x7e  0x4a  0xb5  0x2a     0x3c  0x3e  0xaa  0xdc  0xa6  0xd5  0xae  0x23  0xb2
 0x82  0x2b  0xab  0x06  0xfd  0x8a  0x4a  0xba  0x80     0xb6  0x1a  0x62  0xa0  0x29  0x97  0x61  0x6e  0xf7
 0xb8  0xe6  0x0d  0x21  0x38  0x3a  0x97  0x55  0x58     0x46  0x01  0xe1  0x82  0x34  0xa3  0xfa  0x54  0xb3
 0x09  0xc7  0x2f  0x7b  0x82  0x0c  0x26  0x4d  0xa4     0x1e  0x64  0xc2  0x55  0x41  0x6b  0x14  0x5c  0x0b
 0xf1  0x2c  0x3c  0x0a  0xf1  0x76  0xd4  0x57  0x42  …  0x44  0xb1  0xac  0xb4  0xa2  0x40  0x1e  0xbb  0x44
 0xf8  0x0d  0x6d  0x09  0xb0  0x80  0xe3  0x5e  0x18     0xb3  0x43  0x22  0x82  0x0e  0x50  0xfb  0xf6  0x7b
 0xf0  0x32  0x02  0x28  0x36  0x00  0x4f  0x84  0x2b     0xe8  0xcc  0x89  0x07  0x2f  0xf4  0xcb  0x41  0x53
 0x53  0x9b  0x01  0xf3  0xb2  0x13  0x6a  0x43  0x88     0x22  0xd8  0x33  0xa2  0xab  0xaf  0xe1  0x02  0xf7
 0x59  0x60  0x4a  0x1a  0x9c  0x29  0xb1  0x1b  0xea     0xe9  0xd6  0x07  0x78  0xc6  0xdf  0x16  0xff  0x87
 0xba  0x98  0xff  0x98  0xc3  0xa3  0x7d  0x7c  0x75  …  0xfe  0x75  0x4d  0x43  0x8e  0x5e  0x32  0xb0  0x97
 0x7b  0xc9  0xcf  0x4c  0x99  0xad  0xf1  0x0e  0x0d     0x9f  0xf2  0x92  0x75  0x86  0xd6  0x08  0x74  0x8d
 0x7c  0xd4  0xe7  0x53  0xd3  0x23  0x25  0xce  0x3a     0x19  0xdb  0x14  0xa2  0xf1  0x01  0xd4  0x27  0x20
 0x2a  0x63  0x51  0xcd  0xab  0xc3  0xb5  0xc1  0x74     0xa5  0xa4  0xe1  0xfa  0x13  0xab  0x1f  0x8f  0x9a
 0x93  0xbe  0xf4  0x54  0x2b  0xb9  0x41  0x9d  0xa8     0xbf  0xb7  0x2b  0x1c  0x09  0x36  0xa5  0x7b  0xdc
 0xdc  0x93  0x23  0xf8  0x90  0xaf  0xfb  0xd1  0xcc  …  0x54  0x09  0x8c  0x14  0xfe  0xa7  0x5d  0xd7  0x6d
 0xaf  0x93  0xa2  0x29  0xf9  0x5b  0x24  0xd5  0x2a     0xf1  0x7f  0x3a  0xf5  0x8f  0xd4  0x6e  0x67  0x5b
```
"""
function _packbits(bitsarray::AbstractArray{Bool, 3})
    nbits, dim, batch_size = size(bitsarray)
    dim % 8 == 0 ||
        throw(DomainError("dim should be a multiple of 8!"))
    bitsarray_packed = reinterpret(UInt8, BitArray(vec(bitsarray)).chunks)
    reshape(bitsarray_packed[1:(prod(size(bitsarray)) >> 3)],
        ((dim >> 3) * nbits, batch_size))
end

"""
# Examples

```julia-repl
julia> using ColBERT: _unpackbits;

julia> using Random; Random.seed!(0);

julia> dim, nbits = 128, 2;

julia> bitsarray = rand(Bool, nbits, dim, 200000);

julia> packedbits = _packbits(bitsarray);

julia> unpackedarray = _unpackbits(packedbits, nbits);

julia> isequal(bitsarray, unpackedarray)
```
"""
function _unpackbits(packedbits::AbstractMatrix{UInt8}, nbits::Int)
    prod(size(packedbits, 1)) % nbits == 0 ||
        throw(DomainError("The first dimension of packbits should be " *
                          "a multiple of nbits!"))              # resultant matrix will have an nbits-wide dimension
    _, batch_size = size(packedbits)
    dim = div(size(packedbits, 1), nbits) << 3
    pad_amt = 64 - length(vec(packedbits)) % 64
    chunks = reinterpret(
        UInt64, [vec(packedbits); repeat([zero(UInt8)], pad_amt)])
    bitsvector = BitVector(undef, length(chunks) << 6)
    bitsvector.chunks = chunks
    reshape(
        bitsvector[1:(length(vec(packedbits)) << 3)], nbits, dim, batch_size)
end

"""
    binarize(dim::Int, nbits::Int, bucket_cutoffs::Vector{Float32},
        residuals::AbstractMatrix{Float32})

Convert a matrix of residual vectors into a matrix of integer residual vector
using `nbits` bits.

# Arguments

  - `dim`: The embedding dimension (see [`ColBERTConfig`](@ref)).
  - `nbits`: Number of bits to compress the residuals into.
  - `bucket_cutoffs`: Cutoffs used to determine residual buckets.
  - `residuals`: The matrix of residuals ot be compressed.

# Returns

A `AbstractMatrix{UInt8}` of compressed integer residual vectors.

# Examples

```julia-repl
julia> using ColBERT: binarize;

julia> using Statistics, Random;

julia> Random.seed!(0);

julia> dim, nbits = 128, 2;           # encode residuals in 2 bits

julia> residuals = rand(Float32, dim, 200000);

julia> quantiles = collect(0:(2^nbits - 1)) / 2^nbits;

julia> bucket_cutoffs = Float32.(quantile(residuals, quantiles[2:end]))
3-element Vector{Float32}:
 0.2502231
 0.5001043
 0.75005275

julia> binarize(dim, nbits, bucket_cutoffs, residuals)
32×200000 Matrix{UInt8}:
 0xb4  0xa2  0x0f  0xd5  0xe2  0xd3  0x03  0xbe  0xe3  …  0x44  0xf5  0x8c  0x62  0x59  0xdc  0xc9  0x9e  0x57
 0xce  0x7e  0x23  0xd8  0xea  0x96  0x23  0x3e  0xe1     0xfb  0x29  0xa5  0xab  0x28  0xc3  0xed  0x60  0x90
 0xb1  0x3e  0x96  0xc9  0x84  0x73  0x2c  0x28  0x22     0x27  0x6e  0xca  0x19  0xcd  0x9f  0x1a  0xf4  0xe4
 0xd8  0x85  0x26  0xe2  0xf8  0xfc  0x59  0xef  0x9a     0x51  0xcf  0x06  0x09  0xec  0x0f  0x96  0x94  0x9d
 0xa7  0xfe  0xe2  0x9a  0xa1  0x5e  0xb0  0xd3  0x98     0x41  0x64  0x7b  0x0c  0xa6  0x69  0x26  0x35  0x05
 0x12  0x66  0x0c  0x17  0x05  0xff  0xf2  0x35  0xc0  …  0xa6  0xb7  0xda  0x20  0xb4  0xfe  0x33  0xfc  0xa1
 0x1b  0xa5  0xbc  0xa0  0xc7  0x1c  0xdc  0x43  0x12     0x38  0x81  0x12  0xb1  0x53  0x52  0x50  0x92  0x41
 0x5b  0xea  0xbe  0x84  0x81  0xed  0xf5  0x83  0x7d     0x4a  0xc8  0x7f  0x95  0xab  0x34  0xcb  0x35  0x15
 0xd3  0x0a  0x18  0xc8  0xea  0x34  0x31  0xcc  0x79     0x39  0x3c  0xec  0xe2  0x6a  0xb2  0x59  0x62  0x74
 0x1b  0x01  0xee  0xe7  0xda  0xa9  0xe4  0xe6  0xc5     0x75  0x10  0xa1  0xe1  0xe5  0x50  0x23  0xfe  0xa3
 0xe8  0x38  0x28  0x7c  0x9f  0xd5  0xf7  0x69  0x73  …  0x4e  0xbc  0x52  0xa0  0xca  0x8b  0xe9  0xaf  0xae
 0x2a  0xa2  0x12  0x1c  0x03  0x21  0x6a  0x6e  0xdb     0xa3  0xe3  0x62  0xb9  0x69  0xc0  0x39  0x48  0x9a
 0x76  0x44  0xce  0xd7  0xf7  0x02  0xbd  0xa1  0x7f     0xee  0x5d  0xea  0x9e  0xbe  0x78  0x51  0xbc  0xa3
 0xb2  0xe6  0x09  0x33  0x5b  0xd1  0xad  0x1e  0x9e     0x2c  0x36  0x09  0xd3  0x60  0x81  0x0f  0xe0  0x9e
 0xb8  0x18  0x94  0x0a  0x83  0xd0  0x01  0xe1  0x0f     0x76  0x35  0x6d  0x87  0xfe  0x9e  0x9c  0x69  0xe8
 0x8c  0x6c  0x24  0xf5  0xa9  0xe2  0xbd  0x21  0x83  …  0x1d  0x77  0x11  0xea  0xc1  0xc8  0x09  0xd7  0x4b
 0x97  0x23  0x9f  0x7a  0x8a  0xd1  0x34  0xc6  0xe7     0xe2  0xd0  0x46  0xab  0xbe  0xb3  0x92  0xeb  0xd8
 0x10  0x6f  0xce  0x60  0x17  0x2a  0x4f  0x4a  0xb3     0xde  0x79  0xea  0x28  0xa7  0x08  0x68  0x81  0x9c
 0xae  0xc9  0xc8  0xbf  0x48  0x33  0xa3  0xca  0x8d     0x78  0x4e  0x0e  0xe2  0xe2  0x23  0x08  0x47  0xe6
 0x41  0x29  0x8e  0xff  0x66  0xcc  0xd8  0x58  0x59     0x92  0xd8  0xef  0x9c  0x3c  0x51  0xd4  0x65  0x64
 0xb5  0xc4  0x2d  0x30  0x14  0x54  0xd4  0x79  0x62  …  0xff  0xc1  0xed  0xe4  0x62  0xa4  0x12  0xb7  0x47
 0xcf  0x9a  0x9a  0xd7  0x6f  0xdf  0xad  0x3a  0xf8     0xe5  0x63  0x85  0x0f  0xaf  0x62  0xab  0x67  0x86
 0x3e  0xc7  0x92  0x54  0x8d  0xef  0x0b  0xd5  0xbb     0x64  0x5a  0x4d  0x10  0x2e  0x8f  0xd4  0xb0  0x68
 0x7e  0x56  0x3c  0xb5  0xbd  0x63  0x4b  0xf4  0x8a     0x66  0xc7  0x1a  0x39  0x20  0xa4  0x50  0xac  0xed
 0x3c  0xbc  0x81  0x67  0xb8  0xaf  0x84  0x38  0x8e     0x6e  0x8f  0x3b  0xaf  0xae  0x03  0x0a  0x53  0x55
 0x3d  0x45  0x76  0x98  0x7f  0x34  0x7d  0x23  0x29  …  0x24  0x3a  0x6b  0x8a  0xb4  0x3c  0x2d  0xe2  0x3a
 0xed  0x41  0xe6  0x86  0xf3  0x61  0x12  0xc5  0xde     0xd1  0x26  0x11  0x36  0x57  0x6c  0x35  0x38  0xe2
 0x11  0x57  0x82  0x9b  0x19  0x1f  0x56  0xd7  0x06     0x1e  0x2b  0xd9  0x76  0xa1  0x68  0x27  0xb1  0xde
 0x89  0xb3  0xeb  0x86  0xbb  0x57  0xda  0xd3  0x5b     0x0e  0x79  0x4c  0x8c  0x57  0x3d  0xf0  0x98  0xb7
 0xbf  0xc2  0xac  0xf0  0xed  0x69  0x0e  0x19  0x12     0xfe  0xab  0xcd  0xfc  0x72  0x76  0x5c  0x58  0x8b
 0xe9  0x7b  0xf6  0x22  0xa0  0x60  0x23  0xc9  0x33  …  0x77  0xc7  0xdf  0x8a  0xb9  0xef  0xe3  0x03  0x8a
 0x6b  0x26  0x08  0x53  0xc3  0x17  0xc4  0x33  0x2e     0xc6  0xb8  0x1e  0x54  0xcd  0xeb  0xb9  0x5f  0x38
```
"""
function binarize(dim::Int, nbits::Int, bucket_cutoffs::Vector{Float32},
        residuals::AbstractMatrix{Float32})
    # bucket indices will be encoded in nbits bits
    # so they will be in the range [0, length(bucket_cutoffs) - 1]
    # so length(bucket_cutoffs) should be 2^nbits - 1
    dim % 8 == 0 || throw(DomainError("dims should be a multiple of 8!"))
    length(bucket_cutoffs) == (1 << nbits) - 1 ||
        throw(DomainError("length(bucket_cutoffs) should be 2^nbits - 1!"))

    # get the bucket indices
    bucket_indices = _bucket_indices(residuals, bucket_cutoffs)                     # (dim, batch_size)

    # representing each index in nbits bits
    bucket_indices = _binarize(bucket_indices, nbits)                               # (nbits, dim, batch_size)

    # pack bits into UInt8's for each embedding
    residuals_packed = _packbits(bucket_indices)
    residuals_packed
end

"""
    compress(centroids::Matrix{Float32}, bucket_cutoffs::Vector{Float32},
        dim::Int, nbits::Int, embs::AbstractMatrix{Float32})

Compress a matrix of embeddings into a compact representation.

All embeddings are compressed to their nearest centroid IDs and
their quantized residual vectors (where the quantization is done
in `nbits` bits). If `emb` denotes an embedding and `centroid`
is is nearest centroid, the residual vector is defined to be
`emb - centroid`.

# Arguments

  - `centroids`: The matrix of centroids.
  - `bucket_cutoffs`: Cutoffs used to determine residual buckets.
  - `dim`: The embedding dimension (see [`ColBERTConfig`](@ref)).
  - `nbits`: Number of bits to compress the residuals into.
  - `embs`: The input embeddings to be compressed.

# Returns

A tuple containing a vector of codes and the compressed residuals matrix.

# Examples

```julia-repl
julia> using ColBERT: compress;

julia> using Random; Random.seed!(0);

julia> nbits, dim = 2, 128;

julia> embs = rand(Float32, dim, 100000);

julia> centroids = embs[:, randperm(size(embs, 2))[1:10000]];

julia> bucket_cutoffs = Float32.(sort(rand(2^nbits - 1)));
3-element Vector{Float32}:
 0.08594067
 0.0968812
 0.44113323

julia> @time codes, compressed_residuals = compress(
    centroids, bucket_cutoffs, dim, nbits, embs);
  4.277926 seconds (1.57 k allocations: 4.238 GiB, 6.46% gc time)
```
"""
function compress(centroids::Matrix{Float32}, bucket_cutoffs::Vector{Float32},
        dim::Int, nbits::Int, embs::AbstractMatrix{Float32}; bsize::Int = 10000)
    codes = zeros(UInt32, size(embs, 2))
    compressed_residuals = Matrix{UInt8}(
        undef, div(dim, 8) * nbits, size(embs, 2))
    for offset in 1:bsize:size(embs, 2)
        offset_end = min(size(embs, 2), offset + bsize - 1)
        @views batch_embs = embs[:, offset:offset_end]
        @views batch_codes = codes[offset:offset_end]
        @views batch_compressed_residuals = compressed_residuals[
            :, offset:offset_end]
        compress_into_codes!(batch_codes, centroids, batch_embs)
        @views batch_centroids = centroids[:, batch_codes]
        batch_residuals = batch_embs - batch_centroids
        batch_compressed_residuals .= binarize(
            dim, nbits, bucket_cutoffs, batch_residuals)
    end
    codes, compressed_residuals
end

"""

# Examples

```julia-repl
julia> using ColBERT: binarize, decompress_residuals;

julia> using Statistics, Flux, CUDA, Random;

julia> Random.seed!(0);

julia> dim, nbits = 128, 2;           # encode residuals in 5 bits

julia> residuals = rand(Float32, dim, 200000);

julia> quantiles = collect(0:(2^nbits - 1)) / 2^nbits;

julia> bucket_cutoffs = Float32.(quantile(residuals, quantiles[2:end]))
3-element Vector{Float32}:
 0.2502231
 0.5001043
 0.75005275

julia> bucket_weights = Float32.(quantile(residuals, quantiles .+ 0.5 / 2^nbits))
4-element Vector{Float32}:
 0.1250611
 0.37511465
 0.62501323
 0.87501866

julia> binary_residuals = binarize(dim, nbits, bucket_cutoffs, residuals);

julia> decompressed_residuals = decompress_residuals(
    dim, nbits, bucket_weights, binary_residuals)
128×200000 Matrix{Float32}:
 0.125061  0.625013  0.875019  0.375115  0.625013  0.875019  …  0.375115  0.125061  0.375115  0.625013  0.875019
 0.375115  0.125061  0.875019  0.375115  0.125061  0.125061     0.625013  0.875019  0.625013  0.875019  0.375115
 0.875019  0.625013  0.125061  0.375115  0.625013  0.375115     0.375115  0.375115  0.125061  0.375115  0.375115
 0.625013  0.625013  0.125061  0.875019  0.875019  0.875019     0.375115  0.875019  0.875019  0.625013  0.375115
 0.625013  0.625013  0.875019  0.125061  0.625013  0.625013     0.125061  0.875019  0.375115  0.125061  0.125061
 0.875019  0.875019  0.125061  0.625013  0.625013  0.375115  …  0.625013  0.125061  0.875019  0.125061  0.125061
 0.125061  0.875019  0.625013  0.375115  0.625013  0.375115     0.625013  0.125061  0.625013  0.625013  0.375115
 0.875019  0.375115  0.125061  0.875019  0.875019  0.625013     0.125061  0.875019  0.875019  0.375115  0.625013
 0.375115  0.625013  0.625013  0.375115  0.125061  0.875019     0.375115  0.875019  0.625013  0.125061  0.125061
 0.125061  0.875019  0.375115  0.625013  0.375115  0.125061     0.875019  0.875019  0.625013  0.375115  0.375115
 0.875019  0.875019  0.375115  0.125061  0.125061  0.875019  …  0.125061  0.375115  0.375115  0.875019  0.625013
 0.625013  0.125061  0.625013  0.875019  0.625013  0.375115     0.875019  0.625013  0.125061  0.875019  0.875019
 0.125061  0.375115  0.625013  0.625013  0.125061  0.125061     0.125061  0.875019  0.625013  0.125061  0.375115
 0.625013  0.375115  0.375115  0.125061  0.625013  0.875019     0.875019  0.875019  0.375115  0.375115  0.875019
 0.375115  0.125061  0.625013  0.625013  0.875019  0.875019     0.625013  0.125061  0.375115  0.375115  0.375115
 0.875019  0.625013  0.125061  0.875019  0.875019  0.875019  …  0.875019  0.125061  0.625013  0.625013  0.625013
 0.875019  0.625013  0.625013  0.625013  0.375115  0.625013     0.625013  0.375115  0.625013  0.375115  0.375115
 0.375115  0.875019  0.125061  0.625013  0.125061  0.875019     0.375115  0.625013  0.375115  0.375115  0.375115
 0.625013  0.875019  0.625013  0.375115  0.625013  0.375115     0.625013  0.625013  0.625013  0.875019  0.125061
 0.625013  0.875019  0.875019  0.625013  0.625013  0.375115     0.625013  0.375115  0.125061  0.125061  0.125061
 0.625013  0.625013  0.125061  0.875019  0.375115  0.875019  …  0.125061  0.625013  0.875019  0.125061  0.375115
 0.125061  0.375115  0.875019  0.375115  0.375115  0.875019     0.375115  0.875019  0.125061  0.875019  0.125061
 0.375115  0.625013  0.125061  0.375115  0.125061  0.875019     0.875019  0.875019  0.875019  0.875019  0.625013
 0.125061  0.375115  0.125061  0.125061  0.125061  0.875019     0.625013  0.875019  0.125061  0.875019  0.625013
 0.875019  0.375115  0.125061  0.125061  0.875019  0.125061     0.875019  0.625013  0.125061  0.625013  0.375115
 0.625013  0.375115  0.875019  0.125061  0.375115  0.875019  …  0.125061  0.125061  0.125061  0.125061  0.125061
 0.375115  0.625013  0.875019  0.625013  0.125061  0.375115     0.375115  0.375115  0.375115  0.375115  0.125061
 ⋮                                                 ⋮         ⋱  ⋮
 0.875019  0.375115  0.375115  0.625013  0.875019  0.375115     0.375115  0.875019  0.875019  0.125061  0.625013
 0.875019  0.125061  0.875019  0.375115  0.875019  0.875019     0.875019  0.875019  0.625013  0.625013  0.875019
 0.125061  0.375115  0.375115  0.625013  0.375115  0.125061     0.625013  0.125061  0.125061  0.875019  0.125061
 0.375115  0.375115  0.625013  0.625013  0.875019  0.375115     0.875019  0.125061  0.375115  0.125061  0.625013
 0.875019  0.125061  0.375115  0.375115  0.125061  0.125061  …  0.375115  0.875019  0.375115  0.625013  0.125061
 0.625013  0.125061  0.625013  0.125061  0.875019  0.625013     0.375115  0.625013  0.875019  0.875019  0.625013
 0.875019  0.375115  0.875019  0.625013  0.875019  0.375115     0.375115  0.375115  0.125061  0.125061  0.875019
 0.375115  0.875019  0.625013  0.875019  0.375115  0.875019     0.375115  0.125061  0.875019  0.375115  0.625013
 0.125061  0.375115  0.125061  0.625013  0.625013  0.875019     0.125061  0.625013  0.375115  0.125061  0.875019
 0.375115  0.375115  0.125061  0.375115  0.375115  0.375115  …  0.625013  0.625013  0.625013  0.875019  0.375115
 0.125061  0.375115  0.625013  0.625013  0.125061  0.125061     0.625013  0.375115  0.125061  0.625013  0.875019
 0.375115  0.875019  0.875019  0.625013  0.875019  0.875019     0.875019  0.375115  0.125061  0.125061  0.875019
 0.625013  0.125061  0.625013  0.375115  0.625013  0.375115     0.375115  0.875019  0.125061  0.625013  0.375115
 0.125061  0.875019  0.625013  0.125061  0.875019  0.375115     0.375115  0.875019  0.875019  0.375115  0.875019
 0.625013  0.625013  0.875019  0.625013  0.625013  0.375115  …  0.375115  0.125061  0.875019  0.625013  0.625013
 0.875019  0.625013  0.125061  0.125061  0.375115  0.375115     0.625013  0.625013  0.125061  0.125061  0.875019
 0.875019  0.125061  0.875019  0.125061  0.875019  0.625013     0.125061  0.375115  0.875019  0.625013  0.625013
 0.875019  0.125061  0.625013  0.875019  0.625013  0.625013     0.875019  0.875019  0.375115  0.375115  0.125061
 0.625013  0.875019  0.625013  0.875019  0.875019  0.375115     0.375115  0.375115  0.375115  0.375115  0.625013
 0.375115  0.875019  0.625013  0.625013  0.125061  0.125061  …  0.375115  0.875019  0.875019  0.875019  0.625013
 0.625013  0.625013  0.375115  0.125061  0.125061  0.125061     0.625013  0.875019  0.125061  0.125061  0.625013
 0.625013  0.875019  0.875019  0.625013  0.625013  0.625013     0.875019  0.625013  0.625013  0.125061  0.125061
 0.875019  0.375115  0.875019  0.125061  0.625013  0.375115     0.625013  0.875019  0.875019  0.125061  0.625013
 0.875019  0.625013  0.125061  0.875019  0.875019  0.875019     0.375115  0.875019  0.375115  0.875019  0.125061
 0.625013  0.375115  0.625013  0.125061  0.125061  0.375115  …  0.875019  0.625013  0.625013  0.875019  0.625013
 0.625013  0.625013  0.125061  0.375115  0.125061  0.375115     0.125061  0.625013  0.875019  0.375115  0.875019
 0.375115  0.125061  0.125061  0.375115  0.875019  0.125061     0.875019  0.875019  0.625013  0.375115  0.125061
```
"""
function decompress_residuals(
        dim::Int, nbits::Int, bucket_weights::Vector{Float32},
        binary_residuals::AbstractMatrix{UInt8})
    dim % 8 == 0 || throw(DomainError("dim should be a multiple of 8!"))
    size(binary_residuals, 1) == div(dim, 8) * nbits ||
        throw(DomainError("The dimension each residual in binary_residuals " *
                          "should be (dim / 8) * nbits!"))
    length(bucket_weights) == (1 << nbits) ||
        throw(DomainError("bucket_weights should have length 2^nbits!"))

    # unpacking bits
    unpacked_bits = _unpackbits(binary_residuals, nbits)        # (nbits, dim, batch_size)

    # unbinarze the packed bits, and add 1 to get bin indices
    unpacked_bits = _unbinarize(unpacked_bits)
    unpacked_bits = unpacked_bits .+ 1                          # (dim, batch_size)

    # get the residuals from the bucket weights
    all(in(1:length(bucket_weights)), unpacked_bits) ||
        throw(BoundsError("All the unpacked indices in binary_residuals should " *
                          "be in range 1:length(bucket_weights)!"))
    decompressed_residuals = bucket_weights[unpacked_bits]
    decompressed_residuals
end

"""
# Examples

```julia-repl
julia> using ColBERT: compress, decompress;

julia> using Random; Random.seed!(0);

julia> nbits, dim = 2, 128;

julia> embs = rand(Float32, dim, 100000);

julia> centroids = embs[:, randperm(size(embs, 2))[1:10000]];

julia> bucket_cutoffs = Float32.(sort(rand(2^nbits - 1)))
3-element Vector{Float32}:
 0.08594067
 0.0968812
 0.44113323

julia> bucket_weights = Float32.(sort(rand(2^nbits)));
4-element Vector{Float32}:
 0.10379179
 0.25756857
 0.27798286
 0.47973529

julia> @time codes, compressed_residuals = compress(
    centroids, bucket_cutoffs, dim, nbits, embs);
  4.277926 seconds (1.57 k allocations: 4.238 GiB, 6.46% gc time)

julia> @time decompressed_embeddings = decompress(
    dim, nbits, centroids, bucket_weights, codes, compressed_residuals);
0.237170 seconds (276.40 k allocations: 563.049 MiB, 50.93% compilation time)
```
"""
function decompress(
        dim::Int, nbits::Int, centroids::Matrix{Float32},
        bucket_weights::Vector{Float32},
        codes::Vector{UInt32}, residuals::AbstractMatrix{UInt8}; bsize::Int = 10000)
    length(codes) == size(residuals, 2) ||
        throw(DomainError("The number of codes should be equal to the number of " *
                          "residual embeddings!"))
    all(in(1:size(centroids, 2)), codes) ||
        throw(DomainError("All the codes must be in the valid range of centroid " *
                          "IDs!"))
    embeddings = Matrix{Float32}(undef, dim, length(codes))
    for batch_offset in 1:bsize:length(codes)
        batch_offset_end = min(length(codes), batch_offset + bsize - 1)
        @views batch_embeddings = embeddings[
            :, batch_offset:batch_offset_end]
        @views batch_codes = codes[batch_offset:batch_offset_end]
        @views batch_residuals = residuals[:, batch_offset:batch_offset_end]
        @views centroids_ = centroids[:, batch_codes]
        residuals_ = decompress_residuals(
            dim, nbits, bucket_weights, batch_residuals)

        batch_embeddings .= centroids_ + residuals_
        _normalize_array!(batch_embeddings, dims = 1)
    end
    embeddings
end
