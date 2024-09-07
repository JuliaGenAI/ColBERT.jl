# """
#     _sort_by_length(
#         integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool}, bsize::Int)
#
# Sort sentences by number of attended tokens, if the number of sentences is larger than `bsize`.
#
# # Arguments
#
#   - `integer_ids`: The token IDs of documents to be sorted.
#   - `integer_mask`: The attention masks of the documents to be sorted (attention masks are just bits).
#   - `bsize`: The size of batches to be considered.
#
# # Returns
#
# Depending upon `bsize`, the following are returned:
#
#   - If the number of documents (second dimension of `integer_ids`) is atmost `bsize`, then the
#     `integer_ids` and `integer_mask` are returned unchanged.
#   - If the number of documents is larger than `bsize`, then the passages are first sorted
#     by the number of attended tokens (figured out from the `integer_mask`), and then the
#     sorted arrays `integer_ids`, `integer_mask` are returned, along with a list of
#     `reverse_indices`, i.e a mapping from the documents to their indices in the original
#     order.
# """
# function _sort_by_length(
#         integer_ids::AbstractMatrix{Int32}, bitmask::AbstractMatrix{Bool}, batch_size::Int)
#     size(integer_ids, 2) <= batch_size &&
#         return integer_ids, bitmask, Vector(1:size(integer_ids, 2))
#     lengths = vec(sum(bitmask; dims = 1))                   # number of attended tokens in each passage
#     indices = sortperm(lengths)                             # get the indices which will sort lengths
#     reverse_indices = sortperm(indices)                     # invert the indices list
#     @assert integer_ids isa AbstractMatrix{Int32} "$(typeof(integer_ids))"
#     @assert bitmask isa BitMatrix "$(typeof(bitmask))"
#     @assert reverse_indices isa Vector{Int} "$(typeof(reverse_indices))"
#     integer_ids[:, indices], bitmask[:, indices], reverse_indices
# end

function compute_distances_kernel!(batch_distances::AbstractMatrix{Float32},
        batch_data::AbstractMatrix{Float32},
        centroids::AbstractMatrix{Float32})
    isequal(size(batch_distances), (size(centroids, 2), size(batch_data, 2))) ||
        throw(DimensionMismatch("batch_distances should have size " *
                                "(num_centroids, point_bsize)!"))
    isequal(size(batch_data, 1), size(centroids, 1)) ||
        throw(DimensionMismatch("batch_data and centroids should have " *
                                "the same embedding dimension!"))

    batch_distances .= 0.0f0
    # Compute squared distances: (a-b)^2 = a^2 + b^2 - 2ab
    # a^2 term
    sum_sq_data = sum(batch_data .^ 2, dims = 1)                        # (1, point_bsize)
    # b^2 term
    sum_sq_centroids = sum(centroids .^ 2, dims = 1)'                   # (num_centroids, 1)
    # -2ab term
    mul!(batch_distances, centroids', batch_data, -2.0f0, 1.0f0)        # (num_centroids, point_bsize)
    # Compute (a-b)^2 = a^2 + b^2 - 2ab
    batch_distances .+= sum_sq_centroids
    batch_distances .+= sum_sq_data
end

function update_centroids_kernel!(new_centroids::AbstractMatrix{Float32},
        batch_data::AbstractMatrix{Float32},
        batch_one_hot::AbstractMatrix{Float32})
    isequal(
        size(new_centroids), (size(batch_data, 1), (size(batch_one_hot, 1)))) ||
        throw(DimensionMismatch("new_centroids should have the right shape " *
                                "for multiplying batch_data and batch_one_hot! "))
    mul!(new_centroids, batch_data, batch_one_hot', 1.0f0, 1.0f0)
end

function assign_clusters_kernel!(batch_assignments::AbstractVector{Int32},
        batch_distances::AbstractMatrix{Float32})
    length(batch_assignments) == size(batch_distances, 2) ||
        throw(DimensionMismatch("length(batch_assignments) " *
                                "should be equal to the point " *
                                "batch size of batch_distances!"))
    _, min_indices = findmin(batch_distances, dims = 1)
    batch_assignments .= getindex.(min_indices, 1) |> vec
end

function onehot_encode!(batch_one_hot::AbstractArray{Float32},
        batch_assignments::AbstractVector{Int32}, k::Int)
    isequal(size(batch_one_hot), (k, length(batch_assignments))) ||
        throw(DimensionMismatch("batch_one_hot should have shape " *
                                "(k, length(batch_assignments))!"))
    col_indices = similar(batch_assignments, length(batch_assignments))     # respects device
    copyto!(col_indices, collect(1:length(batch_assignments)))
    batch_one_hot[batch_assignments .+ (col_indices .- 1) .* k] .= 1
end

"""
# Examples

```julia-repl
julia> using ColBERT, Flux, CUDA, Random;

julia> d, n, k = 100, 2000000, 50000 # dimensions, number of points, number of clusters
(100, 2000000, 50000)

julia> data = rand(Float32, d, n) |> Flux.gpu;           # around 800MB

julia> centroids = data[:, randperm(n)[1:k]];

julia> point_bsize = 1000;         # adjust according to your GPU/CPU memory

julia> @time assignments = ColBERT.kmeans_gpu_onehot!(
           data, centroids, k; max_iters = 2, point_bsize = point_bsize)
[ Info: Iteration 1/2, max delta: 0.6814487
[ Info: Iteration 2/2, max delta: 0.28856403
 76.381827 seconds (5.76 M allocations: 606.426 MiB, 4.25% gc time, 0.11% compilation time)
2000000-element Vector{Int32}:
 24360
 10954
 29993
 22113
 19024
 32192
 33033
 32738
 19901
  5142
 23567
 12686
 18894
 23919
  7325
 29809
 27885
 31122
  1457
  9823
 41315
 14311
 21975
 48753
 16162
  7809
 33018
 22410
 26646
  2607
 34833
     ⋮
 15216
 26424
 21939
  9252
  5071
 14570
 22467
 37881
 28239
  8775
 31290
  4625
  7561
  7645
  7277
 36069
 49799
 39307
 10595
  7639
 18879
 12754
  1233
 29389
 24772
 47907
 29380
  1345
  4781
 35313
 30000

julia> centroids
100×50000 CuArray{Float32, 2, CUDA.DeviceMemory}:
 0.573378  0.509291  0.40079   0.614619  0.593501  0.532985  0.79016    0.573517  …  0.544782  0.666605  0.537127  0.490516  0.74021   0.345155  0.613033
 0.710199  0.301702  0.570302  0.302831  0.378944  0.28444   0.577703   0.327737     0.27379   0.352727  0.413396  0.49565   0.685949  0.534816  0.540361
 0.379057  0.424286  0.771943  0.411402  0.319783  0.550557  0.64573    0.679135     0.702826  0.846835  0.608924  0.376951  0.431148  0.642033  0.697345
 0.694464  0.435644  0.422319  0.532234  0.521483  0.627431  0.501389   0.359163     0.328353  0.350925  0.485843  0.437292  0.354213  0.185923  0.427814
 0.221736  0.506781  0.352585  0.678622  0.333673  0.50622   0.463275   0.591525     0.572961  0.473792  0.369353  0.400138  0.733724  0.477619  0.254028
 0.619385  0.51777   0.40583   0.445265  0.224872  0.677207  0.713577   0.620289  …  0.389378  0.487728  0.675865  0.250588  0.614895  0.668617  0.235178
 0.591426  0.395195  0.538931  0.744411  0.533349  0.338823  0.345266   0.327421     0.373282  0.36309   0.681582  0.646208  0.404389  0.251627  0.341416
 0.583477  0.423426  0.247412  0.446173  0.280856  0.614167  0.533047   0.573224     0.45711   0.445103  0.697702  0.474529  0.616773  0.460811  0.286667
 0.49608   0.685452  0.424273  0.683325  0.581213  0.684903  0.382428   0.529762     0.734883  0.71177   0.414117  0.417863  0.543535  0.610839  0.488656
 0.626167  0.540865  0.677231  0.596885  0.378552  0.398865  0.518733   0.497296     0.661245  0.594468  0.288819  0.29435   0.467833  0.722748  0.663824
 0.619386  0.579229  0.441548  0.386045  0.564118  0.646701  0.632154   0.612795  …  0.617854  0.597241  0.490215  0.308035  0.349091  0.486332  0.32071
 0.315375  0.457891  0.642345  0.361314  0.410211  0.380876  0.844302   0.496581     0.726295  0.21279   0.555863  0.468077  0.448128  0.497228  0.688524
 0.302116  0.55576   0.22489   0.50484   0.561481  0.461971  0.605235   0.627733     0.570166  0.536869  0.647504  0.458224  0.27462   0.553473  0.268046
 0.745733  0.403701  0.468518  0.418122  0.533233  0.579005  0.837422   0.538135     0.704916  0.666066  0.571446  0.500032  0.585166  0.555079  0.39484
 0.576735  0.590597  0.312162  0.330425  0.45483   0.279067  0.577954   0.539739     0.644922  0.185377  0.681872  0.36546   0.619736  0.755231  0.818024
 0.548489  0.695465  0.835756  0.478009  0.412736  0.416005  0.118124   0.626901  …  0.313572  0.754964  0.659507  0.677611  0.479118  0.3991    0.622777
 0.285406  0.381637  0.338189  0.544162  0.477955  0.546904  0.309153   0.439008     0.563208  0.346864  0.448714  0.383776  0.55155   0.3148    0.467101
 0.823076  0.652229  0.504614  0.400098  0.357104  0.448227  0.24265    0.696984     0.485136  0.637487  0.643558  0.705938  0.632451  0.424837  0.766686
 0.421668  0.343106  0.530787  0.528398  0.24584   0.699929  0.214073   0.419076     0.331078  0.35033   0.354848  0.46255   0.475431  0.715539  0.688314
 0.779925  0.724435  0.638462  0.482254  0.521571  0.715278  0.621099   0.556042     0.308391  0.492443  0.36217   0.408848  0.73595   0.540198  0.698907
 0.356398  0.544033  0.543013  0.462401  0.402219  0.387093  0.323547   0.373834  …  0.645622  0.674534  0.723415  0.353287  0.613711  0.38006   0.554985
 0.658572  0.401115  0.25994   0.483548  0.52677   0.712259  0.774561   0.438474     0.376936  0.297307  0.455176  0.23899   0.608517  0.76084   0.382525
 0.525316  0.362833  0.361821  0.383153  0.248305  0.401027  0.554528   0.278677     0.415318  0.512563  0.401782  0.674682  0.666895  0.663432  0.378345
 0.580109  0.489022  0.255441  0.590038  0.488305  0.51133   0.508364   0.416333     0.262037  0.348079  0.564498  0.360297  0.702012  0.324764  0.249475
 0.723813  0.548868  0.550225  0.438456  0.455546  0.714484  0.0994013  0.465583     0.590603  0.414145  0.583897  0.41563   0.411714  0.271341  0.440918
 0.62465   0.664534  0.342419  0.648037  0.719117  0.665314  0.256789   0.325002  …  0.636772  0.235229  0.472394  0.656942  0.414241  0.216398  0.799625
 0.409948  0.493941  0.522245  0.38117   0.235328  0.310665  0.557497   0.621436     0.413982  0.577326  0.645292  0.225434  0.430032  0.450371  0.375822
 0.372894  0.635165  0.494829  0.440398  0.380812  0.755357  0.473521   0.487604     0.349699  0.659922  0.626307  0.437899  0.488775  0.404058  0.64511
 0.288256  0.491838  0.338052  0.466105  0.363578  0.456235  0.425795   0.453427     0.226024  0.429285  0.604995  0.403821  0.33844   0.254136  0.42694
 0.314443  0.319862  0.56776   0.652814  0.626939  0.234881  0.274685   0.531139     0.270967  0.547521  0.664938  0.451628  0.531532  0.592488  0.525191
 0.493068  0.306231  0.562287  0.454218  0.199483  0.57302   0.238318   0.567198  …  0.297332  0.460382  0.285109  0.411792  0.356838  0.340022  0.414451
 0.53873   0.258357  0.402785  0.269083  0.594396  0.505856  0.690911   0.738276     0.737582  0.369145  0.409122  0.336054  0.358317  0.392364  0.561769
 0.617347  0.639471  0.333155  0.370546  0.526723  0.293309  0.247984   0.660384     0.647745  0.286011  0.681676  0.624425  0.580846  0.402701  0.297121
 0.496282  0.378267  0.270501  0.475257  0.516464  0.356405  0.175957   0.539904     0.236559  0.58985   0.578107  0.543669  0.563102  0.71473   0.43457
 0.297402  0.476382  0.426692  0.283131  0.626477  0.220255  0.372191   0.615784     0.374197  0.55345   0.495846  0.331621  0.645283  0.578616  0.389071
 0.734077  0.371284  0.826699  0.684061  0.272948  0.693993  0.528874   0.304462  …  0.525932  0.395874  0.500069  0.559787  0.460612  0.798967  0.580689
 ⋮                                                 ⋮                              ⋱                      ⋮
 0.295452  0.589387  0.339522  0.383816  0.63141   0.505792  0.66544    0.479078     0.448193  0.774786  0.607631  0.349403  0.689084  0.619     0.251087
 0.342872  0.684608  0.66651   0.402659  0.424726  0.591997  0.391954   0.667982  …  0.459421  0.376128  0.301928  0.538294  0.530345  0.458879  0.59855
 0.449909  0.409996  0.149798  0.576651  0.290799  0.635566  0.437937   0.511792     0.648198  0.661462  0.61996   0.644484  0.636402  0.527594  0.407358
 0.782475  0.421017  0.69657   0.691838  0.382575  0.805573  0.364693   0.597721     0.652466  0.666937  0.693412  0.490323  0.514455  0.380534  0.427285
 0.314463  0.420641  0.364206  0.348991  0.59921   0.746625  0.617284   0.697596     0.342617  0.45338   0.363351  0.660113  0.674676  0.376416  0.721194
 0.402126  0.588711  0.323173  0.388439  0.34814   0.491494  0.545984   0.648734     0.430481  0.378938  0.309212  0.382807  0.632475  0.367792  0.376823
 0.555737  0.668767  0.490702  0.663971  0.250589  0.445352  0.172075   0.673576  …  0.322794  0.644713  0.394593  0.572583  0.687199  0.662051  0.3559
 0.793682  0.698499  0.67152   0.46898   0.656144  0.353421  0.803591   0.633019     0.803097  0.640827  0.365467  0.679615  0.642185  0.685466  0.296224
 0.428538  0.528681  0.438861  0.625715  0.591183  0.629757  0.456717   0.50485      0.405746  0.437458  0.368839  0.446011  0.488281  0.471933  0.514202
 0.485429  0.738783  0.287516  0.463954  0.188286  0.544762  0.37223    0.58192      0.585194  0.489835  0.506583  0.464377  0.645507  0.804297  0.786932
 0.29249   0.586557  0.608833  0.663233  0.576919  0.267828  0.308029   0.712437     0.533969  0.421972  0.476979  0.530931  0.47962   0.528001  0.621458
 0.279038  0.445135  0.177712  0.515837  0.300508  0.281383  0.400402   0.651     …  0.58635   0.443282  0.657886  0.697657  0.552504  0.329047  0.399654
 0.832609  0.485713  0.600559  0.699044  0.714713  0.606326  0.273329   0.440225     0.623437  0.667127  0.41734   0.767461  0.702767  0.601694  0.506635
 0.297328  0.287248  0.36852   0.657753  0.698171  0.719895  0.238376   0.638514     0.343874  0.373995  0.511818  0.377467  0.389039  0.522639  0.686664
 0.301796  0.737757  0.635025  0.666437  0.393605  0.346305  0.547774   0.689093     0.519264  0.361948  0.718109  0.475808  0.573496  0.514178  0.598478
 0.549563  0.248966  0.364826  0.57668   0.590149  0.533822  0.664503   0.553704     0.284555  0.591084  0.316526  0.660029  0.516786  0.824489  0.689313
 0.247931  0.238425  0.23728   0.516849  0.732181  0.405793  0.724634   0.5149    …  0.380765  0.696078  0.41157   0.642839  0.384414  0.493493  0.552407
 0.606629  0.601705  0.319954  0.533014  0.382539  0.410641  0.29247    0.506377     0.615707  0.501867  0.475531  0.405969  0.333115  0.358202  0.502586
 0.583896  0.619858  0.593031  0.451623  0.58986   0.349512  0.536081   0.298436     0.396871  0.239656  0.406909  0.541055  0.416507  0.547856  0.424243
 0.691322  0.50077   0.323869  0.500225  0.420282  0.436531  0.703267   0.541637     0.539365  0.725134  0.693945  0.676646  0.556313  0.374397  0.583554
 0.701328  0.488743  0.35439   0.613276  0.493706  0.399695  0.728355   0.467517     0.261417  0.575774  0.37854   0.490462  0.461564  0.556492  0.424225
 0.718797  0.550606  0.565344  0.561342  0.355202  0.578364  0.786034   0.562179  …  0.289592  0.183233  0.524043  0.335948  0.333167  0.476679  0.65326
 0.701058  0.380252  0.444291  0.532477  0.540552  0.696061  0.403728   0.58757      0.520714  0.510013  0.547041  0.564867  0.532286  0.501574  0.595203
 0.365637  0.531816  0.565021  0.602144  0.548403  0.764079  0.365481   0.613074     0.360902  0.527056  0.375336  0.544605  0.689852  0.837963  0.459323
 0.288392  0.268179  0.332016  0.689326  0.234238  0.23735   0.756387   0.532537     0.403286  0.471491  0.602447  0.429769  0.293544  0.437438  0.349532
 0.664517  0.31624   0.59785   0.230114  0.376591  0.773395  0.752942   0.636399     0.326092  0.72005   0.333086  0.339832  0.325618  0.461294  0.524966
 0.222333  0.305546  0.673752  0.762977  0.307967  0.312146  0.663083   0.58212   …  0.69865   0.643548  0.640484  0.755733  0.496422  0.649607  0.720769
 0.411979  0.370252  0.237112  0.311196  0.610508  0.447023  0.506591   0.213862     0.721287  0.373431  0.594912  0.621447  0.43674   0.258687  0.560904
 0.617416  0.641325  0.560164  0.313925  0.490977  0.337085  0.714373   0.506699     0.253813  0.470016  0.584523  0.447376  0.51011   0.270167  0.484992
 0.623836  0.324357  0.734953  0.790519  0.455406  0.52695   0.403097   0.446101     0.633619  0.403004  0.694153  0.717927  0.47924   0.576069  0.253169
 0.73859   0.344694  0.183747  0.69547   0.458342  0.481904  0.737565   0.720339     0.447743  0.619669  0.367867  0.34662   0.607812  0.251007  0.509758
 0.530767  0.332264  0.550998  0.364326  0.722955  0.580428  0.490779   0.426905  …  0.793421  0.713281  0.779156  0.54861   0.674266  0.21644   0.493613
 0.343766  0.379023  0.630344  0.744247  0.567047  0.377182  0.73119    0.615484     0.761156  0.264631  0.510148  0.481783  0.453394  0.410757  0.335559
 0.568994  0.332011  0.631839  0.455666  0.631383  0.453398  0.654253   0.276721     0.268318  0.658483  0.523244  0.549092  0.485578  0.342858  0.436086
 0.686312  0.268361  0.414777  0.437959  0.617892  0.582933  0.649577   0.342277     0.70994   0.435503  0.24157   0.668377  0.412632  0.667489  0.544822
 0.446142  0.527333  0.160024  0.325712  0.330222  0.368513  0.661516   0.431168     0.44104   0.665175  0.286649  0.534375  0.67307   0.571995  0.3261
```
"""
function kmeans_gpu_onehot!(
        data::AbstractMatrix{Float32}, centroids::AbstractMatrix{Float32}, k::Int; max_iters::Int = 10,
        tol::Float32 = 1.0f-4, point_bsize::Int = 1000)
    # TODO: move point_bsize to config?
    size(centroids, 2) == k ||
        throw(DimensionMismatch("size(centroids, 2) must be k!"))

    # randomly initialize centroids
    centroids .= data[:, randperm(size(data, 2))[1:k]]

    # allocations
    d, n = size(data)  # dimension, number of inputs
    assignments = Vector{Int32}(undef, n) |> Flux.gpu
    distances = Matrix{Float32}(undef, k, point_bsize) |> Flux.gpu
    new_centroids = Matrix{Float32}(undef, d, k) |> Flux.gpu
    counts = Vector{Int32}(undef, k) |> Flux.gpu
    one_hot = Matrix{Float32}(undef, k, point_bsize) |> Flux.gpu

    for iter in 1:max_iters
        new_centroids .= 0.0f0
        counts .= 0

        for batch_start in 1:point_bsize:n
            batch_end = min(n, batch_start + point_bsize - 1)
            batch_size_actual = batch_end - batch_start + 1

            batch_data = @view(data[:, batch_start:batch_end])
            batch_distances = @view(distances[:, 1:batch_size_actual])
            batch_assignments = @view(assignments[batch_start:batch_end])
            batch_one_hot = @view(one_hot[:, 1:batch_size_actual])

            # Kernel 1: Compute distances
            compute_distances_kernel!(batch_distances, batch_data, centroids)

            # Kernel 2: Assign clusters
            assign_clusters_kernel!(batch_assignments, batch_distances)

            # Kernel 3: One-hot encoding
            batch_one_hot .= 0.0f0
            onehot_encode!(batch_one_hot, batch_assignments, k)

            # Kernel 4: Update centroids
            update_centroids_kernel!(new_centroids, batch_data, batch_one_hot)

            # Update centroids on GPU
            counts += sum(batch_one_hot, dims = 2)
        end

        # Update centroids
        counts_safe = max.(counts', 1)
        new_centroids ./= counts_safe

        # Check for convergence
        delta = maximum(abs.(centroids - new_centroids))
        @info "Iteration $(iter)/$max_iters, max delta: $(delta)"
        if delta < tol
            @info "Terminating as max delta $delta < $(tol)"
            break
        end

        # Update centroids on GPU
        centroids .= new_centroids
    end

    Flux.cpu(assignments)
end

function _normalize_array!(
        X::AbstractArray{T}; dims::Int = 1) where {T <: AbstractFloat}
    norms = sqrt.(sum(abs2, X, dims = dims))
    epsilon = eps(T)
    X ./= (norms .+ epsilon)
end

function _topk(data::Matrix{T}, k::Int; dims::Int = 1) where {T <: Number}
    # TODO: only works on CPU; make it work on GPUs?
    # partialsortperm is not available in CUDA.jl
    dims in [1, 2] || throw(DomainError("dims must be 1 or 2!"))
    mapslices(v -> partialsortperm(v, 1:k, rev = true), data, dims = dims)
end

function _head(v::Vector)
    length(v) > 0 ? collect(take(v, length(v) - 1)) : similar(v, 0)
end
