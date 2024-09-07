using LinearAlgebra: __normalize!
using ColBERT: _sample_pids, _heldout_split, setup, _bucket_cutoffs_and_weights,
               _normalize_array!, _compute_avg_residuals!

@testset "_sample_pids tests" begin
    # Test 1: More pids than given can't be sampled
    num_documents = rand(0:100000)
    pids = _sample_pids(num_documents)
    @test length(pids) <= num_documents

    # Test 2: Edge case, when
    num_documents = rand(0:1)
    pids = _sample_pids(num_documents)
    @test length(pids) <= num_documents
end

@testset "_heldout_split" begin
    # Test 1: A basic test with a large size
    sample = rand(Float32, rand(1:20), 100000)
    for heldout_fraction in Float32.(collect(0.1:0.1:1.0))
        sample_train, sample_heldout = _heldout_split(
            sample; heldout_fraction = heldout_fraction)
        heldout_size = min(50000, Int(floor(100000 * heldout_fraction)))
        @test size(sample_train, 2) == 100000 - heldout_size
        @test size(sample_heldout, 2) == heldout_size
    end

    # Test 2: Edge case with 1 column, should return empty train and full heldout
    sample = rand(Float32, 3, 1)
    heldout_fraction = 0.5f0
    sample_train, sample_heldout = _heldout_split(
        sample; heldout_fraction = heldout_fraction)
    @test size(sample_train, 2) == 0            # No columns should be left in the train set
    @test size(sample_heldout, 2) == 1          # All columns in the heldout set
end

@testset "setup" begin
    # Test 1: Number of documents and chunksize should not be altered
    collection = string.(rand('a':'z', rand(1:1000)))
    avg_doclen_est = Float32(100 * rand())
    nranks = rand(1:10)
    num_clustering_embs = rand(1:1000)
    chunksize = rand(1:20)
    plan_dict = setup(
        collection, avg_doclen_est, num_clustering_embs, chunksize, nranks)
    @test plan_dict["avg_doclen_est"] == avg_doclen_est
    @test plan_dict["chunksize"] == chunksize
    @test plan_dict["num_documents"] == length(collection)
    @test plan_dict["num_embeddings_est"] == avg_doclen_est * length(collection)

    # Test 2: Tests for number of chunks
    avg_doclen_est = 1.0f0
    nranks = rand(1:10)
    num_clustering_embs = rand(1:1000)

    ## without remainders
    chunksize = rand(1:20)
    collection = string.(rand('a':'z', chunksize * rand(1:100)))
    plan_dict = setup(
        collection, avg_doclen_est, num_clustering_embs, chunksize, nranks)
    @test plan_dict["num_chunks"] == div(length(collection), chunksize)

    ## with remainders
    chunksize = rand(1:20)
    collection = string.(rand(
        'a':'z', chunksize * rand(1:100) + rand(1:(chunksize - 1))))
    plan_dict = setup(
        collection, avg_doclen_est, num_clustering_embs, chunksize, nranks)
    @test plan_dict["num_chunks"] == div(length(collection), chunksize) + 1

    # Test 3: Tests for number of clusters
    collection = string.(rand('a':'z', rand(1:1000)))
    avg_doclen_est = Float32(100 * rand())
    nranks = rand(1:10)
    num_clustering_embs = rand(1:10000)
    chunksize = rand(1:20)
    plan_dict = setup(
        collection, avg_doclen_est, num_clustering_embs, chunksize, nranks)
    @test plan_dict["num_partitions"] <= num_clustering_embs
    @test plan_dict["num_partitions"] <=
          16 * sqrt(avg_doclen_est * length(collection))
end

@testset "_bucket_cutoffs_and_weights" begin
    # Test 1: Basic test with 2x2 matrix and nbits=2
    heldout_avg_residual = [0.0f0 0.2f0; 0.4f0 0.6f0; 0.8f0 1.0f0]
    nbits = 2
    cutoffs, weights = _bucket_cutoffs_and_weights(nbits, heldout_avg_residual)
    expected_cutoffs = Float32[0.25, 0.5, 0.75]
    expected_weights = Float32[0.125, 0.375, 0.625, 0.875]
    @test cutoffs ≈ expected_cutoffs
    @test weights ≈ expected_weights

    # Test 2: Uniform values
    value = rand(Float32)
    heldout_avg_residual = value * ones(Float32, rand(1:20), rand(1:20))
    nbits = rand(1:10)
    cutoffs, weights = _bucket_cutoffs_and_weights(nbits, heldout_avg_residual)
    @test all(isequal(value), cutoffs)
    @test all(isequal(value), weights)

    # Test 3: Shapes and types
    heldout_avg_residual = rand(Float32, rand(1:20), rand(1:20))
    nbits = rand(1:10)
    cutoffs, weights = _bucket_cutoffs_and_weights(nbits, heldout_avg_residual)
    @test length(cutoffs) == (1 << nbits) - 1
    @test length(weights) == 1 << nbits
    @test cutoffs isa Vector{Float32}
    @test weights isa Vector{Float32}
end

@testset "_compute_avg_residuals!" begin
    # Test 1: centroids and heldout_avg_residual have the same columns with different perms
    nbits = rand(1:20)
    centroids = rand(Float32, rand(1:20), rand(1:20))
    _normalize_array!(centroids; dims = 1)
    perm = randperm(size(centroids, 2))[1:rand(1:size(centroids, 2))]
    heldout = centroids[:, perm]
    codes = Vector{UInt32}(undef, size(heldout, 2))
    bucket_cutoffs, bucket_weights, avg_residual = _compute_avg_residuals!(
        nbits, centroids, heldout, codes)
    @test all(iszero, bucket_cutoffs)
    @test all(iszero, bucket_weights)
    @test iszero(avg_residual)

    # Test 2: some tolerance level
    tol = 1e-5
    nbits = rand(1:20)
    centroids = rand(Float32, rand(1:20), rand(1:20))
    _normalize_array!(centroids; dims = 1)
    perm = randperm(size(centroids, 2))[1:rand(1:size(centroids, 2))]
    heldout = centroids[:, perm]
    for col in eachcol(heldout)
        col .+= -tol + 2 * tol * rand()
    end
    codes = Vector{UInt32}(undef, size(heldout, 2))
    bucket_cutoffs, bucket_weights, avg_residual = _compute_avg_residuals!(
        nbits, centroids, heldout, codes)
    @test all(<=(tol), bucket_cutoffs)
    @test all(<=(tol), bucket_weights)
    @test avg_residual <= tol

    # Test 3: Shapes and types
    nbits = rand(1:20)
    dim = rand(1:20)
    centroids = rand(Float32, dim, rand(1:20))
    heldout = rand(Float32, dim, rand(1:20))
    codes = Vector{UInt32}(undef, size(heldout, 2))
    bucket_cutoffs, bucket_weights, avg_residual = _compute_avg_residuals!(
        nbits, centroids, heldout, codes)
    @test length(bucket_cutoffs) == (1 << nbits) - 1
    @test length(bucket_weights) == 1 << nbits
    @test bucket_cutoffs isa Vector{Float32}
    @test bucket_weights isa Vector{Float32}
    @test avg_residual isa Float32

    # Test 4: Correct errors are thrown
    nbits = 2
    centroids = Float32[1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]  # (3, 3) matrix
    heldout = Float32[1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]    # (3, 3) matrix
    codes = UInt32[0, 0]                                        # Length is 2, but `heldout` has 3 columns
    # Check for DimensionMismatch error
    @test_throws DimensionMismatch _compute_avg_residuals!(
        nbits, centroids, heldout, codes)
end
