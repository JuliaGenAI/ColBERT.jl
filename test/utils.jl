using ColBERT: compute_distances_kernel!, update_centroids_kernel!,
               assign_clusters_kernel!, onehot_encode!, kmeans_gpu_onehot!,
               _normalize_array!, _topk, _head

@testset "compute_distances_kernel!" begin
    # Test 1: when all entries are the same
    dim = rand(1:20)
    batch_data = ones(Float32, dim, rand(1:20))
    centroids = ones(Float32, dim, rand(1:20))
    batch_distances = Matrix{Float32}(
        undef, size(centroids, 2), size(batch_data, 2))
    compute_distances_kernel!(batch_distances, batch_data, centroids)
    @test all(iszero, batch_distances)

    # Test 2: Edge case, single point and centroid
    batch_data = reshape(Float32[1.0; 2.0], 2, 1)
    centroids = reshape(Float32[2.0; 3.0], 2, 1)
    batch_distances = Matrix{Float32}(undef, 1, 1)
    compute_distances_kernel!(batch_distances, batch_data, centroids)
    @test batch_distances ≈ Float32[2]

    # Test 3: Special case
    dim = rand(1:20)
    bsize = rand(1:20)
    batch_data = ones(Float32, dim, bsize)
    centroids = ones(Float32, dim, bsize)
    for idx in 1:bsize
        batch_data[:, idx] .*= idx
        centroids[:, idx] .*= idx
    end
    expected_distances = ones(Float32, bsize, bsize)
    for (i, j) in product(1:bsize, 1:bsize)
        expected_distances[i, j] = dim * (i - j)^2
    end
    batch_distances = Matrix{Float32}(undef, bsize, bsize)
    compute_distances_kernel!(batch_distances, batch_data, centroids)
    @test isequal(expected_distances, batch_distances)

    # Test 4: Correct errors are thrown
    batch_data = Float32[1.0 2.0; 3.0 4.0]  # 2x2 matrix
    centroids = Float32[1.0 0.0; 0.0 1.0]   # 2x2 matrix
    batch_distances = zeros(Float32, 3, 2)  # Incorrect size: should be 2x2
    @test_throws DimensionMismatch compute_distances_kernel!(
        batch_distances, batch_data, centroids)

    batch_data = Float32[1.0 2.0; 3.0 4.0]                      # 2x2 matrix
    centroids = Float32[1.0 0.0 1.0; 0.0 1.0 0.0; 1.0 1.0 1.0]  # 3x3 matrix, different row count
    batch_distances = zeros(Float32, 3, 2)                      # Should match 3x2, but embedding dim is wrong
    @test_throws DimensionMismatch compute_distances_kernel!(
        batch_distances, batch_data, centroids)
end

@testset "update_centroids_kernel!" begin
    # Test 1: Generic test to see if results are accumulated correctly
    dim = rand(1:20)
    num_centroids = rand(1:20)
    num_points = rand(1:20)
    point_to_centroid = rand(1:num_centroids, num_points)
    new_centroids = ones(Float32, dim, num_centroids)
    batch_data = ones(Float32, dim, num_points)
    batch_one_hot = zeros(Float32, num_centroids, num_points)
    for idx in 1:num_points
        batch_one_hot[point_to_centroid[idx], idx] = 1.0f0
    end
    expected = zeros(Float32, dim, num_centroids)
    for centroid in point_to_centroid
        expected[:, centroid] .+= 1.0f0
    end
    update_centroids_kernel!(new_centroids, batch_data, batch_one_hot)
    @test isequal(new_centroids, expected .+ 1.0f0)

    # Test 2: error, incorrect `new_centroids` size
    batch_data = Float32[1.0 2.0; 3.0 4.0]      # 2x2 matrix
    batch_one_hot = Float32[1.0 0.0; 0.0 1.0]   # 2x2 matrix (one-hot encoded)
    new_centroids = zeros(Float32, 3, 2)        # Incorrect size: should be 2x2
    @test_throws DimensionMismatch update_centroids_kernel!(
        new_centroids, batch_data, batch_one_hot)

    # Test 3: error, incorrect `batch_one_hot` size
    batch_data = Float32[1.0 2.0; 3.0 4.0]              # 2x2 matrix
    batch_one_hot = Float32[1.0 0.0 0.0; 0.0 1.0 0.0]   # Incorrect size: should be 2x2, not 2x3
    new_centroids = zeros(Float32, 2, 2)                # Correct size, but the error should be triggered by batch_one_hot
    @test_throws DimensionMismatch update_centroids_kernel!(
        new_centroids, batch_data, batch_one_hot)
end

@testset "assign_clusters_kernel!" begin
    # Test 1: testing the correct minimum assignment with random permutations
    num_points = rand(1:100)
    batch_assignments = Vector{Int32}(undef, num_points)
    batch_distances = Matrix{Float32}(undef, rand(1:100), num_points)
    expected_assignments = Vector{Int32}(undef, num_points)
    for (idx, col) in enumerate(eachcol(batch_distances))
        perm = randperm(size(batch_distances, 1))
        col .= Float32.(perm)
        expected_assignments[idx] = sortperm(perm)[1]
    end
    assign_clusters_kernel!(batch_assignments, batch_distances)
    @test isequal(expected_assignments, batch_assignments)

    # Test 2: check DimensionMismatch error
    batch_distances = Float32[1.0 2.0;
                              4.0 5.0]
    batch_assignments = Int32[0]
    @test_throws DimensionMismatch assign_clusters_kernel!(
        batch_assignments, batch_distances)
end

@testset "onehot_encode!" begin
    # Test 1: Basic functionality
    k = rand(1:100)
    batch_assignments = Int32.(collect(1:k))
    batch_one_hot = zeros(Float32, k, k)
    onehot_encode!(batch_one_hot, batch_assignments, k)
    @test isequal(batch_one_hot, I(k))

    # Test 2: Slightly convoluted example
    batch_assignments = Int32[4, 2, 3, 1]
    batch_one_hot = zeros(Float32, 4, 4)
    onehot_encode!(batch_one_hot, batch_assignments, 4)
    @test batch_one_hot == Float32[0 0 0 1;
                                   0 1 0 0;
                                   0 0 1 0;
                                   1 0 0 0]
    # Test 3: Edge case with k = 1
    batch_assignments = Int32[1, 1, 1]
    batch_one_hot = zeros(Float32, 1, 3)
    onehot_encode!(batch_one_hot, batch_assignments, 1)
    @test batch_one_hot == Float32[1 1 1]

    # Test 4: Dimension mismatch error
    batch_assignments = Int32[1, 2]
    batch_one_hot = zeros(Float32, 3, 3)
    @test_throws DimensionMismatch onehot_encode!(
        batch_one_hot, batch_assignments, 3)
end

@testset "kmeans_gpu_onehot!" begin
    # Test 1: When all points are centroids
    data = rand(Float32, rand(1:100), rand(1:100))
    centroids = similar(data) 
    point_bsize = rand(1:size(data, 2))
    cluster_ids = kmeans_gpu_onehot!(data, centroids, size(data, 2))
    @test isequal(centroids[:, cluster_ids], data)
end

@testset "_normalize_array!" begin
    # column normalization
    X = rand(Float32, rand(1:100), rand(1:100))
    _normalize_array!(X, dims = 1)
    for col in eachcol(X)
        @test isapprox(norm(col), 1)
    end

    # row normalization
    X = rand(Float32, rand(1:100), rand(1:100))
    _normalize_array!(X, dims = 2)
    for row in eachrow(X)
        @test isapprox(norm(row), 1)
    end
end

@testset "_topk" begin
    # Test 1: Basic functionality with k = 2, along dimension 1 (columns)
    data = [3.0 1.0 4.0;
            1.0 5.0 9.0;
            2.0 6.0 5.0]
    k = 2
    result = _topk(data, k, dims = 1)
    @test result == [1 3 2;
                     3 2 3]

    # Test 2: Basic functionality with k = 2, along dimension 2 (rows)
    result = _topk(data, k, dims = 2)
    @test result == [3 1;
                     3 2;
                     2 3]

    # Test 3: Check DomainError for invalid dims value
    @test_throws DomainError _topk(data, k, dims = 3)
end

@testset "_head" begin
    # Test 1: Basic functionality with a non-empty vector
    v = [1, 2, 3, 4]
    result = _head(v)
    @test result == [1, 2, 3]

    # Test 2: Edge case with a single-element vector
    v = [10]
    result = _head(v)
    @test result == Int[]

    # Test 3: Edge case with an empty vector
    v = Int[]
    result = _head(v)
    @test result == Int[]

    # Test 4: Test with a vector of strings
    v = ["a", "b", "c"]
    result = _head(v)
    @test result == ["a", "b"]

    # Test 5: Test with a vector of floating-point numbers
    v = [1.5, 2.5, 3.5]
    result = _head(v)
    @test result == [1.5, 2.5]

    # Test 6: Test with a vector of characters
    v = ['a', 'b', 'c']
    result = _head(v)
    @test result == ['a', 'b']
end
