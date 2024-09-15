using ColBERT: _cids_to_eids!, retrieve, _collect_compressed_embs_for_pids,
               maxsim

@testset "_cids_to_eids!" begin
    # Test 1: Correct conversion
    eids = Vector{Int}(undef, 5)
    centroid_ids = [2, 1]
    ivf = [1, 2, 3, 4, 5, 6]
    ivf_lengths = [3, 2, 1]
    _cids_to_eids!(eids, centroid_ids, ivf, ivf_lengths)
    @test eids == [4, 5, 1, 2, 3]

    # Test 2: Random partitioning over a large vector
    # centroid_ids don't have to be sorted
    num_embeddings = rand(1:1000)
    num_partitions = rand(1:20)
    ivf = Int[]
    ivf_lengths = zeros(Int, num_partitions)
    assignments = rand(1:num_partitions, num_embeddings)        # eid to cid
    ivf_mapping = [Int[] for centroid_id in 1:num_partitions]
    for (eid, assignment) in enumerate(assignments)
        push!(ivf_mapping[assignment], eid)
    end
    for centroid_id in 1:num_partitions
        shuffle!(ivf_mapping[centroid_id])
        append!(ivf, ivf_mapping[centroid_id])
        ivf_lengths[centroid_id] = length(ivf_mapping[centroid_id])
    end
    centroid_ids = randperm(num_partitions)[1:rand(1:num_partitions)]
    eids = Vector{Int}(undef, sum(ivf_lengths[centroid_ids]))
    _cids_to_eids!(eids, centroid_ids, ivf, ivf_lengths)
    for centroid_id in centroid_ids
        @test collect(take(eids, ivf_lengths[centroid_id])) ==
              ivf_mapping[centroid_id]
        eids = eids[(ivf_lengths[centroid_id] + 1):end]
    end

    # Test 3: Empty eids
    eids = Vector{Int}(undef, 0)
    centroid_ids = Int[]
    ivf = Int[]
    ivf_lengths = Int[]
    _cids_to_eids!(eids, centroid_ids, ivf, ivf_lengths)
    @test eids == []

    # Test 4: Empty centroid_ids
    eids = Vector{Int}(undef, 0)
    centroid_ids = Int[]
    ivf = [1, 2, 3, 4, 5, 6]
    ivf_lengths = [3, 2, 1]
    _cids_to_eids!(eids, centroid_ids, ivf, ivf_lengths)
    @test all(iszero, eids)

    # Test 5: eids DimensionMismatch
    eids = Vector{Int}(undef, 5)
    centroid_ids = [1, 2, 3]
    ivf = [1, 2, 3, 4, 5, 6]
    ivf_lengths = [2, 2, 2]
    @test_throws DimensionMismatch _cids_to_eids!(
        eids, centroid_ids, ivf, ivf_lengths)

    # Test 6: ivf and ivf_lengths DimensionMismatch
    eids = Vector{Int}(undef, 6)
    centroid_ids = [1, 2, 3]
    ivf = [1, 2, 3, 4, 5]
    ivf_lengths = [2, 2, 2]
    @test_throws DimensionMismatch _cids_to_eids!(
        eids, centroid_ids, ivf, ivf_lengths)
end

@testset "retrieve" begin
    # Test 1: A small and basic case
    # The first and last centroids are closest to the query
    ivf = [3, 1, 4, 5, 6, 2]
    ivf_lengths = [2, 3, 1]
    centroids = Float32[1.0 0.0 0.0; 0.0 0.0 1.0]
    emb2pid = [10, 20, 30, 40, 50, 60]
    nprobe = 2
    Q = Float32[0.5 0.5]'
    expected_pids = [10, 20, 30]
    pids = retrieve(ivf, ivf_lengths, centroids, emb2pid, nprobe, Q)
    @test pids == expected_pids
end

@testset "_collect_compressed_embs_for_pids" begin
    # Test 1: Small example
    doclens = [3, 2, 4]
    codes = UInt32[1, 2, 3, 4, 5, 6, 7, 8, 9]
    residuals = [0x11 0x12 0x13 0x14 0x15 0x16 0x17 0x18 0x19;
                 0x21 0x22 0x23 0x24 0x25 0x26 0x27 0x28 0x29]
    pids = [1, 3]
    expected_codes_packed = UInt32[1, 2, 3, 6, 7, 8, 9]
    expected_residuals_packed = UInt8[0x11 0x12 0x13 0x16 0x17 0x18 0x19;
                                      0x21 0x22 0x23 0x26 0x27 0x28 0x29]
    codes_packed, residuals_packed = _collect_compressed_embs_for_pids(
        doclens, codes, residuals, pids)
    @test codes_packed == expected_codes_packed
    @test residuals_packed == expected_residuals_packed

    # Test 2: Edge case - Empty pids
    pids = Int[]
    expected_codes_packed = UInt32[]
    expected_residuals_packed = zeros(UInt8, 2, 0)
    codes_packed, residuals_packed = _collect_compressed_embs_for_pids(
        doclens, codes, residuals, pids)
    @test codes_packed == expected_codes_packed
    @test residuals_packed == expected_residuals_packed

    # Test 3: Edge case - doclens with zero values
    doclens = [3, 0, 4]
    codes = UInt32[1, 2, 3, 6, 7, 8, 9]
    residuals = [0x11 0x12 0x13 0x16 0x17 0x18 0x19;
                 0x21 0x22 0x23 0x26 0x27 0x28 0x29]
    pids = [1, 3]
    expected_codes_packed = UInt32[1, 2, 3, 6, 7, 8, 9]
    expected_residuals_packed = UInt8[0x11 0x12 0x13 0x16 0x17 0x18 0x19;
                                      0x21 0x22 0x23 0x26 0x27 0x28 0x29]
    codes_packed, residuals_packed = _collect_compressed_embs_for_pids(
        doclens, codes, residuals, pids)
    @test codes_packed == expected_codes_packed
    @test residuals_packed == expected_residuals_packed

    # Test 4: Shapes and types
    num_pids = rand(1:1000)
    doclens = rand(1:100, num_pids)
    codes = rand(UInt32, sum(doclens))
    residuals = rand(UInt8, 16, sum(doclens))
    pids = rand(1:num_pids, rand(1:num_pids))
    codes_packed, residuals_packed = _collect_compressed_embs_for_pids(
        doclens, codes, residuals, pids)
    @test length(codes_packed) == sum(doclens[pids])
    @test size(residuals_packed) == (16, sum(doclens[pids]))
    @test codes_packed isa Vector{UInt32}
    @test residuals_packed isa Matrix{UInt8}
end

@testset "maxsim" begin
    # Test 1: Basic test
    Q = Float32[1.0 0.5; 0.5 1.0]                  # Query matrix (2 query vectors)
    D = Float32[0.8 0.3 0.1; 0.2 0.7 0.4]          # Document matrix (3 document vectors)
    pids = [1, 2]                           # Document pids to match
    doclens = [1, 2]                        # Number of document vectors per pid
    expected_scores = Float32[1.5, 1.5]
    scores = maxsim(Q, D, pids, doclens)
    @test scores == expected_scores

    # Test 2: Dimension mismatch case
    Q = Float32[1.0 0.5; 0.5 1.0]                  # Query matrix
    D = Float32[0.8 0.3]                           # Document matrix with incorrect dimension
    pids = [1, 2]                           # pids to match
    doclens = [1, 2]                        # Document lengths
    @test_throws DimensionMismatch maxsim(Q, D, pids, doclens)

    # Test 3: Shapes and types
    doclens = rand(1:10, 1000)
    Q = rand(Float32, 128, 100)             # 100 query vectors, each of size 128
    D = rand(Float32, 128, sum(doclens))    # document vectors, each of size 128
    pids = collect(1:1000)
    scores = maxsim(Q, D, pids, doclens)
    @test length(scores) == length(pids)  
    @test scores isa Vector{Float32} 
end
