using ColBERT: mask_skiplist!, _clear_masked_embeddings!, _flatten_embeddings,
               _remove_masked_tokens

@testset "mask_skiplist!" begin
    # Test Case 1: Simple case with no skips
    mask = trues(3, 3)
    integer_ids = Int32[1 2 3; 4 5 6; 7 8 9]
    skiplist = Int[]
    expected_mask = trues(3, 3)
    mask_skiplist!(mask, integer_ids, skiplist)
    @test mask == expected_mask

    # Test Case 2: Skip one value
    mask = trues(3, 3)
    integer_ids = Int32[1 2 3; 4 5 6; 7 8 9]
    skiplist = [5]
    expected_mask = [true true true; true false true; true true true]
    mask_skiplist!(mask, integer_ids, skiplist)
    @test mask == expected_mask

    # Test Case 3: Skip multiple values
    mask = trues(3, 3)
    integer_ids = Int32[1 2 3; 4 5 6; 7 8 9]
    skiplist = [2, 6, 9]
    expected_mask = [true false true; true true false; true true false]
    mask_skiplist!(mask, integer_ids, skiplist)
    @test mask == expected_mask

    # Test Case 4: All values in skiplist
    mask = trues(3, 3)
    integer_ids = Int32[1 2 3; 4 5 6; 7 8 9]
    skiplist = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    expected_mask = falses(3, 3)
    mask_skiplist!(mask, integer_ids, skiplist)
    @test mask == expected_mask

    # Test Case 5: Empty integer_ids matrix
    mask = trues(0, 0)
    integer_ids = rand(Int32, 0, 0)
    skiplist = [1]
    expected_mask = trues(0, 0)
    mask_skiplist!(mask, integer_ids, skiplist)
    @test mask == expected_mask

    # Test Case 6: Skiplist with no matching values
    mask = trues(3, 3)
    integer_ids = Int32[1 2 3; 4 5 6; 7 8 9]
    skiplist = [10, 11]
    expected_mask = trues(3, 3)
    mask_skiplist!(mask, integer_ids, skiplist)
    @test mask == expected_mask
end

@testset "_clear_masked_embeddings!" begin
    # Test Case 1: No skiplist entries
    dim, len, bsize = rand(1:20, 3)
    D = rand(Float32, dim, len, bsize)
    integer_ids = rand(Int32, len, bsize)
    skiplist = Int[]
    expected_D = copy(D)
    _clear_masked_embeddings!(D, integer_ids, skiplist)
    @test D == expected_D

    # Test Case 2: Single skiplist entry
    dim, len, bsize = rand(1:20, 3)
    D = rand(Float32, dim, len, bsize)
    integer_ids = rand(Int32, len, bsize)
    skiplist = Int[integer_ids[rand(1:(len * bsize))]]
    expected_D = copy(D)
    expected_D[:, findall(in(skiplist), integer_ids)] .= 0.0f0
    _clear_masked_embeddings!(D, integer_ids, skiplist)
    @test D == expected_D

    # Test Case 3: Multiple skiplist entries
    dim, len, bsize = rand(1:20, 3)
    D = rand(Float32, dim, len, bsize)
    integer_ids = rand(Int32, len, bsize)
    skiplist = unique(Int.(rand(vec(integer_ids), rand(1:(len * bsize)))))
    expected_D = copy(D)
    expected_D[:, findall(in(skiplist), integer_ids)] .= 0.0f0
    _clear_masked_embeddings!(D, integer_ids, skiplist)
    @test D == expected_D

    # Test Case 4: Skip all tokens
    dim, len, bsize = rand(1:20, 3)
    D = rand(Float32, dim, len, bsize)
    integer_ids = rand(Int32, len, bsize)
    skiplist = unique(Int.(vec(integer_ids)))
    expected_D = similar(D)
    expected_D .= 0.0f0
    _clear_masked_embeddings!(D, integer_ids, skiplist)
    @test D == expected_D

    # Test Case 5: Skiplist with no matching tokens
    dim, len, bsize = rand(1:20, 3)
    D = rand(Float32, dim, len, bsize)
    integer_ids = Int32.(rand(1:100, len, bsize))
    skiplist = unique(rand(101:1000, rand(1:20)))
    expected_D = copy(D)
    _clear_masked_embeddings!(D, integer_ids, skiplist)
    @test D == expected_D

    # Test 6: Types and shapes
    dim, len, bsize = rand(1:20, 3)
    D = rand(Float32, dim, len, bsize)
    integer_ids = rand(Int32, len, bsize)
    skiplist = unique(rand(Int, rand(1:20)))
    mask = _clear_masked_embeddings!(D, integer_ids, skiplist)
    @test mask isa Array{Bool, 3}
    @test isequal(size(mask), (1, size(D)[2:end]...))
end

@testset "_flatten_embeddings" begin
    # Test Case 1: Generic case; len will correspond to a vector of constants
    dim, len, bsize = rand(1:20, 3)
    D = Array{Float32}(undef, dim, len, bsize)
    for idx in 1:len
        D[:, idx, :] .= idx
    end
    expected = Matrix{Float32}(undef, dim, len * bsize)
    for idx in 1:len
        expected[:, [idx + k * len for k in 0:(bsize - 1)]] .= idx
    end
    @test _flatten_embeddings(D) == expected

    # Test Case 2: Edge case with 0x3x2 array (should return 0x6 array)
    D = Float32[]
    D = reshape(D, 0, 3, 2)
    expected_output = reshape(Float32[], 0, 6)
    @test _flatten_embeddings(D) == expected_output
end

@testset "_remove_masked_tokens" begin
    # Test 1: Generic case; build a skiplist, and manually build the expected tensor
    dim, len, bsize = rand(1:20, 3)
    mask = trues(len, bsize)
    skiplist = unique(rand(1:len, rand(1:len)))
    for id in skiplist
        mask[id, :] .= false
    end
    D = Matrix{Float32}(undef, dim, len * bsize)
    for idx in 1:len
        D[:, [idx + k * len for k in 0:(bsize - 1)]] .= idx
    end
    expected = rand(Float32, dim, 0)
    for emb_id in 1:size(D, 2)
        if !(D[1, emb_id] in skiplist)
            expected = hcat(expected, D[:, emb_id])
        end
    end
    @test _remove_masked_tokens(D, mask) == expected

    # Test 2: Test for errors
    @test_throws DimensionMismatch _remove_masked_tokens(
        rand(Float32, 12, 20), rand(Bool, 4, 4))
end
