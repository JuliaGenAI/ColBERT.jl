using ColBERT: _normalize_array!, compress_into_codes!, _binarize, _unbinarize,
               _bucket_indices, _packbits, _unpackbits, binarize, compress,
               decompress_residuals, decompress

@testset "compress_into_codes!" begin
    # In most tests, we'll need unit vectors (so that dot-products become cosines)
    # Test 1: Edge case, 1 centroid and 1 embedding
    embs = rand(Float32, rand(1:128), 1)
    centroids = embs
    codes = zeros(UInt32, 1)
    bsize = rand(1:(size(embs, 2) + 5))
    compress_into_codes!(codes, centroids, embs; bsize = bsize)
    @test isequal(codes, UInt32[1])

    # Test 2: Edge case, equal # of centroids and embedings
    embs = rand(Float32, rand(1:128), rand(1:20))
    _normalize_array!(embs; dims = 1)
    perm = randperm(size(embs, 2))
    centroids = embs[:, perm]
    codes = zeros(UInt32, size(embs, 2))
    bsize = rand(1:(size(embs, 2) + 5))
    compress_into_codes!(codes, centroids, embs; bsize = bsize)
    @test isequal(codes, sortperm(perm))                    # sortperm(perm) -> inverse mapping

    # Test 3: sample centroids randomly from embeddings
    embs = rand(Float32, rand(2:128), rand(1:20))
    _normalize_array!(embs; dims = 1)
    perm = collect(take(randperm(size(embs, 2)), rand(1:size(embs, 2))))
    centroids = embs[:, perm]
    codes = zeros(UInt32, size(embs, 2))
    bsize = rand(1:(size(embs, 2) + 5))
    compress_into_codes!(codes, centroids, embs)
    @test all(in(1:size(centroids, 2)), codes)              # in the right range
    @test isequal(codes[perm], collect(1:length(perm)))     # centroids have the right mappings

    # Test 4: Build embs by extending centroids
    dim = rand(1:128)
    tol = 1.0e-5
    scale = rand(2:5)                                       # scaling factor
    centroids = rand(Float32, dim, rand(1:20))
    _normalize_array!(centroids; dims = 1)
    extension_mapping = rand(1:size(centroids, 2), scale * size(centroids, 2))
    embs = zeros(Float32, dim, length(extension_mapping))
    for (idx, col) in enumerate(eachcol(embs))
        # get some random noise
        noise = -tol + 2 * tol * rand()
        col .= centroids[:, extension_mapping[idx]] .+ noise
    end
    codes = zeros(UInt32, size(embs, 2))
    bsize = rand(1:(size(embs, 2) + 5))
    compress_into_codes!(codes, centroids, embs)
    @test isequal(codes, extension_mapping)

    # Test 5: Check that an error is thrown if the lengths don't match
    codes = zeros(UInt32, rand(1:(size(embs, 2) - 1)))
    @test_throws DimensionMismatch compress_into_codes!(codes, centroids, embs)
end

@testset "_binarize" begin
    # defining the datapacks
    datapacks = [
        (
            # Test 1: Basic functionality with a 2x2 matrix and 3 bits
            data = [0 1; 2 3],
            nbits = 3,
            expected_output = reshape(
                Bool[0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0], 3, 2, 2)
        ),
        (
            # Test 2: 2x2 matrix with 2 bits
            data = [0 1; 2 3],
            nbits = 2,
            expected_output = reshape(
                Bool[0, 0, 0, 1, 1, 0, 1, 1], 2, 2, 2)
        ),
        (
            # Test 3: Single value matrix
            data = reshape([7], 1, 1),
            nbits = 3,
            expected_output = reshape(Bool[1, 1, 1], 3, 1, 1)
        ),
        (
            # Test 4: Edge case with nbits = 1
            data = [0 1; 0 1],
            nbits = 1,
            expected_output = reshape(Bool[0, 0, 1, 1], 1, 2, 2)
        )
    ]

    for (data, nbits, expected_output) in datapacks
        try
            @test isequal(_binarize(data, nbits), expected_output)
        catch
            @show data, nbits, expected_output
        end
    end

    # Test 5: Invalid input with out-of-range values (should throw an error)
    data = [0 1; 4 2]  # 4 is out of range for 2 bits
    nbits = 2
    @test_throws DomainError _binarize(data, nbits)

    # Test 6: Testing correct shapes and types
    for int_type in INT_TYPES
        nbits = rand(1:5)
        dim = rand(1:500)
        batch_size = rand(1:20)
        data = map(Base.Fix1(convert, int_type),
            rand(0:((1 << nbits) - 1), dim, batch_size))
        output = _binarize(data, nbits)
        @test output isa Array{Bool, 3}
        @test isequal(size(output), (nbits, dim, batch_size))
    end
end

@testset "_unbinarize" begin
    # Test 1: All bits are 0, should return zeros
    nbits = rand(1:10)
    data = falses(nbits, rand(1:20), rand(1:20))
    @test isequal(_unbinarize(data), zeros(Int, size(data, 2), size(data, 3)))

    # Test 2: All bits set to 1
    nbits = rand(1:10)
    data = trues(nbits, rand(1:20), rand(1:20))
    @test isequal(_unbinarize(data),
        (1 << nbits - 1) * ones(Int, size(data)[2], size(data)[3]))

    # Test 3: Edge case, single element
    data = reshape(Bool[1, 0, 0, 1, 1], 5, 1, 1)
    @test isequal(_unbinarize(data), reshape([25], 1, 1))

    # Test 4: Multiple bits forming non-zero integers
    # inputs matrix: [55 20; 49 24]
    data = reshape(
        Bool[1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        6,
        2,
        2)
    @test isequal(_unbinarize(data), [55 20; 49 24])

    # Test 4: Edge case with empty array
    data = reshape(Bool[], 0, 0, 0)
    @test isequal(_unbinarize(data), Matrix{Int}(undef, 0, 0))

    # Test 5: Checking shapes and types
    nbits = rand(1:20)
    dim = rand(1:20)
    batch_size = rand(1:20)
    data = rand(Bool, nbits, dim, batch_size)
    @test isequal(size(_unbinarize(data)), (dim, batch_size))
end

@testset "_unbinarize inverts _binarize" begin
    # Take any random integer matrix, apply the ops and test equality with result
    nbits = rand(1:20)
    data = rand(0:((1 << nbits) - 1), rand(1:20), rand(1:20))
    binarized_data = _binarize(data, nbits)
    unbinarzed_data = _unbinarize(binarized_data)
    @test isequal(data, unbinarzed_data)
end

@testset "_bucket_indices" begin
    # defining datapacks
    datapacks = [
        (
            # Test 1: Test with a matrix
            data = [1 6; 3 12],
            bucket_cutoffs = [0, 5, 10, 15],
            expected = [1 2; 1 3]
        ),
        (

            # Test 2: Edge case with empty data
            data = Matrix{Float32}(undef, 0, 0),
            bucket_cutoffs = [0, 10, 20],
            expected = Matrix{Int}(undef, 0, 0)
        ),
        (
            # Test 3: Edge case with empty bucket_cutoffs
            data = [5 15],
            bucket_cutoffs = Float32[],
            expected = [0 0]
        ),
        (
            # Test 4: with floats
            data = [1.1 2.5 7.8],
            bucket_cutoffs = [0.0, 2.0, 5.0, 10.0],
            expected = [1 2 3]
        )
    ]

    for (data, bucket_cutoffs, expected) in datapacks
        try
            @test isequal(_bucket_indices(data, bucket_cutoffs), expected)
        catch
            @show data, bucket_cutoffs, expected
        end
    end

    # Test 5: Check that range of indices is correct
    data = rand(Float32, rand(1:20), rand(1:20))
    bucket_cutoffs = sort(rand(rand(1:100)))
    @test all(
        in(0:(length(bucket_cutoffs))), _bucket_indices(data, bucket_cutoffs))

    # Test 6: Checking shapes, dimensions and types
    for (T, S) in collect(product(
        [INT_TYPES; FLOAT_TYPES], [INT_TYPES; FLOAT_TYPES]))
        data = rand(T, rand(1:20), rand(1:20))
        bucket_cutoffs = sort(rand(rand(1:100)))
        @test isequal(size(_bucket_indices(data, bucket_cutoffs)), size(data))
    end
end

@testset "_packbits" begin
    # In all tests, remember: BitArray.chunks reverses the endianess
    # Test 1: Basic case with 1x64x1 array (one 64-bit block)
    bitsarray = reshape(
        Bool[1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
            0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
            1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
        (1, 64, 1))
    expected = reshape(
        UInt8[0b11011011; 0b01101010; 0b00111011; 0b11011010; 0b11100000;
              0b01011001; 0b11111011; 0b01010001],
        8,
        1)
    @test isequal(_packbits(bitsarray), expected)

    # Test 2: All bits are 0s
    bitsarray = falses(rand(1:20), 8 * rand(1:20), rand(1:20))
    expected = zeros(
        UInt8, div(div(prod(size(bitsarray)), size(bitsarray, 3)), 8),
        size(bitsarray, 3))
    @test isequal(_packbits(bitsarray), expected)

    # Test 3: All bits are 1s
    bitsarray = trues(rand(1:20), 8 * rand(1:20), rand(1:20))
    expected = 0xff * ones(
        UInt8, div(div(prod(size(bitsarray)), size(bitsarray, 3)), 8),
        size(bitsarray, 3))
    @test isequal(_packbits(bitsarray), expected)

    # Test 4: Alternating bits; each byte is 0b10101010, or 0xaa
    # Again, BitArray.chunks reverses endianess; so the byte is really 0b01010101
    # or 0x55
    bitsarray = trues(rand(1:20), 8 * rand(1:20), rand(1:20))
    bitsarray[collect(2:2:prod(size(bitsarray)))] .= 0
    expected = 0x55 * ones(
        UInt8, div(div(prod(size(bitsarray)), size(bitsarray, 3)), 8),
        size(bitsarray, 3))
    @test isequal(_packbits(bitsarray), expected)

    # Test 4: Edge case with an empty array (should be empty)
    bitsarray = reshape(Bool[], (0, 0, 0))
    expected = Matrix{UInt8}(undef, 0, 0)
    @test isequal(_packbits(bitsarray), expected)

    # Test 5: Ensure that proper errors are thrown
    @test_throws DomainError _packbits(trues(3, 7, 5))                      # dim not a multiple of 64

    # Test 6: Test shapes and types
    dim = 8 * rand(1:20)
    nbits = rand(1:20)
    batch_size = rand(1:20)
    bitsarray = rand(Bool, nbits, dim, batch_size)
    output = _packbits(bitsarray)
    @test output isa Matrix{UInt8}
    @test isequal(size(output), (div(dim * nbits, 8), batch_size))
end

@testset "_unpackbits" begin
    # Again, remember: BitArray.chunks reverses the endianess
    # Test 1: Basic case with 1x8 matrix, with nbits = 1
    nbits = 1
    packed_bits = reshape(
        UInt8[0b00101010, 0b00010001, 0b11111111, 0b01000000,
            0b10000000, 0b11001000, 0b00100001, 0b01010111,
            0b00101010, 0b00010001, 0b11111111, 0b01000000,
            0b10000000, 0b11001000, 0b00100001, 0b01010111,
            0b00101010, 0b00010001, 0b11111111, 0b01000000,
            0b10000000, 0b11001000, 0b00100001, 0b01010111,
            0b00101010, 0b00010001, 0b11111111, 0b01000000,
            0b10000000, 0b11001000, 0b00100001, 0b01010111,
            0b00101010, 0b00010001, 0b11111111, 0b01000000,
            0b10000000, 0b11001000, 0b00100001, 0b01010111,
            0b00101010, 0b00010001, 0b11111111, 0b01000000,
            0b10000000, 0b11001000, 0b00100001, 0b01010111,
            0b00101010, 0b00010001, 0b11111111, 0b01000000,
            0b10000000, 0b11001000, 0b00100001, 0b01010111,
            0b00101010, 0b00010001, 0b11111111, 0b01000000,
            0b10000000, 0b11001000, 0b00100001, 0b01010111
        ],
        1, 64)

    expected = reshape(
        Bool[
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0
        ],
        nbits,
        8 * div(prod(size(packed_bits)),
            nbits * size(packed_bits, 2)), size(packed_bits, 2))
    unpacked_bits = _unpackbits(packed_bits, nbits)
    @test isequal(unpacked_bits, expected)

    # Test 2: All zeros
    nbits = rand(1:10)
    packed_bits = zeros(UInt8, nbits * rand(1:20), rand(1:20))
    expected = falses(
        nbits, 8 * div(prod(size(packed_bits)), nbits * size(packed_bits, 2)),
        size(packed_bits, 2))
    @test isequal(_unpackbits(packed_bits, nbits), expected)

    # Test 3: All ones
    nbits = rand(1:10)
    packed_bits = 0xff * ones(UInt8, nbits * rand(1:20), rand(1:20))
    expected = trues(
        nbits, 8 * div(prod(size(packed_bits)), nbits * size(packed_bits, 2)),
        size(packed_bits, 2))
    @test isequal(_unpackbits(packed_bits, nbits), expected)

    # Test 4: types and shapes
    nbits = rand(1:10)
    batch_size = rand(1:20)
    packed_bits = rand(UInt8, nbits * rand(1:20), batch_size)
    dim = 8 * div(prod(size(packed_bits)), nbits * batch_size)
    output = _unpackbits(packed_bits, nbits)
    @test output isa AbstractArray{Bool, 3}
    @test isequal(size(output), (nbits, dim, batch_size))
end

@testset "_unpackbits inverts _packbits" begin
    nbits = rand(1:10)
    bitsarray = rand(Bool, nbits, 8 * rand(1:20), rand(1:20))
    packed_bits = _packbits(bitsarray)
    unpacked_bits = _unpackbits(packed_bits, nbits)
    @test isequal(unpacked_bits, bitsarray)
end

@testset "binarize" begin
    # Test 1: Checking the types and dimensions
    dim = 8 * rand(1:20)
    nbits = rand(1:20)
    bucket_cutoffs = sort(rand(Float32, (1 << nbits) - 1))
    residuals = rand(Float32, dim, rand(1:100))
    binarized_residuals = binarize(dim, nbits, bucket_cutoffs, residuals)
    @test binarized_residuals isa Matrix{UInt8}
    @test isequal(size(residuals, 2), size(binarized_residuals, 2))
    @test isequal(size(binarized_residuals, 1), div(dim, 8) * nbits)

    # Test 2: Checking correct errors being thrown
    @test_throws DomainError binarize(
        7, 7, sort(rand(Float32, 1 << 7 - 1)), rand(Float32, 7, 10))    # dim not multiple of 8
    @test_throws DomainError binarize(
        8, 8, sort(rand(Float32, 2^8 - 2)), rand(Float32, 8, 10))       # incorrect length for bucket_cutoffs
end

@testset "compress" begin
    # Test 1: Edge case, 1 centroid and 1 embedding
    dim = 8 * rand(1:20)
    nbits = rand(1:20)
    bucket_cutoffs = sort(rand(Float32, (1 << nbits) - 1))
    embs = rand(Float32, dim, 1)
    centroids = embs
    bsize = rand(1:(size(embs, 2) + 5))
    codes, residuals = compress(
        centroids, bucket_cutoffs, dim, nbits, embs; bsize = bsize)
    @test isequal(codes, UInt32[1])
    @test all(isequal(zero(UInt8)), residuals)

    # Test 2: Edge case, equal # of centroids and embeddings
    dim = 8 * rand(1:20)
    nbits = rand(1:20)
    bucket_cutoffs = sort(rand(Float32, (1 << nbits) - 1))
    embs = rand(Float32, dim, rand(1:20))
    _normalize_array!(embs; dims = 1)
    perm = randperm(size(embs, 2))
    centroids = embs[:, perm]
    bsize = rand(1:(size(embs, 2) + 5))
    codes, residuals = compress(
        centroids, bucket_cutoffs, dim, nbits, embs; bsize = bsize)
    @test isequal(codes, sortperm(perm))                    # sortperm(perm) -> inverse mapping
    @test all(isequal(zero(UInt8)), residuals)

    # Test 3: sample centroids randomly from embeddings
    dim = 8 * rand(1:20)
    nbits = rand(1:20)
    bucket_cutoffs = sort(rand(Float32, (1 << nbits) - 1))
    embs = rand(Float32, dim, rand(1:20))
    _normalize_array!(embs; dims = 1)
    perm = collect(take(randperm(size(embs, 2)), rand(1:size(embs, 2))))
    centroids = embs[:, perm]
    bsize = rand(1:(size(embs, 2) + 5))
    codes, residuals = compress(
        centroids, bucket_cutoffs, dim, nbits, embs; bsize = bsize)
    @test all(in(1:size(centroids, 2)), codes)              # in the right range
    @test isequal(codes[perm], collect(1:length(perm)))     # centroids have the right mappings
    @test all(isequal(zero(UInt8)), residuals[:, perm])        # centroids have zero residuals

    # Test 4: Build embs by extending centroids
    tol = 1.0e-5
    scale = rand(2:5)                                       # scaling factor
    dim = 8 * rand(1:20)
    nbits = rand(1:20)
    bucket_cutoffs = sort(rand(Float32, (1 << nbits) - 1))
    centroids = rand(Float32, dim, rand(1:20))
    _normalize_array!(centroids; dims = 1)
    extension_mapping = rand(1:size(centroids, 2), scale * size(centroids, 2))
    embs = zeros(Float32, dim, length(extension_mapping))
    for (idx, col) in enumerate(eachcol(embs))
        # get some random noise
        noise = -tol + 2 * tol * rand()
        col .= centroids[:, extension_mapping[idx]] .+ noise
    end
    bsize = rand(1:(size(embs, 2) + 5))
    codes, residuals = compress(
        centroids, bucket_cutoffs, dim, nbits, embs; bsize = bsize)
    @test isequal(codes, extension_mapping)
    # TODO somehow test that all the absmax of all residuals is atmost tol

    # Test 5: Shapes and types
    dim = 8 * rand(1:20)
    nbits = rand(1:20)
    bucket_cutoffs = sort(rand(Float32, (1 << nbits) - 1))
    embs = rand(Float32, dim, rand(1:20))
    _normalize_array!(embs; dims = 1)
    perm = collect(take(randperm(size(embs, 2)), rand(1:size(embs, 2))))
    centroids = embs[:, perm]
    bsize = rand(1:(size(embs, 2) + 5))
    codes, residuals = compress(
        centroids, bucket_cutoffs, dim, nbits, embs; bsize = bsize)
    @test codes isa Vector{UInt32}
    @test residuals isa Matrix{UInt8}
    @test isequal(length(codes), size(embs, 2))
    @test isequal(size(residuals, 2), size(embs, 2))
    @test isequal(size(residuals, 1), div(dim, 8) * nbits)
end

@testset "decompress_residuals" begin
    # Test 1: Checking types and dimensions, and correct range of values
    dim = 8 * rand(1:20)
    nbits = rand(1:20)
    bucket_weights = sort(rand(Float32, 1 << nbits))
    binarized_residuals = rand(UInt8, div(dim, 8) * nbits, rand(1:20))
    residuals = decompress_residuals(
        dim, nbits, bucket_weights, binarized_residuals)
    @test residuals isa Matrix{Float32}
    @test isequal(size(residuals, 2), size(binarized_residuals, 2))
    @test isequal(size(residuals, 1), dim)
    @test all(in(bucket_weights), residuals)

    # Test 2: Checking correct errors being thrown
    @test_throws DomainError decompress_residuals(
        7, 7, sort(rand(Float32, 1 << 7)),
        rand(UInt8, div(7, 8) * 7, rand(1:20)))                                 # dim not a multiple of 8
    @test_throws DomainError decompress_residuals(
        8, 8, sort(rand(Float32, (1 << 8) - 1)),
        rand(UInt8, div(8, 8) * 8, rand(1:20)))                                 # bucket_weights not having correct length
    @test_throws DomainError decompress_residuals(
        8, 8, sort(rand(Float32, 1 << 8)), rand(UInt8, 7, 64 * rand(1:100)))    # binarized_residuals having an incorrect dim
end

@testset "decompress_residuals inverts binarize" begin
    # not exactly inverse, but a close inverse
    dim = 8 * rand(1:20)
    nbits = rand(1:20)
    bucket_cutoffs = sort(rand(Float32, (1 << nbits) - 1))
    bucket_weights = sort(rand(Float32, 1 << nbits))
    residuals = rand(Float32, dim, rand(1:100))

    # map each residual to it's expected weight
    expected_indices = map(
        Base.Fix1(searchsortedfirst, bucket_cutoffs), residuals)
    expected = bucket_weights[expected_indices]
    binarized_residuals = binarize(dim, nbits, bucket_cutoffs, residuals)
    decompressed_residuals = decompress_residuals(
        dim, nbits, bucket_weights, binarized_residuals)
    @test isequal(expected, decompressed_residuals)
end

@testset "decompress" begin
    # Test 1: Types and shapes, and right range of values
    dim = 8 * rand(1:20)
    nbits = rand(1:20)
    batch_size = rand(1:100)
    bucket_weights = sort(rand(Float32, 1 << nbits))
    centroids = rand(Float32, dim, rand(1:100))
    codes = UInt32.(rand(1:size(centroids, 2), batch_size))
    binarized_residuals = rand(UInt8, div(dim, 8) * nbits, batch_size)
    bsize = rand(1:(batch_size + 5))
    embeddings = decompress(dim, nbits, centroids, bucket_weights,
        codes, binarized_residuals; bsize = bsize)
    @test embeddings isa Matrix{Float32}
    @test isequal(size(embeddings), (dim, length(codes)))
end
