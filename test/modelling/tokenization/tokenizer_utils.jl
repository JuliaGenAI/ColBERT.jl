using ColBERT: _add_marker_row

@testset "_add_marker_row" begin
    for type in [INT_TYPES; FLOAT_TYPES]
        # Test 1: Generic
        num_rows, num_cols = rand(1:20), rand(1:20)
        x = rand(type, num_rows, num_cols)
        x = _add_marker_row(x, zero(type))
        @test isequal(size(x), (num_rows + 1, num_cols))
        @test isequal(x[2, :], repeat([zero(type)], num_cols))

        # Test 2: Edge case, empty array
        num_cols = rand(1:20)
        x = rand(type, 0, num_cols)
        x = _add_marker_row(x, zero(type))
        @test isequal(size(x), (1, num_cols))
        @test isequal(x[1, :], repeat([zero(type)], num_cols))
    end
end
