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

