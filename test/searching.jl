using ColBERT: _build_emb2pid

@testset "_build_emb2pid" begin
    # Test 1: A single document
    doclens = rand(1:1000, 1)
    emb2pid = _build_emb2pid(doclens)
    @test emb2pid == ones(Int, doclens[1])

    # Test 2: Small test with a custom output 
    doclens = [3, 2, 4]
    emb2pid = _build_emb2pid(doclens)
    @test emb2pid == [1, 1, 1, 2, 2, 3, 3, 3, 3]

    # Test 3: With some zero document lengths
    doclens = [0, 2, 0, 3]
    emb2pid = _build_emb2pid(doclens)
    @test emb2pid == [2, 2, 4, 4, 4]

    # Test 3: Large, random inputs with equal doclengths
    doclen = rand(1:1000)
    doclens = doclen * ones(Int, rand(1:500))
    emb2pid = _build_emb2pid(doclens)
    @test emb2pid == repeat(1:length(doclens), inner = doclen)

    # Test 4: with no documents
    doclens = Int[]
    emb2pid = _build_emb2pid(doclens)
    @test emb2pid == Int[]

    # Test 5: Range of values, shapes and type 
    doclens = rand(0:100, rand(1:500))
    non_zero_docs = findall(>(0), doclens)
    zero_docs = findall(==(0), doclens)
    emb2pid = _build_emb2pid(doclens)
    @test all(in(non_zero_docs), emb2pid)
    @test issorted(emb2pid)
    for pid in non_zero_docs
        @test count(==(pid), emb2pid) == doclens[pid]
    end
    @test length(emb2pid) == sum(doclens[non_zero_docs])
    @test emb2pid isa Vector{Int}
end
