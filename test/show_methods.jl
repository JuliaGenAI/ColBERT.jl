@testset "show methods" begin
    mktempdir() do dir
        config = ColBERTConfig(
            checkpoint = "dummy-checkpoint",
            index_path = dir,
            collection = ["doc1", "doc2"]
        )

        str = sprint(show, MIME("text/plain"), config)
        @test occursin("    checkpoint: dummy-checkpoint", str)
        @test occursin("    path: $dir", str)
        @test occursin("    collection: 2 documents", str)
    end
end
