@testset "config.jl" begin
    index_path = "./test_index"
    config = ColBERTConfig(index_path = index_path)
    key_vals = Dict([field => getproperty(
                         config, field)
                     for field in fieldnames(ColBERTConfig)])

    ColBERT.save(config)
    @test isfile(joinpath(
        index_path, "config.json"))

    config = ColBERT.load_config(index_path)
    @test config isa ColBERTConfig
    for field in fieldnames(ColBERTConfig)
        @test isequal(
            getproperty(config, field), key_vals[field])
    end
end
