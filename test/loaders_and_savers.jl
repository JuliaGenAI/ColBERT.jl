using ColBERT: load_codec, save_codec, load_config, save, load_doclens,
               load_compressed_embs, load_chunk_metadata_property,
               save_chunk_metadata_property, load_codes, save_chunk

@testset "load_codec inverts save_codec" begin
    mktempdir() do index_path
        # create and save the dummy codec
        centroids = ones(Float32, 10, 10)
        avg_residual = one(Float32)
        bucket_cutoffs = ones(Float32, 10)
        bucket_weights = ones(Float32, 10)
        save_codec(
            index_path, centroids, bucket_cutoffs, bucket_weights, avg_residual)

        # Load the data and check for correctness
        loaded_data = load_codec(index_path)
        @test loaded_data["centroids"] == ones(Float32, 10, 10)
        @test loaded_data["avg_residual"] == one(Float32)
        @test loaded_data["bucket_cutoffs"] == ones(Float32, 10)
        @test loaded_data["bucket_weights"] == ones(Float32, 10)

        # Delete files one by one and check for errors
        file_names = ["centroids.jld2", "avg_residual.jld2",
            "bucket_cutoffs.jld2", "bucket_weights.jld2"]

        for file in file_names
            # Delete the file
            rm(joinpath(index_path, file))

            # Check that the correct error is thrown
            @test_throws ErrorException load_codec(index_path)

            # Re-create the file to restore the original state
            save_codec(
                index_path, centroids, bucket_cutoffs, bucket_weights, avg_residual)
        end
    end
end

@testset "load_config inverts save_config" begin
    mktempdir() do index_path
        config = ColBERTConfig(
            use_gpu = true,
            rank = 1,
            nranks = 4,
            query_token_id = "[unused5]",
            doc_token_id = "[unused6]",
            query_token = "[CustomQ]",
            doc_token = "[CustomD]",
            checkpoint = "dummy-checkpoint-path",
            collection = "dummy-collection-path",
            dim = 256,
            doc_maxlen = 512,
            mask_punctuation = false,
            query_maxlen = 64,
            attend_to_mask_tokens = true,
            index_path = index_path,
            index_bsize = 128,
            chunksize = 50000,
            passages_batch_size = 10000,
            nbits = 4,
            kmeans_niters = 40,
            nprobe = 4,
            ncandidates = 16384
        )

        # save the config and verify that it's saved
        save(config)
        @test isfile(joinpath(index_path, "config.json"))

        # load the saved config and check correctness
        loaded_config = load_config(index_path)
        @test isequal(config, loaded_config)
    end
end

@testset "load_doclens" begin
    mktempdir() do index_path
        num_docs_per_chunk = rand(1:10)
        num_chunks = rand(1:10)
        doclens_data = [rand(1:10, num_docs_per_chunk) for _ in 1:num_chunks]
        num_embeddings = sum(vcat(doclens_data...))
        plan_metadata = Dict(
            "num_chunks" => num_chunks, "num_embeddings" => num_embeddings)

        # Save the dummy data
        open(joinpath(index_path, "plan.json"), "w") do io
            JSON.print(io, plan_metadata)
        end
        for chunk_idx in 1:num_chunks
            doclens_file = joinpath(index_path, "doclens.$(chunk_idx).jld2")
            JLD2.save_object(doclens_file, doclens_data[chunk_idx])
        end

        # Load and verify doclens
        loaded_doclens = load_doclens(index_path)
        @test loaded_doclens == vcat(doclens_data...)

        # Delete plan.json and check for error
        rm(joinpath(index_path, "plan.json"))
        @test_throws ErrorException load_doclens(index_path)
    end
end

@testset "load_compressed_embs" begin
    mktempdir() do index_path
        # creating some dummy data
        config = ColBERTConfig(index_path = index_path)
        num_chunks = rand(1:10)
        num_embeddings = num_chunks * rand(5:10)
        chunk_size = div(num_embeddings, num_chunks)
        codes = rand(UInt32, num_embeddings)
        residuals = rand(
            UInt8, div(config.dim, 8) * config.nbits, num_embeddings)

        # save the dummy data
        save(config)
        plan_metadata = Dict(
            "num_embeddings" => num_embeddings, "num_chunks" => num_chunks)
        open(joinpath(index_path, "plan.json"), "w") do io
            JSON.print(io, plan_metadata)
        end
        for i in 1:num_chunks
            chunk_codes = codes[((i - 1) * chunk_size + 1):(i * chunk_size)]
            chunk_residuals = residuals[
                :, ((i - 1) * chunk_size + 1):(i * chunk_size)]
            JLD2.save_object(
                joinpath(index_path, "$(i).codes.jld2"), chunk_codes)
            JLD2.save_object(
                joinpath(index_path, "$(i).residuals.jld2"), chunk_residuals)
        end

        # Load the codes and residuals and verify correctness
        loaded_codes, loaded_residuals = load_compressed_embs(index_path)
        @test loaded_codes == codes
        @test loaded_residuals == residuals
    end
end

@testset "load_chunk_metadata_property inverts save_chunk_metadata_property" begin
    mktempdir() do index_path
        # Create dummy data and save it in chunks
        num_chunks = rand(1:10)
        plan_metadata = Dict("num_chunks" => num_chunks)
        open(joinpath(index_path, "plan.json"), "w") do io
            JSON.print(io, plan_metadata, 4)
        end
        chunk_metadata = [Dict("property" => "value" * string(idx)
                          ) for idx in 1:num_chunks]
        for (i, metadata) in enumerate(chunk_metadata)
            open(joinpath(index_path, "$(i).metadata.json"), "w") do f
                JSON.print(f, metadata, 4)
            end
        end

        # Save a new property to the chunks
        new_property = "property_new"
        new_properties = ["new_value" * string(idx) for idx in 1:num_chunks]
        save_chunk_metadata_property(index_path, new_property, new_properties)

        # Load the data from the chunks
        new_property_values = load_chunk_metadata_property(
            index_path, new_property)

        # Check for correctness
        @test new_property_values == new_properties
    end
end

@testset "load_codes" begin
    mktempdir() do index_path
        # Creating and saving some dummy data
        plan_metadata = Dict("num_chunks" => rand(1:10))
        open(joinpath(index_path, "plan.json"), "w") do io
            JSON.print(io, plan_metadata)
        end
        codes = Vector{UInt32}()
        for chunk_idx in 1:plan_metadata["num_chunks"]
            chunk_codes = UInt32.(rand(1:1000, rand(1:10)))  # Random UInt32 codes
            append!(codes, chunk_codes)
            JLD2.save_object(
                joinpath(index_path, "$(chunk_idx).codes.jld2"), chunk_codes)
        end

        # Load the data and verify for correctness
        loaded_codes = load_codes(index_path)
        @test loaded_codes == codes
    end
end

@testset "save_chunk" begin
    mktempdir() do index_path
        # Create some dummy data
        codes = rand(UInt32, 10)
        residuals = rand(UInt8, 10, 10)
        chunk_idx = 1
        passage_offset = 0
        doclens = rand(1:10, 10)

        # Save the data to disk using the save_chunk function
        save_chunk(
            index_path, codes, residuals, chunk_idx, passage_offset, doclens)

        # Load the data from disk and check for correctness
        codes_path = "$(joinpath(index_path, string(chunk_idx))).codes.jld2"
        residuals_path = "$(joinpath(index_path, string(chunk_idx))).residuals.jld2"
        doclens_path = joinpath(index_path, "doclens.$(chunk_idx).jld2")
        metadata_path = joinpath(index_path, "$(chunk_idx).metadata.json")

        # Load and check codes
        loaded_codes = JLD2.load_object(codes_path)
        @test loaded_codes == codes

        # Load and check residuals
        loaded_residuals = JLD2.load_object(residuals_path)
        @test loaded_residuals == residuals

        # Load and check doclens
        loaded_doclens = JLD2.load_object(doclens_path)
        @test loaded_doclens == doclens

        # Load and check metadata
        metadata_str = read(metadata_path, String)
        metadata = JSON.parse(metadata_str)
        @test metadata["passage_offset"] == passage_offset
        @test metadata["num_passages"] == length(doclens)
        @test metadata["num_embeddings"] == length(codes)
    end
end
