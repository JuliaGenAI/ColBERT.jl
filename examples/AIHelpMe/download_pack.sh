# the knowledge packs
wget https://media.githubusercontent.com/media/svilupp/AIHelpMeArtifacts/main/artifacts/genie__v20240818__textembedding3large-1024-Bool__v1.0.tar.gz?download=true \
    -O genie_knowledge_pack.tar.gz
wget https://media.githubusercontent.com/media/svilupp/AIHelpMeArtifacts/main/artifacts/julia__v1.10.2__textembedding3large-1024-Bool__v1.0.tar.gz?download=true \
    -O julia_v1.10.2_knowledge_pack.tar.gz
wget https://media.githubusercontent.com/media/svilupp/AIHelpMeArtifacts/main/artifacts/juliadata__v20240716__textembedding3large-1024-Bool__v1.0.tar.gz?download=true \
    -O juliadata_knowledge_pack.tar.gz
wget https://media.githubusercontent.com/media/svilupp/AIHelpMeArtifacts/main/artifacts/julialang__v20240819__textembedding3large-1024-Bool__v1.0.tar.gz?download=true \
    -O julialang_knowledge_pack.tar.gz
wget https://media.githubusercontent.com/media/svilupp/AIHelpMeArtifacts/main/artifacts/makie__v20240716__textembedding3large-1024-Bool__v1.0.tar.gz?download=true \
    -O makie_knowledge_pack.tar.gz
wget https://media.githubusercontent.com/media/svilupp/AIHelpMeArtifacts/main/artifacts/plots__v20240716__textembedding3large-1024-Bool__v1.0.tar.gz?download=true \
    -O plots_knowledge_pack.tar.gz
wget https://media.githubusercontent.com/media/svilupp/AIHelpMeArtifacts/main/artifacts/sciml__v20240716__textembedding3large-1024-Bool__v1.0.tar.gz?download=true \
    -O sciml_knowledge_pack.tar.gz
wget https://media.githubusercontent.com/media/svilupp/AIHelpMeArtifacts/main/artifacts/tidier__v20240716__textembedding3large-1024-Bool__v1.0.tar.gz?download=true \
    -O tider_knowledge_pack.tar.gz

# unpack all the packs
tar -xvzf genie_knowledge_pack.tar.gz 
tar -xvzf julia_v1.10.2_knowledge_pack.tar.gz 
tar -xvzf juliadata_knowledge_pack.tar.gz 
tar -xvzf julialang_knowledge_pack.tar.gz 
tar -xvzf makie_knowledge_pack.tar.gz 
tar -xvzf plots_knowledge_pack.tar.gz 
tar -xvzf sciml_knowledge_pack.tar.gz 
tar -xvzf tider_knowledge_pack.tar.gz 


# eval pack
wget https://raw.githubusercontent.com/svilupp/AIHelpMe.jl/main/evaluations/JuliaData/dataframe_combined_filtered-qa-evals.json -O qa_evals.json
