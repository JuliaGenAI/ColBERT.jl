module ColBERT

# datasets
include("data/collection.jl")
include("data/queries.jl")

# config and other infra
include("infra/settings.jl")
include("infra/config.jl")

# models
include("modelling/checkpoint.jl")
include("modelling/colbert.jl")

end
