# for now, we load collections in memory.
# will be good to implement on-disk data structures too.
Base.@kwdef struct Collection
    path::String
    data::Vector{String}
end
