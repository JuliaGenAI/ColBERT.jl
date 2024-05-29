# for now, we load collections in memory.
# will be good to implement on-disk data structures too.
struct Collection
    path::String
    data::Vector{String}
end

function Collection(path::String)
    file = CSV.File(path; delim='\t', header = [:pid, :text], types = Dict(:pid => Int, :text => String), debug=true, quoted=false)
    Collection(path, file.text)
end
