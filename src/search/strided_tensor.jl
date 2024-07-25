"""
    StridedTensor(packed_tensor::Vector{Int}, lengths::Vector{Int})

Type to perform `ivf` operations efficiently.

# Arguments

- `packed_tensor`: The `ivf`, i.e the centroid to embedding map build during indexing. It is assumed that this map is stored as a `Vector`, wherein the embedding IDs are stored consecutively for each centroid ID.
- `lengths`: The total number of embeddings for a centroid ID, for each centroid.

# Returns

A [`StridedTensor`](@ref), which computes and stores all relevant data to lookup the `ivf` efficiently.

# Examples

```julia-repl

julia> using JLD2;

julia> ivf_path = joinpath(index_path, "ivf.jld2");

julia> ivf_dict = load(ivf_path);

julia> ivf, ivf_lengths = ivf_dict["ivf"], ivf_dict["ivf_lengths"];

julia> ivf = StridedTensor(ivf, ivf_lengths)
```
"""
struct StridedTensor
    tensor::Vector{Int}
    lengths::Vector{Int}
    strides::Vector{Int}
    offsets::Vector{Int}
    views::Dict{Int, Array{Int}}
end

function StridedTensor(packed_tensor::Vector{Int}, lengths::Vector{Int})
    tensor = packed_tensor
    strides = cat(_select_strides(lengths, [.5, .75, .9, .95]), [max(lengths...)], dims = 1) 
    strides = Int.(trunc.(strides))
    offsets = cat([0], cumsum(lengths), dims = 1)

    if offsets[length(offsets) - 1] + max(lengths...) > length(tensor)
        padding = zeros(Int, max(lengths...))
        tensor = cat(tensor, padding, dims = 1)
    end

    views = Dict(stride => _create_view(tensor, stride) for stride in strides)

    StridedTensor(
        tensor,
        lengths,
        strides,
        offsets,
        views
    )
end

function _select_strides(lengths::Vector{Int}, quantiles::Vector{Float64})
    if length(lengths) < 5000
        quantile(lengths, quantiles)
    else
        sample = rand(1:length(lengths), 2000)
        quantile(lengths[sample], quantiles)
    end
end

function _create_view(tensor::Vector{Int}, stride::Int)
    outdim = length(tensor) - stride + 1 
    size = (stride, outdim)
    tensor_view = zeros(Int, size)

    for column in 1:outdim
        tensor_view[:, column] = copy(tensor[column:column + stride - 1])
    end

    tensor_view
end
