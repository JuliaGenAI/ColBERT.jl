using ColBERT
using .Iterators
using LinearAlgebra
using Random
using Test

const INT_TYPES = [
    Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128]
const FLOAT_TYPES = [Float16, Float32, Float64]

# include("Aqua.jl")
include("indexing/codecs/residual.jl")
include("utils.jl")
