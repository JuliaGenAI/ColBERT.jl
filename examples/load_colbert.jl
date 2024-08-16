using Transformers
using JSON3
using Transformers.HuggingFace
const HF = Transformers.HuggingFace

"""
    _load_tokenizer_config(path_config)

Load tokenizer config locally.
"""
function _load_tokenizer_config(path_config::AbstractString)
    @assert isfile(path_config) "Tokenizer config file not found: $path_config"
    return JSON3.read(read(path_config))
end

"""
    extract_tokenizer_type(tkr_type::AbstractString)

Extract tokenizer type from config.
"""
function extract_tokenizer_type(tkr_type::AbstractString)
    m = match(r"(\S+)Tokenizer(Fast)?", tkr_type)
    isnothing(m) &&
        error("Unknown tokenizer: $tkr_type")
    tkr_type = Symbol(lowercase(m.captures[1]))
end

"""
    _load_tokenizer(cfg::HF.HGFConfig; path_tokenizer_config::AbstractString,
        path_special_tokens_map::AbstractString, path_tokenizer::AbstractString)

Local tokenizer loader.
"""
function _load_tokenizer(cfg::HF.HGFConfig;
        path_tokenizer_config::AbstractString,
        path_special_tokens_map::AbstractString, path_tokenizer::AbstractString)
    @assert isfile(path_tokenizer_config) "Tokenizer config file not found: $path_tokenizer_config"
    @assert isfile(path_special_tokens_map) "Special tokens map file not found: $path_special_tokens_map"
    @assert isfile(path_tokenizer) "Tokenizer file not found: $path_tokenizer"
    ## load tokenizer config
    tkr_cfg = _load_tokenizer_config(path_tokenizer_config)
    tkr_type_sym = extract_tokenizer_type(tkr_cfg.tokenizer_class)
    tkr_type = HF.tokenizer_type(tkr_type_sym) # eg, Val(:bert)()
    ## load special tokens
    special_tokens = HF.load_special_tokens_map(path_special_tokens_map)
    ## load tokenizer
    kwargs = HF.extract_fast_tkr_kwargs(
        tkr_type, tkr_cfg, cfg, special_tokens)
    tokenizer, vocab, process_config, decode, textprocess = HF.load_fast_tokenizer(
        tkr_type, path_tokenizer, cfg)
    for (k, v) in process_config
        kwargs[k] = v
    end
    ## construct tokenizer and mutate the decode +textprocess pipelines
    tkr = HF.encoder_construct(
        tkr_type, tokenizer, vocab; kwargs...)
    tkr = HF.setproperties!!(
        tkr, (; decode, textprocess))
    return tkr
end

"""
    _load_model(cfg::HF.HGFConfig; path_model::AbstractString,
        trainmode::Bool = false, lazy::Bool = false, mmap::Bool = true)

Local model loader.
"""
function _load_model(cfg::HF.HGFConfig;
        path_model::AbstractString,
        trainmode::Bool = false, lazy::Bool = false, mmap::Bool = true)
    @assert isfile(path_model) "Model file not found: $path_model"
    @assert endswith(path_model, ".bin") "Model file must end with .bin (type torch `pickle`): $path_model"
    ## Assume fixed
    task = :model

    ## Load state dict
    # We know we have pytorch_model.bin -> so format is :pickle and it's a single file
    # status = HF.singlefilename(HF.WeightStatus{:pickle}) 
    status = HF.HasSingleFile{:pickle}(path_model)
    state_dict = HF.load_state_dict_from(
        status; lazy, mmap)

    ##
    model_type = HF.get_model_type(
        HF.getconfigname(cfg), task)
    basekey = String(HF.basemodelkey(model_type))
    if HF.isbasemodel(model_type)
        prefix = HF.haskeystartswith(
            state_dict, basekey) ? basekey : ""
    else
        prefix = ""
        if !HF.haskeystartswith(
            state_dict, basekey)
            new_state_dict = OrderedDict{
                Any, Any}()
            for (key, val) in state_dict
                new_state_dict[joinname(basekey, key)] = val
            end
            state_dict = new_state_dict
        end
    end
    model = load_model(
        model_type, cfg, state_dict, prefix)
    trainmode || (model = Layers.testmode(model))
    return model
end

"""
    load_hgf_pretrained_local(dir_spec::AbstractString;
        path_config::Union{Nothing, AbstractString} = nothing,
        path_tokenizer_config::Union{Nothing, AbstractString} = nothing,
        path_special_tokens_map::Union{Nothing, AbstractString} = nothing,
        path_tokenizer::Union{Nothing, AbstractString} = nothing,
        path_model::Union{Nothing, AbstractString} = nothing,
        kwargs...

)

Local model loader. Honors the `load_hgf_pretrained` interface, where you can request
specific files to be loaded, eg, `my/dir/to/model:tokenizer` or `my/dir/to/model:config`.

# Arguments

  - `dir_spec::AbstractString`: Directory specification (item specific after the colon is optional), eg, `my/dir/to/model` or `my/dir/to/model:tokenizer`.
  - `path_config::Union{Nothing, AbstractString}`: Path to config file.
  - `path_tokenizer_config::Union{Nothing, AbstractString}`: Path to tokenizer config file.
  - `path_special_tokens_map::Union{Nothing, AbstractString}`: Path to special tokens map file.
  - `path_tokenizer::Union{Nothing, AbstractString}`: Path to tokenizer file.
  - `path_model::Union{Nothing, AbstractString}`: Path to model file.
  - `kwargs...`: Additional keyword arguments for `_load_model` function like `mmap`, `lazy`, `trainmode`.
"""
function load_hgf_pretrained_local(
        dir_spec::AbstractString;
        path_config::Union{
            Nothing, AbstractString} = nothing,
        path_tokenizer_config::Union{
            Nothing, AbstractString} = nothing,
        path_special_tokens_map::Union{
            Nothing, AbstractString} = nothing,
        path_tokenizer::Union{
            Nothing, AbstractString} = nothing,
        path_model::Union{
            Nothing, AbstractString} = nothing,
        kwargs...
)

    ## Extract if item was provided
    name_item = rsplit(dir_spec, ':'; limit = 2)
    all = length(name_item) == 1
    dir, item = if all
        dir_spec, "model"
    else
        Iterators.map(String, name_item)
    end
    item = lowercase(item)
    ## Set paths
    @assert isdir(dir) "Local directory not found: $dir"
    if isnothing(path_config)
        path_config = joinpath(dir, "config.json")
    end
    if isnothing(path_tokenizer_config)
        path_tokenizer_config = joinpath(
            dir, "tokenizer_config.json")
    end
    if isnothing(path_special_tokens_map)
        path_special_tokens_map = joinpath(
            dir, "special_tokens_map.json")
    end
    if isnothing(path_tokenizer)
        path_tokenizer = joinpath(
            dir, "tokenizer.json")
    end
    if isnothing(path_model)
        path_model = joinpath(
            dir, "pytorch_model.bin")
    end
    ## Check if they exist
    @assert isfile(path_config) "Config file not found: $path_config"
    @assert isfile(path_tokenizer_config) "Tokenizer config file not found: $path_tokenizer_config"
    @assert isfile(path_special_tokens_map) "Special tokens map file not found: $path_special_tokens_map"
    @assert isfile(path_tokenizer) "Tokenizer file not found: $path_tokenizer"
    @assert isfile(path_model) "Model file not found: $path_model"

    ## load config
    cfg = HF._load_config(path_config)
    item == "config" && return cfg

    ## load tokenizer
    if item == "tokenizer" || all
        tkr = _load_tokenizer(
            cfg; path_tokenizer_config,
            path_special_tokens_map,
            path_tokenizer)
    end
    item == "tokenizer" && return tkr

    ## load model
    model = _load_model(
        cfg; path_model, kwargs...)

    if all
        return tkr, model
    else
        return model
    end
end

## Example
using Transformers.TextEncoders

# My files can be found in this directory
dir = "colbert-ir"
textenc, model = load_hgf_pretrained_local(dir)

encoded = encode(textenc, "test it")
output = model(encoded)
