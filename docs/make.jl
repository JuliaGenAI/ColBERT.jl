using ColBERT
using Documenter

DocMeta.setdocmeta!(ColBERT, :DocTestSetup, :(using ColBERT); recursive = true)

makedocs(;
    modules = [ColBERT],
    authors = "Siddhant Chaudhary <urssidd@gmail.com> and contributors",
    sitename = "ColBERT.jl",
    format = Documenter.HTML(;
        canonical = "https://codetalker7.github.io/ColBERT.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(;
    repo = "github.com/codetalker7/ColBERT.jl",
    devbranch = "main"
)
