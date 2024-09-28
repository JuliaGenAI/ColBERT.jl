using ColBERT
using Documenter

DocMeta.setdocmeta!(ColBERT, :DocTestSetup, :(using ColBERT); recursive = true)

makedocs(;
    modules = [ColBERT],
    authors = "Siddhant Chaudhary <urssidd@gmail.com> and contributors",
    sitename = "ColBERT",
    format = Documenter.HTML(;
        assets = String[],
        sidebar_sitename = false
    ),
    pages = [
        "Home" => "index.md",
        "Reference" => "api.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaGenAI/ColBERT.jl",
    target = "build",
    devbranch = "main",
    push_preview = true
)
