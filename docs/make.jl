using Documenter
using WamIPEDensity

DocMeta.setdocmeta!(
    WamIPEDensity,
    :DocTestSetup,
    :(using WamIPEDensity, Dates);
    recursive=true,
)

makedocs(
    sitename="WamIPEDensity.jl",
    modules=[WamIPEDensity],
    checkdocs=:none,
    format=Documenter.HTML(
        canonical="https://bourbon8464.github.io/WamIPEDensity.jl/stable/",
        edit_link="main",
        assets=["assets/custom.css"],
        prettyurls=get(ENV, "CI", "false") == "true",
    ),
    pages=[
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "WFS Workflows" => "wfs-workflows.md",
        "How It Works" => "how-it-works.md",
        "SpaceAGORA Integration" => "spaceagora.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo="github.com/Bourbon8464/WamIPEDensity.jl.git",
    devbranch="main",
    push_preview=true,
)
