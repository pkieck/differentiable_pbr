"""
plotting.jl — Shared field-plotting helper for the example cases.

`plot_state(u, s, g; ...)` renders every quantity in the flat state vector
`[u_r | u_z | p | T_f | T_s | Y_1..Y_Ns]` as a panel of (r,z) heatmaps and
returns the assembled `Plots` figure.

The grid is 2D axisymmetric: the axial coordinate z (flow direction) is on the
horizontal axis, the radial coordinate r (axis→wall) on the vertical axis. Each
field matrix is stored (nr × nz), which is exactly the orientation `heatmap`
wants for `heatmap(z, r, M)`.
"""

using Plots
using Printf

# Pretty labels + units for the five fixed fields.
const _FIELD_LABELS = ("u_r  [m/s]", "u_z  [m/s]", "p  [Pa]",
                       "T_f  [K]",   "T_s  [K]")

"""
    plot_state(u, s, g; species_names, title, density, extras) -> Plots.Plot

Build a paneled heatmap of all state fields for state vector `u` on grid `g`
with layout `s`. `species_names` (length `s.Ns`) labels the species panels;
pass `density = ρ_matrix` to add an extra ρ(r,z) panel (compressible cases).
`extras` is a vector of `label => matrix` pairs for additional panels (e.g. a
catalyst-activity or porosity field), drawn first so they sit at the top.
"""
function plot_state(u, s, g; species_names::Vector{String}=String[],
                    title::AbstractString="", density=nothing,
                    extras::AbstractVector=Pair{String,Any}[])
    z = g.z .* 1e3          # mm for readable axes
    r = g.r .* 1e3

    panels = Any[]
    labels = String[]

    # Extra fields (catalyst activity, porosity, …) — drawn first.
    for (lab, M) in extras
        push!(panels, M)
        push!(labels, lab)
    end

    # Fixed fields u_r, u_z, p, T_f, T_s
    for f in 1:5
        push!(panels, field_mat(u, s, f))
        push!(labels, _FIELD_LABELS[f])
    end
    # Species
    for k in 1:s.Ns
        push!(panels, species_mat(u, s, k))
        name = k <= length(species_names) ? species_names[k] : "Y_$k"
        push!(labels, "$name  [mass frac]")
    end
    # Optional density
    if density !== nothing
        push!(panels, density)
        push!(labels, "ρ  [kg/m³]")
    end

    plts = Plots.Plot[]
    for (M, lab) in zip(panels, labels)
        p = heatmap(z, r, M;
                    title=lab, titlefontsize=9,
                    xlabel="z [mm]", ylabel="r [mm]",
                    guidefontsize=7, tickfontsize=6,
                    colorbar=true, c=:viridis, aspect_ratio=:auto)
        push!(plts, p)
    end

    n = length(plts)
    ncols = n <= 3 ? n : (n <= 8 ? 3 : 4)
    nrows = cld(n, ncols)
    fig = plot(plts...; layout=(nrows, ncols),
               size=(360 * ncols, 260 * nrows),
               plot_title=title, plot_titlefontsize=12,
               left_margin=4Plots.mm, bottom_margin=4Plots.mm)
    return fig
end

"""
    save_fig(fig, name) -> path

Save `fig` as a PNG under examples/figures/ and print the path.
"""
function save_fig(fig, name::AbstractString)
    dir  = joinpath(@__DIR__, "figures")
    isdir(dir) || mkpath(dir)
    path = joinpath(dir, name)
    savefig(fig, path)
    @printf "  saved figure: %s\n" path
    return path
end
