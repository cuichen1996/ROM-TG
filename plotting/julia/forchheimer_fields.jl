#!/usr/bin/env julia

"""Render the representative Forchheimer fields as a one-by-four figure.

The renderer uses saved full-resolution fields without solving a PDE.
Pass an input NPZ and output basename as the two required arguments. It accepts
either the archival pressure array or verified pressure_high/pressure_low arrays.
Mobility panels share one scale and pressure panels share another. The
sequential colormaps are perceptually uniform and avoid rainbow encoding.
"""

using CairoMakie
using LaTeXStrings
using NPZ

length(ARGS) == 2 || error("Usage: forchheimer_fields.jl INPUT.npz OUTPUT_BASENAME")
const DATA_PATH = abspath(ARGS[1])
const OUTPUT_BASE = abspath(ARGS[2])

const FOREGROUND = "#252338"
const AXIS_BACKGROUND = "#FCFCFE"
const FONT = "TeX Gyre Heros Makie"
const BOLD_FONT = "TeX Gyre Heros Makie Bold"
const MOBILITY_COLORMAP = :inferno
const PRESSURE_COLORMAP = :cividis
const MOBILITY_RANGE = (-0.001, 5.0)
const PRESSURE_RANGE = (0.0, 0.5)

function load_fields(path::String)
    isfile(path) || error("Missing saved Forchheimer fields: $path")
    data = npzread(path)
    legacy = Set(["phi", "kappa0", "alphas", "pressure", "mobility"])
    split = Set(["phi", "kappa0", "alphas", "pressure_high", "pressure_low", "mobility"])
    Set(keys(data)) in (legacy, split) || error("Unexpected arrays in $path")

    phi = Float64.(data["phi"])
    alphas = Float64.(data["alphas"])
    # Summation is exclusively for the displayed colors. Residual acceptance
    # continues to use the archived split-pressure independent validator.
    pressure = haskey(data, "pressure") ? Float64.(data["pressure"]) :
        Float64.(data["pressure_high"]) .+ Float64.(data["pressure_low"])
    mobility = Float64.(data["mobility"])
    size(phi) == (1023, 1023) || error("Unexpected phi shape: $(size(phi))")
    size(pressure) == (2, 1023, 1023) ||
        error("Unexpected pressure shape: $(size(pressure))")
    size(mobility) == size(pressure) || error("Mobility and pressure shapes differ")
    alphas == [0.0, 10.0] || error("Unexpected alpha values: $alphas")
    all(isfinite, phi) || error("phi contains non-finite values")
    all(isfinite, pressure) || error("pressure contains non-finite values")
    all(isfinite, mobility) || error("mobility contains non-finite values")
    all(>(0.0), mobility) || error("mobility must be strictly positive")

    log_mobility = log10.(mobility)
    extrema(log_mobility)[1] >= MOBILITY_RANGE[1] ||
        error("Mobility color range clips the saved minimum")
    extrema(log_mobility)[2] <= MOBILITY_RANGE[2] ||
        error("Mobility color range clips the saved maximum")
    extrema(pressure)[1] >= PRESSURE_RANGE[1] ||
        error("Pressure color range clips the saved minimum")
    extrema(pressure)[2] <= PRESSURE_RANGE[2] ||
        error("Pressure color range clips the saved maximum")
    return phi, alphas, pressure, log_mobility
end

function field_axis(position, title; show_ylabel::Bool)
    axis = Axis(
        position;
        backgroundcolor=AXIS_BACKGROUND,
        aspect=DataAspect(),
        title=title,
        titlefont=BOLD_FONT,
        titlesize=20,
        titlecolor=FOREGROUND,
        xlabel=L"x",
        ylabel=show_ylabel ? L"y" : "",
        xlabelsize=16,
        ylabelsize=16,
        xlabelcolor=FOREGROUND,
        ylabelcolor=FOREGROUND,
        xticks=([0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        yticks=([0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        xticklabelsize=14,
        yticklabelsize=14,
        xticklabelcolor=FOREGROUND,
        yticklabelcolor=FOREGROUND,
        xtickcolor=FOREGROUND,
        ytickcolor=FOREGROUND,
        xticksize=4,
        yticksize=4,
        xgridvisible=false,
        ygridvisible=false,
        topspinevisible=true,
        rightspinevisible=true,
        bottomspinevisible=true,
        leftspinevisible=true,
        topspinecolor=FOREGROUND,
        rightspinecolor=FOREGROUND,
        bottomspinecolor=FOREGROUND,
        leftspinecolor=FOREGROUND,
        spinewidth=1.0,
    )
    xlims!(axis, 0.0, 1.0)
    ylims!(axis, 0.0, 1.0)
    if !show_ylabel
        hideydecorations!(axis; grid=false, ticks=false, minorticks=false)
    end
    return axis
end

function draw_field!(axis, coordinates, field, phi; colormap, colorrange)
    image = heatmap!(
        axis,
        coordinates,
        coordinates,
        permutedims(field);
        colormap=colormap,
        colorrange=colorrange,
        interpolate=false,
        rasterize=3,
    )
    # A dark underlay keeps the white channel boundary visible at both light
    # and dark ends of the sequential color scales.
    contour!(
        axis,
        coordinates,
        coordinates,
        permutedims(phi);
        levels=[0.0],
        color=FOREGROUND,
        linewidth=2.2,
    )
    contour!(
        axis,
        coordinates,
        coordinates,
        permutedims(phi);
        levels=[0.0],
        color="#FFFFFF",
        linewidth=1.1,
    )
    return image
end

function main()
    phi, alphas, pressure, log_mobility = load_fields(DATA_PATH)
    nside = size(phi, 1)
    coordinates = range(0.5 / nside, 1.0 - 0.5 / nside; length=nside)
    mkpath(dirname(OUTPUT_BASE))

    figure = Figure(
        size=(1200, 390),
        backgroundcolor="#FFFFFF",
        font=FONT,
        figure_padding=(18, 22, 16, 16),
    )

    mobility_axes = [
        field_axis(figure[1, 1], L"\mathbf{(a)}\quad \alpha=0"; show_ylabel=true),
        field_axis(figure[1, 2], L"\mathbf{(b)}\quad \alpha=10"; show_ylabel=false),
    ]
    mobility_images = [
        draw_field!(
            mobility_axes[index],
            coordinates,
            view(log_mobility, index, :, :),
            phi;
            colormap=MOBILITY_COLORMAP,
            colorrange=MOBILITY_RANGE,
        ) for index in eachindex(mobility_axes)
    ]

    pressure_axes = [
        field_axis(figure[1, 3], L"\mathbf{(c)}\quad \alpha=0"; show_ylabel=true),
        field_axis(figure[1, 4], L"\mathbf{(d)}\quad \alpha=10"; show_ylabel=false),
    ]
    pressure_images = [
        draw_field!(
            pressure_axes[index],
            coordinates,
            view(pressure, index, :, :),
            phi;
            colormap=PRESSURE_COLORMAP,
            colorrange=PRESSURE_RANGE,
        ) for index in eachindex(pressure_axes)
    ]

    Colorbar(
        figure[2, 1:2],
        mobility_images[1];
        vertical=false,
        label=L"\log_{10}\lambda(\mathbf{x},p)",
        ticks=0:1:5,
        labelsize=16,
        ticklabelsize=13,
        labelcolor=FOREGROUND,
        ticklabelcolor=FOREGROUND,
        topspinecolor=FOREGROUND,
        rightspinecolor=FOREGROUND,
        bottomspinecolor=FOREGROUND,
        leftspinecolor=FOREGROUND,
        tickcolor=FOREGROUND,
        height=14,
    )
    Colorbar(
        figure[2, 3:4],
        pressure_images[1];
        vertical=false,
        label=L"p(\mathbf{x})",
        ticks=0.0:0.1:0.5,
        labelsize=16,
        ticklabelsize=13,
        labelcolor=FOREGROUND,
        ticklabelcolor=FOREGROUND,
        topspinecolor=FOREGROUND,
        rightspinecolor=FOREGROUND,
        bottomspinecolor=FOREGROUND,
        leftspinecolor=FOREGROUND,
        tickcolor=FOREGROUND,
        height=14,
    )

    colgap!(figure.layout, 14)
    rowgap!(figure.layout, 8)
    rowsize!(figure.layout, 2, Fixed(58))

    save("$OUTPUT_BASE.png", figure; px_per_unit=3)
    save("$OUTPUT_BASE.pdf", figure)
    save("$OUTPUT_BASE.svg", figure)

    println("saved $OUTPUT_BASE.{png,pdf,svg}")
    println("mobility extrema: $(extrema(log_mobility))")
    println("pressure extrema: $(extrema(pressure))")
end

main()
