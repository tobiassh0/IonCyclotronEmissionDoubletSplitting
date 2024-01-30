
using ArgParse
const argsettings = ArgParseSettings()

@add_arg_table! argsettings begin
    "--minorityenergyMeV"
        help = "The energy (in MeV) of the minority particle"
        arg_type = Float64
        default = 3.5
    "--secondfuelionconcentrationratio"
        help = "The concentration ratio of the second fuel ion species with respect to electron density"
        arg_type = Float64
        default = 0.11
    "--pitch"
        help = "The cosine of the pitch angle, or pitch of the energetic species"
        arg_type = Float64
        default = -0.646
    "--ngridpoints"
        help = "The number of grid points in each direciton cyclotronfrequency wavenumber space (default 1024)"
        arg_type = Int
        default = 2^10
    "--kparamax"
        help = "The upper limit in kpara in units of Ωi / Va"
        arg_type = Float64
        default = 4.0
    "--kperpmax"
        help = "The upper limit in kperp in units of Ωi / Va"
        arg_type = Float64
        default = 15.0
    "--vthpararatio"
        help = "The parallel thermal speed of the enerrgtic speices as a function of its bulk speed"
        arg_type = Float64
        default = 0.01
    "--vthperpratio"
        help = "The perpendicular thermal speed of the enerrgtic speices as a function of its bulk speed"
        arg_type = Float64
        default = 0.01
    "--nameextension"
        help = "The name extension to append to the figure files"
        arg_type = String
        default = ""
    "--temperaturekeV"
        help = "The temperature (in keV) of the electrons and bulks ions, assumed equal"
        arg_type = Float64
        default = 1.0
    "--magneticField"
        help = "The magnetic field (T)"
        arg_type = Float64
        default = 2.07
    "--electronDensity"
        help = "The number density of the electrons (n0 == ne) "
        arg_type = Float64
        default = 1.7e19
    "--minorityConcentration"
        help = "The concentration of the minority ion"
        arg_type = Float64
        default = 1.5e-4
end

const parsedargs = parse_args(ARGS, argsettings)
@show parsedargs

const _B0 = parsedargs["magneticField"]
const _n0 = parsedargs["electronDensity"]
const ximin = parsedargs["minorityConcentration"]
const ξ2 = parsedargs["secondfuelionconcentrationratio"]

using Distributed, Dates
using Random, ImageFiltering, Statistics
using Dierckx, Contour, JLD2, DelimitedFiles
using Plots; gr()
# Plots.gr() # .pyplot()

const nprocsadded = div(Sys.CPU_THREADS, 2)

# mass ratios
const mp = 1836.2
const md = 3671.5
const mT = 5497.93
const mHe3 = 5497.885
const mα = 7294.3

@everywhere using ProgressMeter # for some reason must be up here on its own
@everywhere using StaticArrays
@everywhere using FastClosures
@everywhere using NLsolve
@everywhere using LinearMaxwellVlasov, LinearAlgebra, WindingNelderMead

function getVA(m1, m2, mmin, _xi2, z1, z2, zmin)
  ξ2 = _xi2
  n2 = ξ2*n0
  nmin = ξ*n0
  n1 = (1/z1)*(n0-z2*n2-zmin*nmin) # 1 / (1.0 + 2*ξ)
  @assert n0 ≈ z1*n1 + z2*n2 + zmin*nmin
  density_weighted = n1*m1 + n2*m2 + nmin*mmin
  Va = B0 / sqrt(LinearMaxwellVlasov.μ₀*density_weighted)
  return Va
end

function make2d(rowedges, coledges, rowvals, colvals, vals)
  @assert issorted(rowedges)
  @assert issorted(coledges)
  @assert length(rowvals) == length(colvals) == length(vals)
  Z = Array{Union{Float64, Missing}}(zeros(length(rowedges), length(coledges)) .- Inf)
  for k in eachindex(rowvals, colvals, vals)
    (rowedges[1] <= rowvals[k] <= rowedges[end]) || continue
    (coledges[1] <= colvals[k] <= coledges[end]) || continue
    i = findlast(rowvals[k] >= x for x in rowedges)
    j = findlast(colvals[k] >= x for x in coledges)
    isnothing(i) && continue
    isnothing(j) && continue
    @assert 1 <= i <= size(Z, 1)
    @assert 1 <= j <= size(Z, 2)
    Z[i, j] = max(Z[i, j], vals[k])
  end
  for i in eachindex(Z)
      Z[i] == -Inf && (Z[i] = missing)
  end
  return Z
end

function plotit(sols, file_extension=name_extension, fontsize=9)
  sols = sort(sols, by=s->imag(s.frequency))
  ωs = [sol.frequency for sol in sols]./w0
  nunstable = sum(imag(sol.frequency) > 0 for sol in sols)
  @info "There are $nunstable unstable solutions out of $(length(sols))"
  kzs = [para(sol.wavenumber) for sol in sols]./k0
  k⊥s = [perp(sol.wavenumber) for sol in sols]./k0
  xk⊥s = sort(unique(k⊥s))
  ykzs = sort(unique(kzs))

  ks = [abs(sol.wavenumber) for sol in sols]./k0
  kθs = atan.(k⊥s, kzs)
  extremaangles = collect(extrema(kθs))

  _median(patch) = median(filter(x->!ismissing(x), patch))
  realωssmooth = make2d(ykzs, xk⊥s, kzs, k⊥s, real.(ωs))
  try
  #    realωssmooth = mapwindow(_median, realωssmooth, (5, 5))
  #    realωssmooth = imfilter(realωssmooth, Kernel.gaussian(3))
  catch
    @warn "Smoothing failed"
  end

  realspline = nothing
  imagspline = nothing
  try
    smoothing = length(ωs) * 1e-4
  #    realspline = Dierckx.Spline2D(xk⊥s, ykzs, realωssmooth'; kx=4, ky=4, s=smoothing)
  #    imagspline = Dierckx.Spline2D(k⊥s, kzs, imag.(ωs); kx=4, ky=4, s=smoothing)
  catch err
    @warn "Caught $err. Continuing."
  end

  function plotangles(;writeangles=true)
    for θdeg ∈ vcat(collect.((30:5:80, 81:99, 100:5:150))...)
      θ = θdeg * π / 180
      xs = sin(θ) .* [0, maximum(ks)]
      ys = cos(θ) .* [0, maximum(ks)]
      linestyle = mod(θdeg, 5) == 0 ? :solid : :dash
      Plots.plot!(xs, ys, linecolor=:grey, linewidth=0.5, linestyle=linestyle)
      writeangles || continue
      if atan(maximum(k⊥s), maximum(kzs)) < θ < atan(maximum(k⊥s), minimum(kzs))
        xi, yi = xs[end], ys[end]
        isapprox(yi, maximum(kzs), rtol=0.01, atol=0.01) && (yi += 0.075)
        isapprox(xi, maximum(k⊥s), rtol=0.01, atol=0.01) && (xi += 0.1)
        isapprox(xi, minimum(k⊥s), rtol=0.01, atol=0.01) && (xi -= 0.2)
        Plots.annotate!([(xi, yi, Plots.text("\$ $(θdeg)^{\\circ}\$", fontsize, :black))])
      end
    end
  end

  function plotcontours(spline, contourlevels, skipannotation=x->false)
    isnothing(spline) && return nothing
    x, y = sort(unique(k⊥s)), sort(unique(kzs))
    z = evalgrid(spline, x, y)
    for cl ∈ Contour.levels(Contour.contours(x, y, z, contourlevels))
      lvl = try; Int(Contour.level(cl)); catch; Contour.level(cl); end
      for line ∈ Contour.lines(cl)
          xs, ys = Contour.coordinates(line)
          θs = atan.(xs, ys)
          ps = sortperm(θs, rev=true)
          xs, ys, θs = xs[ps], ys[ps], θs[ps]
          mask = minimum(extremaangles) .< θs .< maximum(extremaangles)
          any(mask) || continue
          xs, ys = xs[mask], ys[mask]
          Plots.plot!(xs, ys, color=:grey, linewidth=0.5)
          skipannotation(ys) && continue
          yi, index = findmax(ys)
          xi = xs[index]
          if !(isapprox(xi, minimum(k⊥s), rtol=0.01, atol=0.5) ||
               isapprox(yi, maximum(kzs), rtol=0.01, atol=0.01))
            continue
          end
          isapprox(xi, maximum(k⊥s), rtol=0.1, atol=0.5) && continue
          isapprox(yi, maximum(kzs), rtol=0.1, atol=0.5) && (yi += 0.075)
          isapprox(xi, minimum(k⊥s), rtol=0.1, atol=0.5) && (xi = -0.1)
          Plots.annotate!([(xi, yi, Plots.text("\$\\it{$lvl}\$", fontsize-1, :black))])
      end
    end
  end

  msize = 2
  mshape = :square
  function plotter2d(z, xlabel, ylabel, colorgrad,
      climmin=minimum(z[@. !ismissing(z)]), climmax=maximum(z[@. !ismissing(z)]))
    zcolor = make2d(ykzs, xk⊥s, kzs, k⊥s, z)
    dx = (xk⊥s[2] - xk⊥s[1]) / (length(xk⊥s) - 1)
    dy = (ykzs[2] - ykzs[1]) / (length(ykzs) - 1)
    h = Plots.heatmap(xk⊥s, ykzs, zcolor, framestyle=:box, c=colorgrad,
      xlims=(minimum(xk⊥s) - dx/2, maximum(xk⊥s) + dx/2),
      ylims=(minimum(ykzs) - dy/2, maximum(ykzs) + dy/2),
      clims=(climmin, climmax), xticks=0:Int(round(maximum(xk⊥s))),
      xlabel=xlabel, ylabel=ylabel)
  end
 
  xlabel = "\$\\mathrm{Perpendicular\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
  ylabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
  zs = real.(ωs)
  climmax = maximum(zs)
  plotter2d(zs, xlabel, ylabel, Plots.cgrad(), 0.0, climmax)
  Plots.title!(" ")
  plotangles(writeangles=false)
  Plots.plot!(legend=false)
  Plots.savefig("$dirpath/ICE2D_real_$file_extension.pdf")

  ω0s = [fastzerobetamagnetoacousticfrequency(Va, sol.wavenumber, Ω1) for
    sol in sols] / w0
  zs = real.(ωs) ./ ω0s
  climmin = minimum(zs)
  climmax = maximum(zs)
  plotter2d(zs, xlabel, ylabel, Plots.cgrad(), climmin, climmax)
  Plots.title!(" ")
  plotangles(writeangles=false)
  Plots.plot!(legend=false)
  Plots.savefig("$dirpath/ICE2D_real_div_guess_$file_extension.pdf")

  zs = iseven.(Int64.(floor.(real.(ωs))))
  climmax = maximum(zs)
  plotter2d(zs, xlabel, ylabel, Plots.cgrad(), 0.0, climmax)
  Plots.title!(" ")
  plotangles(writeangles=false)
  plotcontours(realspline, collect(1:50), y -> y[end] < 0)
  Plots.plot!(legend=false)
  Plots.savefig("$dirpath/ICE2D_evenfloorreal_real_$file_extension.pdf")

  zs = imag.(ωs)
  climmax = maximum(zs)
  colorgrad = Plots.cgrad([:cyan, :blue, :darkblue, :midnightblue, :black, :darkred, :red, :orange, :yellow])
  plotter2d(zs, xlabel, ylabel, colorgrad, -climmax, climmax)
  Plots.title!(" ")
  Plots.plot!(legend=false)
  plotcontours(realspline, collect(1:50), y -> y[end] < 0)
  plotangles(writeangles=false)
  Plots.savefig("$dirpath/ICE2D_imag_$file_extension.pdf")

  xlabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
  ylabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"

  maxrealfreq = 15

  xωrs = collect(range(0, stop=maxrealfreq, length=length(ykzs)))
  ykzs_ = ykzs[1:2:end]
  xωrs_ = xωrs[1:2:end]
  zcolor = make2d(ykzs_, xωrs_, kzs, real.(ωs), imag.(ωs))
  climmin, climmax = extrema(zcolor[@. !ismissing(zcolor)])
  dx = (xωrs_[2] - xωrs_[1]) / (length(xωrs_) - 1)
  dy = (ykzs_[2] - ykzs_[1]) / (length(ykzs_) - 1)
  h_kwhp = Plots.heatmap(xωrs_, ykzs_, zcolor, framestyle=:box, c=colorgrad,
    xlims=(minimum(xωrs_) - dx/2, maximum(xωrs_) + dx/2),
    ylims=(minimum(ykzs_) - dy/2, maximum(ykzs_) + dy/2),
    clims=(-climmax, climmax), xticks=0:Int(round(maximum(xωrs))),
    xlabel=xlabel, ylabel=ylabel)
  Plots.title!(" ")
  Plots.plot!(legend=false)
  #plotcontours(realspline, collect(1:50), y -> y[end] < 0)
  #plotangles(writeangles=false)
  Plots.savefig("$dirpath/ICE2D_kw_grid_$file_extension.pdf")

  imaglolim = 1e-5
  mask = shuffle(findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= maxrealfreq)))
  @warn "Scatter plots rendering with $(length(mask)) points."

  colorgrad = Plots.cgrad([:black, :darkred, :red, :orange, :yellow])
  perm = sortperm(imag.(ωs[mask]))
  h_kwsc = Plots.scatter(real.(ωs[mask][perm]), kzs[mask][perm],
    zcolor=imag.(ωs[mask][perm]), framestyle=:box, lims=:round,
    markersize=msize+1, markerstrokewidth=-1, markershape=:circle,
    c=colorgrad, xticks=(0:maxrealfreq), yticks=unique(Int.(round.(ykzs))),
    xlabel=xlabel, ylabel=ylabel, legend=:topleft)
  Plots.plot!(legend=false)
  Plots.savefig("$dirpath/ICE2D_KF_scatter_$file_extension.pdf")

  zlabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
  xlabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
  ylabel = "\$\\mathrm{Perpendicular\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
  Plots.scatter(kzs[mask], k⊥s[mask], 0.0 .* real.(ωs[mask]), framestyle=:box, lims=:round,
     markeralpha=0.1, markersize=msize+1, markerstrokewidth=-1, markershape=:circle, c=:grey)
  Plots.scatter!(0 .* kzs[mask] .+ minimum(kzs), k⊥s[mask], real.(ωs[mask]), framestyle=:box, lims=:round,
     markeralpha=0.1, markersize=msize+1, markerstrokewidth=-1, markershape=:circle, c=:grey)
  Plots.scatter!(kzs[mask], 0 .* k⊥s[mask] .+ maximum(k⊥s), real.(ωs[mask]), framestyle=:box, lims=:round,
     markeralpha=0.1, markersize=msize+1, markerstrokewidth=-1, markershape=:circle, c=:grey)
  Plots.scatter!(kzs[mask], k⊥s[mask], real.(ωs[mask]), zcolor=imag.(ωs[mask]),
    framestyle=:box, lims=:round, markeralpha=0.2,
    markersize=msize+1, markerstrokewidth=-1, markershape=:circle,
    c=colorgrad, zticks=(0:1:maxrealfreq), camera = (10, 30),
    xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
  Plots.plot!(legend=false)
  Plots.savefig("$dirpath/ICE2D_KKF_$file_extension.pdf")

  xlabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
  ylabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"


  ylabel = "\$\\mathrm{Growth\\ Rate} \\ [\\Omega_{i}]\$"
  mask = shuffle(findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= maxrealfreq)))
  h_gf = Plots.scatter(real.(ωs[mask]), imag.(ωs[mask]),
    zcolor=kzs[mask], framestyle=:box, lims=:round,
    markersize=msize+1, markerstrokewidth=-1, markershape=:circle,
    c=colorgrad, xticks=(0:maxrealfreq),
    xlabel=xlabel, ylabel=ylabel, legend=:topleft)
  Plots.plot!(legend=false)
  Plots.savefig("$dirpath/ICE2D_F_$file_extension.pdf")

  colorgrad1 = Plots.cgrad([:cyan, :red, :blue, :orange, :green,
                            :black, :yellow])
  mask = shuffle(findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= maxrealfreq)))
  zcolor=(real.(ωs[mask]) .- vαz/Va .* kzs[mask])
  h2 = Plots.scatter(real.(ωs[mask]), imag.(ωs[mask]),
    zcolor=zcolor, framestyle=:box, lims=:round,
    markersize=msize+1, markerstrokewidth=-1, markershape=:circle,
    c=colorgrad1, clims=(0, 15), xticks=(0:maxrealfreq), # increase maximum freq for doppler shift plot
    xlabel=xlabel, ylabel=ylabel, legend=:topleft)
  Plots.plot!(legend=false)
  Plots.savefig("$dirpath/ICE2D_F_Doppler_$file_extension.pdf")

  maxw = 25
  Δθ = 0.1
  for θ in 85:0.5:95
    try
      mask1 = [-Δθ < angle(sol.wavenumber) * 180 / pi - θ < Δθ for sol in sols]
      mask2 = collect(@. (imag(ωs) > imaglolim) & (real(ωs) <= maxw))
      mask = findall(mask1 .& mask2)
      h1 = Plots.scatter(real.(ωs[mask]), imag.(ωs[mask]),
        zcolor=kzs[mask], framestyle=:box, lims=:round, xlims=(0, maxw),
        markersize=msize+1, markerstrokewidth=0, markershape=:circle,
        c=colorgrad, xticks=(0:2:maxw), label="($θ)ᵒ",
        xlabel=xlabel, ylabel=ylabel, legend=:topleft)
      Plots.plot!(legend=false)
      Plots.savefig("$dirpath/ICE2D_1D_Angle_$(θ)_$file_extension.pdf")
    catch
    end
  end

  xlabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
  ylabel = "\$\\mathrm{Propagation\\ Angle} \\ [^{\\circ}]\$"

  colorgrad = Plots.cgrad([:cyan, :black, :darkred, :red, :orange, :yellow])
  mask = findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= maxrealfreq))
  zcolor=imag.(ωs[mask])
  h4 = Plots.scatter(real.(ωs[mask]), kθs[mask] .* 180 / π,
    zcolor=imag.(ωs[mask]), lims=:round,
    markersize=msize, markerstrokewidth=-1, markershape=mshape, framestyle=:box,
    c=Plots.cgrad([:black, :darkred, :red, :orange, :yellow]),
    clims=(0, maximum(imag.(ωs[mask]))),
    yticks=(0:10:180), xticks=(0:maxrealfreq), xlabel=xlabel, ylabel=ylabel)
  Plots.plot!(legend=false)
  Plots.savefig("$dirpath/ICE2D_TF_$file_extension.pdf")

  function relative(p, rx, ry)
    xlims = Plots.xlims(p)
    ylims = Plots.ylims(p)
    return xlims[1] + rx * (xlims[2]-xlims[1]), ylims[1] + ry * (ylims[2] - ylims[1])
   end
  
  Plots.xlabel!(h_gf, "")
  Plots.xticks!(h_gf, 0:maxrealfreq)
  Plots.annotate!(h_gf, [(relative(h_gf, 0.02, 0.95)..., Plots.text("(a)", fontsize, :black))])
  Plots.annotate!(h_kwsc, [(relative(h_kwsc, 0.02, 0.95)..., Plots.text("(b)", fontsize, :black))])
  Plots.plot!(h_gf, xlims=(0, maxrealfreq))
  Plots.plot!(h_kwsc, xlims=(0, maxrealfreq), ylims=(-kparamax, kparamax))
  Plots.plot(h_gf, h_kwsc, link=:x, layout=@layout [a; b])
  Plots.plot!(size=(800,600))
  Plots.savefig("$dirpath/ICE2D_Combo_$file_extension.pdf")
end

function plotter2d(z, xlabel, ylabel, colorgrad, climmin=minimum(z[@. !ismissing(z)]), climmax=maximum(z[@. !ismissing(z)]))
  zcolor = make2d(ykzs, xk⊥s, kzs, k⊥s, z)
  dx = (xk⊥s[2] - xk⊥s[1]) / (length(xk⊥s) - 1)
  dy = (ykzs[2] - ykzs[1]) / (length(ykzs) - 1)

  h = Plots.heatmap(xk⊥s, ykzs, zcolor, framestyle=:box, c=colorgrad,
    xlims=(minimum(xk⊥s) - dx/2, maximum(xk⊥s) + dx/2),
    ylims=(minimum(ykzs) - dy/2, maximum(ykzs) + dy/2),
    clims=(climmin, climmax), xticks=0:Int(round(maximum(xk⊥s))),
    xlabel=xlabel, ylabel=ylabel)
  return h
end

function getWavenumbers(sols, w0, k0)
  ωs = [sol.frequency for sol in sols]./w0
  nunstable = sum(imag(sol.frequency) > 0 for sol in sols)
  @info "There are $nunstable unstable solutions out of $(length(sols))"
  kzs = [para(sol.wavenumber) for sol in sols]./k0
  k⊥s = [perp(sol.wavenumber) for sol in sols]./k0
  xk⊥s = sort(unique(k⊥s))
  ykzs = sort(unique(kzs))

  ks = [abs(sol.wavenumber) for sol in sols]./k0
  kθs = atan.(k⊥s, kzs)
  extremaangles = collect(extrema(kθs))
  return ωs, kzs, k⊥s, xk⊥s, ykzs, ks, kθs, extremaangles
end

function relative(p, rx, ry)
  xlims = Plots.xlims(p)
  ylims = Plots.ylims(p)
  return xlims[1] + rx * (xlims[2]-xlims[1]), ylims[1] + ry * (ylims[2] - ylims[1])
end

@everywhere begin
  rmprocs(nprocsadded)
  const dir = dirname(pwd())*"/ICE_DS/JET26148/default_params_with_Triton_concentration"
  fontsize=9

  # electron mass and charge
  mₑ = LinearMaxwellVlasov.mₑ
  ze = -1

  const ξ = Float64(@fetchfrom 1 ximin)
  # concentrations and densities
  # Fig 18 Cottrell 1993
  n0 = Float64(@fetchfrom 1 _n0) #1.5e19# 5e19 # 1.7e19 # central electron density 3.6e19
  B0 = Float64(@fetchfrom 1 _B0) #3.7 #2.07 = 2.8T * 2.96 m / 4m
  # 2.23 T is 17MHz for deuterium cyclotron frequency

  # set secondary fuel concentration
  # ξ2 = 0.11
  m1 = md*mₑ
  m2 = mT*mₑ
  mmin = mα*mₑ
  z1 = 1
  z2 = 1
  zmin = 2
 
  const name_extension = parsedargs["nameextension"]
  const filecontents = [i for i in readlines(open(@__FILE__))]
  
  colorgrad = :haline #Plots.cgrad([:cyan, :blue, :darkblue, :midnightblue, :black, :darkred, :red, :orange, :yellow])
  clims = (-0.15,0.15)
  xi2arr = [0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9] #LinRange(0.45,0.95,11)
  harr = []
  # loop through each concentration
  for i = 1:length(xi2arr)
    xi2 = xi2arr[i] # parsedargs["secondfuelionconcentrationratio"]
    @show xi2
    # get densities
    n2 = xi2*n0
    nmin = ξ*n0
    n1 = (1/z1)*(n0-z2*n2-zmin*nmin) # 1 / (1.0 + 2*ξ)
    @assert n0 ≈ z1*n1 + z2*n2 + zmin*nmin
    # get Va
    Va = getVA(m1, m2, mmin, xi2, z1, z2, zmin)  
    # freqs
    Ωe = cyclotronfrequency(B0, mₑ, ze)
    Ω1 = cyclotronfrequency(B0, m1, z1)
    Ωmin = cyclotronfrequency(B0, mmin, zmin)
    Πe = plasmafrequency(n0, mₑ, ze)
    Π1 = plasmafrequency(n1, m1, z1)
    Πmin = plasmafrequency(nmin, mmin, zmin)
    if xi2 != 0
      Ω2 = cyclotronfrequency(B0, m2, z2)
      Π2 = plasmafrequency(n2, m2, z2)
    else
      Ω2 = 0
      Π2 = 0
    end
    w0 = abs(Ωmin)
    k0 = w0 / abs(Va)
    @show w0
      
    # get directory
    #dirpath = mapreduce(i->"_$(i[2])", *, parsedargs; init="$dir/ICE_DS/JET26148/default_params_with_Triton_concentration/run")
    dirpath = "$dir/run_2.07_$xi2"*"_0.01_-0.646_0.01_15.0_3.5__1.0_4.0_1.7e19_0.00015_1024"
    @show dirpath
    
    @load "$dirpath/solutions2D_.jld" filecontents plasmasols w0 k0
    sols = sort(plasmasols, by=s->imag(s.frequency))
    ωs, kzs, k⊥s, xk⊥s, ykzs, ks, kθs, extremaangles = getWavenumbers(sols, w0, k0)
    
    # plot
    zs = imag.(ωs)
    climmax = maximum(zs)
    xlabel = "\$\\mathrm{Perpendicular\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
    ylabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"

    climmin=minimum(zs[@. !ismissing(zs)])
    climmax=maximum(zs[@. !ismissing(zs)])
    zcolor = make2d(ykzs, xk⊥s, kzs, k⊥s, zs)
    dx = (xk⊥s[2] - xk⊥s[1]) / (length(xk⊥s) - 1)
    dy = (ykzs[2] - ykzs[1]) / (length(ykzs) - 1)
  
    h = Plots.heatmap(xk⊥s, ykzs, zcolor, clims=clims, 
        framestyle=:box, c=colorgrad, cbar=false)#,
      # framestyle=:box, c=colorgrad,
      # xlims=(minimum(xk⊥s) - dx/2, maximum(xk⊥s) + dx/2),
      # ylims=(minimum(ykzs) - dy/2, maximum(ykzs) + dy/2),
      # clims=(climmin, climmax), xticks=0:Int(round(maximum(xk⊥s))),
      # xlabel=xlabel, ylabel=ylabel)
    
    Plots.annotate!(h, [(relative(h, 0.02, 0.95)..., Plots.text("$xi2", fontsize, :black))])
    Plots.plot!(xlims=(11, 15),ylims=(-1,1),ticks=false)
    push!(harr, h) # append handle to handle array
    @show "here plotted"
  end
  h2 = scatter([0,0], [0,1], zcolor=[0,3], clims=clims, xlims=(1,1.1), 
                label="", c=colorgrad, colorbar_title="\$\\mathrm{Growth\\ Rate} \\ [\\Omega_{i}]\$", framestyle=:none)
  push!(harr, h2)
  @show harr
  @show size(harr)
  # layout of plot
  l = @layout [grid(2,5) a{0.035w}]
  Plots.plot(harr..., link=:x, margin=0.01Plots.mm, layout=l)#@layout [a c e g i; b d f h j])
  Plots.savefig("$dir/total_combined.png")#.pdf
end


println("Ending at ", now())


