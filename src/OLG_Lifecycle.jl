# OLG_Lifecycle.jl — Long‑lived (S=100) OLG model with endogenous labor supply
# ---------------------------------------------------------------------------
# Features
# - Agents live S periods (default S=100). Ages indexed 1..S.
# - Life cycle: at age s agent supplies labor l_s ∈ [0,1], with age-specific productivity e_s.
# - Productivity schedule e_s is chosen so that working ages cluster roughly between 20 and 65.
# - Agents hold assets a≥0 and choose next‑period assets a' and labor l each period.
# - Period utility: u(c) - χ * l^{1+η}/(1+η) where u(c)=c^{1-σ}/(1-σ).
# - No borrowing (a' ≥ 0). Assets pay market return r and wages are w per unit of effective labor.
# - Firm: Cobb–Douglas, competitive factor prices. Steady state computed by fixed point on k.
# - Solved by finite-horizon backward induction (dynamic programming) for households
#   given prices (r,w). Steady state found by iterating on k until aggregate capital
#   (sum of chosen assets of cohorts) equals k.
#
# Notes / caveats
# - The DP with S=100 and asset grid Na ~ 200 can be computationally heavy but is
#   fully explicit and dependency free. Set smaller S/Na for faster runs.
# - For pedagogical clarity the code focuses on steady state GE. You can extend to
#   full transition paths (TPI) but that will add complexity and runtime.

module OLG_Lifecycle

using Printf
using Statistics

export Params, SteadyState, steady_state, pretty_print, simulate_household

################################################################################
# 1) Parameters
################################################################################
Base.@kwdef mutable struct Params
    # production
    α::Float64 = 0.36
    A::Float64 = 1.0
    δ::Float64 = 0.08
    n::Float64 = 0.01
    # preferences
    β::Float64 = 0.96
    σ::Float64 = 2.0
    χ::Float64 = 1.0       # labor disutility scale
    η::Float64 = 1.0       # labor Frisch parameter: disutility ~ l^{1+η}/(1+η)
    # life cycle / numerical
    S::Int = 100          # number of ages
    Na::Int = 200         # asset grid points
    amax::Float64 = 50.0  # max asset (per effective worker)
    tol::Float64 = 1e-8
    maxit::Int = 5000
    θ_damp::Float64 = 0.6
    verbose::Bool = false
end

################################################################################
# 2) Productivity (age profile) — e_s
################################################################################
function age_productivity(S::Int; μ_age::Float64=42.0, σ_age::Float64=12.0)
    ages = collect(1:S)
    pdf = exp.(-0.5 .* ((ages .- μ_age)./σ_age).^2)
    pdf = pdf ./ maximum(pdf)
    return pdf
end

################################################################################
# 3) Technology and prices
################################################################################
prices(k::Float64, par::Params) = (par.α * par.A * k^(par.α - 1.0) - par.δ,
                                   (1.0 - par.α) * par.A * k^par.α)

################################################################################
# 4) Household finite-horizon DP solver (backward induction)
################################################################################
# utility
function u_cons(c, σ)
    if c <= 0.0
        return -1e20
    end
    return σ == 1.0 ? log(c) : (c^(1.0-σ))/(1.0-σ)
end
function u_lab(l, χ, η)
    return - χ * l^(1.0 + η) / (1.0 + η)
end

# intratemporal optimization: for given a,a', age s, find optimal l in [0,1]
function optimal_l_for_choice(a::Float64, a_prime::Float64, w::Float64, r::Float64, e_s::Float64, par::Params)
    # consumption implied with l variable: c(l) = w*e_s*l + (1+r)*a - a'
    if w * e_s == 0.0
        # no earnings possible, best to set l=0
        c0 = (1.0 + r)*a - a_prime
        if c0 <= 0.0
            return 0.0, -1e20, max(1e-12, c0)
        end
        return 0.0, u_cons(c0, par.σ) + u_lab(0.0, par.χ, par.η), c0
    end
    # minimal feasible l to ensure nonnegative consumption
    lmin = max(0.0, (a_prime - (1.0 + r)*a) / (w * e_s))
    if lmin > 1.0
        return 1.0, -1e20, 0.0
    end
    # objective in l
    function obj_l(l)
        c = w * e_s * l + (1.0 + r)*a - a_prime
        if c <= 0.0
            return -1e20
        end
        return u_cons(c, par.σ) + u_lab(l, par.χ, par.η)
    end
    # golden-section search on [lmin, 1]
    a_g, b_g = lmin, 1.0
    φ = (sqrt(5) - 1.0) / 2.0
    c_g = b_g - φ*(b_g - a_g)
    d_g = a_g + φ*(b_g - a_g)
    fc = obj_l(c_g); fd = obj_l(d_g)
    for _ in 1:40
        if fc > fd
            b_g = d_g; d_g = c_g; fd = fc; c_g = b_g - φ*(b_g - a_g); fc = obj_l(c_g)
        else
            a_g = c_g; c_g = d_g; fc = fd; d_g = a_g + φ*(b_g - a_g); fd = obj_l(d_g)
        end
    end
    l_star = (a_g + b_g)/2
    return l_star, obj_l(l_star), w * e_s * l_star + (1.0 + r)*a - a_prime
end

# DP solver
function solve_household_DP(r::Float64, w::Float64, par::Params, evec::Vector{Float64})
    S = par.S; Na = par.Na
    agrid = collect(range(0.0, par.amax; length=Na))
    V_next = zeros(Na)  # V_{S+1} = 0
    policy_ap = zeros(S, Na)
    policy_l = zeros(S, Na)

    for s in S:-1:1
        V = fill(-Inf, Na)
        for (i,a) in enumerate(agrid)
            best_val = -Inf; best_ap = 0.0; best_l = 0.0
            for (j,aprime) in enumerate(agrid)
                # compute intraperiod optimal l and utility
                l_star, util_intraperiod, c = optimal_l_for_choice(a, aprime, w, r, evec[s], par)
                if util_intraperiod <= -1e19
                    continue
                end
                # continuation value
                cont = (j <= length(V_next)) ? par.β * V_next[j] : 0.0
                val = util_intraperiod + cont
                if val > best_val
                    best_val = val; best_ap = aprime; best_l = l_star
                end
            end
            V[i] = best_val
            policy_ap[s,i] = best_ap
            policy_l[s,i] = best_l
        end
        V_next = copy(V)
        if par.verbose
            @info("Solved age", s)
        end
    end
    return agrid, policy_ap, policy_l
end

################################################################################
# 5) Aggregate capital from stationary individual policies
################################################################################
function aggregate_assets_from_policies(agrid, policy_ap, par::Params)
    S, Na = size(policy_ap)
    # assume cohort mass normalized to 1 at each generation (per-worker terms)
    # compute assets held by each age in steady state by simulating a cohort forward
    aset = zeros(S)
    # initial assets (age 1) = 0 for newborns
    a_now = 0.0
    for s in 1:S
        # find nearest grid index for a_now
        i = searchsortedfirst(agrid, a_now)
        i = clamp(i, 1, length(agrid))
        aprime = policy_ap[s,i]
        aset[s] = a_now
        a_now = aprime
    end
    # aggregate capital per worker: sum_{s=1}^{S-1} a_{s+1} / (1+n)^s? For stationary per-worker terms with population growth,
    # with normalization (one newborn each period), aggregate capital per worker equals sum_{s=1}^S a_s * ζ_s where ζ_s accounts
    # for population weights. For simplicity, assume stationary population so per‑worker capital = average asset of working cohort.
    # Here we take the assets of middle-aged workers as representative and compute mean asset across ages (approx).
    return mean(aset)
end

################################################################################
# 6) Steady State Solver (fixed point on k)
################################################################################
Base.@kwdef struct SteadyState
    k::Float64
    r::Float64
    w::Float64
    e::Vector{Float64}
    agrid::Vector{Float64}
    policy_ap::Array{Float64,2}
    policy_l::Array{Float64,2}
    Kagg::Float64
end

function steady_state(par::Params; k0::Float64=0.3)
    k = max(1e-6, k0)
    evec = age_productivity(par.S)
    for it in 1:par.maxit
        r, w = prices(k, par)
        agrid, policy_ap, policy_l = solve_household_DP(r, w, par, evec)
        Kagg = aggregate_assets_from_policies(agrid, policy_ap, par)
        knext = Kagg
        k_new = (1.0 - par.θ_damp) * k + par.θ_damp * knext
        if par.verbose && it%50==0
            @info("SS iter", it, k, k_new, abs(k_new-k))
        end
        if abs(k_new - k) < par.tol
            k = k_new
            r, w = prices(k, par)
            return SteadyState(k, r, w, evec, agrid, policy_ap, policy_l, Kagg)
        end
        k = k_new
    end
    r, w = prices(k, par)
    return SteadyState(k, r, w, evec, agrid, policy_ap, policy_l, Kagg)
end

################################################################################
# 7) Utilities / printing
################################################################################
function pretty_print(ss::SteadyState)
    println("
— S-period Steady State —")
    println(@sprintf("k* = %.6f", ss.k))
    println(@sprintf("r* = %.6f", ss.r))
    println(@sprintf("w* = %.6f", ss.w))
    println(@sprintf("Aggregate K (approx) = %.6f", ss.Kagg))
    println("Sample of age-productivity (first 10 ages):")
    println(join(ss.e[1:min(end,10)], ", "))
end

# helper to simulate a single household's path given starting asset a0
function simulate_household(a0::Float64, ss::SteadyState, par::Params)
    agrid = ss.agrid; policy_ap = ss.policy_ap; policy_l = ss.policy_l; evec = ss.e
    S = par.S
    a_path = zeros(S); l_path = zeros(S)
    a = a0
    for s in 1:S
        i = searchsortedfirst(agrid, a); i = clamp(i, 1, length(agrid))
        a_next = policy_ap[s,i]; l = policy_l[s,i]
        a_path[s] = a; l_path[s] = l
        a = a_next
    end
    return a_path, l_path
end

end # module OLG_Lifecycle

# Demo (uncomment to run):
# using .OLG_Lifecycle
# par = OLG_Lifecycle.Params(S=100, Na=200, amax=100.0, χ=2.0, η=1.0, σ=2.0)
# ss = OLG_Lifecycle.steady_state(par)
# OLG_Lifecycle.pretty_print(ss)
# a_path, l_path = OLG_Lifecycle.simulate_household(0.0, ss, par)
# println("Labor path (ages 1..100) sample: ", l_path[1:100])
