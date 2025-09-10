# OLG_Edu.jl — 3‑Period OLG with Education Loans and Productivity
# -----------------------------------------------------------------
# Features
# - Three-period agents: Education (period 1), Work (period 2), Retirement (period 3)
# - Education is financed by a loan L taken in period 1 and repaid in period 2
# - Education increases labor productivity linearly: phi(L) = 1 + phi1 * L
# - Households choose L ≥ 0 and savings s ≥ 0 when working to maximize utility
# - CRRA utility; Cobb–Douglas production; competitive factor prices
# - Solves steady state and transition path (TPI) — dependency free
#
# Usage
#   include("OLG_Edu.jl")
#   using .OLG_Edu
#   par = OLG_Edu.Params()
#   ss  = OLG_Edu.steady_state(par)
#   OLG_Edu.pretty_print(ss)
#   tp  = OLG_Edu.transition_path(0.2, 80, par)
#   OLG_Edu.pretty_print(tp)


module OLG_Edu

using Printf
using Statistics

export Params, SteadyState, TransitionPath,
       prices, household_choices, steady_state, transition_path,
       pretty_print

################################################################################
# 1) Parameters
################################################################################
Base.@kwdef mutable struct Params
    α::Float64 = 0.36        # capital share
    β::Float64 = 0.96        # discount factor
    σ::Float64 = 2.0         # CRRA
    δ::Float64 = 0.08        # depreciation
    n::Float64 = 0.01        # pop growth
    A::Float64 = 1.0         # TFP
    phi1::Float64 = 0.5      # productivity return per unit of education (linear)
    edu_rate::Float64 = 0.04 # interest rate on education loan (can be market r or exogenous)
    θ_damp::Float64 = 0.5    # damping
    maxit::Int = 20000
    tol::Float64 = 1e-10
    verbose::Bool = false
end

################################################################################
# 2) Technology and Prices
################################################################################
""" prices(k, par) -> (r, w): interest rate (net of depreciation) and wage per worker """
function prices(k::Float64, par::Params)
    @assert k > 0.0 "k must be positive"
    r = par.α * par.A * k^(par.α - 1.0) - par.δ
    w = (1.0 - par.α) * par.A * k^par.α
    return r, w
end

################################################################################
# 3) Household problem with Education Loan
################################################################################
# Timeline for an individual born at t:
#  t (period 1): takes loan L >=0, consumes c1 = L
#  t+1 (period 2): works, receives wage w * phi(L), repays (1+edu_rate)*L, chooses s>=0
#      c2 = w*phi(L) - (1+edu_rate)*L - s
#  t+2 (period 3): consumes c3 = (1+r_{t+2}) * s
# Utility: u(c1) + β u(c2) + β^2 u(c3)

# Closed form for optimal s given L and r_{t+2} (same manipulations as in 2-period):
#   s = A / (1 + χ)  where A = w^edu - (1+edu_rate)*L  and χ = β^{-1/σ} (1+r_{t+2})^{1-1/σ}

"""
    household_choices(w, r_next, par) -> (L_opt, s_opt, c1, c2, c3)

Given contemporary wage per worker w and expected next-period interest rate r_next
(which affects retirement consumption), solve for optimal education loan L and
savings s. We assume φ(L) = 1 + phi1 * L.

This function performs a 1D numeric maximization over L (bounded search) and
uses the closed‑form for s conditional on L.
"""
function household_choices(w::Float64, r_next::Float64, par::Params)
    # utility function
    σ = par.σ
    u(c) = σ == 1.0 ? log(c) : (c^(1.0-σ))/(1.0-σ)

    # helper: given L compute A = w^edu - (1+edu_rate)*L and then s* and consumptions
    function eval_for_L(L)
        # productivity multiplier
        phi = 1.0 + par.phi1 * L
        wedu = phi * w
        A = wedu - (1.0 + par.edu_rate) * L
        if A <= 0.0
            # infeasible: cannot repay loan and save nonnegatively; return very low utility
            return -Inf, 0.0, 0.0, -Inf, -Inf
        end
        χ = par.β^(-1.0/par.σ) * (1.0 + r_next)^(1.0 - 1.0/par.σ)
        s = A / (1.0 + χ)
        c1 = L
        c2 = A - s
        c3 = (1.0 + r_next) * s
        if c2 <= 0.0 || c3 <= 0.0
            return -Inf, s, c1, c2, c3
        end
        util = u(c1) + par.β * u(c2) + par.β^2 * u(c3)
        return util, s, c1, c2, c3
    end

    # Search bounds for L: start with 0..Lmax. Choose Lmax so that phi effect plausibly bounded.
    # A simple heuristic: Lmax = min( max(10.0, 5*w), 100.0 )
    Lmax = max(10.0, 5.0*w)
    Ngrid = 400
    best_util = -Inf
    best = (0.0, 0.0, 0.0, 0.0, 0.0)
    for i in 0:Ngrid
        L = (i / Ngrid) * Lmax
        util, s, c1, c2, c3 = eval_for_L(L)
        if util > best_util
            best_util = util
            best = (L, s, c1, c2, c3)
        end
    end

    # simple local search (refinement) around best L using golden-section on small interval
    Lb = max(0.0, best[1] - Lmax/Ngrid)
    Ub = min(Lmax, best[1] + Lmax/Ngrid)
    φ_gs = (sqrt(5)-1)/2
    a, b = Lb, Ub
    c = b - φ_gs*(b-a)
    d = a + φ_gs*(b-a)
    util_c, _, _, _, _ = eval_for_L(c)
    util_d, _, _, _, _ = eval_for_L(d)
    iter_gs = 50
    for _ in 1:iter_gs
        if util_c > util_d
            b = d; d = c; util_d = util_c; c = b - φ_gs*(b-a); util_c, _, _, _, _ = eval_for_L(c)
        else
            a = c; c = d; util_c = util_d; d = a + φ_gs*(b-a); util_d, _, _, _, _ = eval_for_L(d)
        end
    end
    L_star = (a + b)/2
    util, s, c1, c2, c3 = eval_for_L(L_star)
    if util == -Inf
        # fallback to zero education
        util, s, c1, c2, c3 = eval_for_L(0.0)
        L_star = 0.0
    end
    return L_star, s, c1, c2, c3
end

################################################################################
# 4) Aggregation and Law of Motion
################################################################################
# Each generation chooses (L, s) when working (period 2). Aggregate capital next
# period equals aggregate savings of the working generation divided by (1+n).
# In per-worker terms (k), with a continuum of identical agents, k_{t+1} = s_t/(1+n)

function law_of_motion_from_k(k::Float64, par::Params)
    r, w = prices(k, par)
    # r_next we use r as a guess for r_{t+2} in stationary mapping (steady state uses same r)
    L, s, _, _, _ = household_choices(w, r, par)
    return s / (1.0 + par.n)
end

################################################################################
# 5) Steady State Solver
################################################################################
Base.@kwdef struct SteadyState
    k::Float64
    r::Float64
    w::Float64
    L::Float64
    s::Float64
    c1::Float64
    c2::Float64
    c3::Float64
    y::Float64
end

function steady_state(par::Params; k0::Float64=0.3)
    k = max(k0, 1e-8)
    for it in 1:par.maxit
        knext = law_of_motion_from_k(k, par)
        k_new = (1.0 - par.θ_damp)*k + par.θ_damp*knext
        if abs(k_new - k) < par.tol
            k = k_new
            break
        end
        k = k_new
    end
    r, w = prices(k, par)
    L, s, c1, c2, c3 = household_choices(w, r, par)
    y = par.A * k^par.α
    return SteadyState(k, r, w, L, s, c1, c2, c3, y)
end

################################################################################
# 6) Transition Path Iteration (TPI)
################################################################################
Base.@kwdef struct TransitionPath
    k::Vector{Float64}
    r::Vector{Float64}
    w::Vector{Float64}
    L::Vector{Float64}
    s::Vector{Float64}
end

function transition_path(K0::Float64, T::Int, par::Params; θ::Float64=0.5)
    @assert T >= 3 "T must be at least 3 for 3-period agents"
    ss = steady_state(par)

    # initial guess linear from K0 to k*
    k = [ (1.0 - t/T)*K0 + (t/T)*ss.k for t in 0:T ]

    for it in 1:par.maxit
        # prices along path
        rpath = Float64[]; wpath = Float64[]
        for t in 0:T
            r_t, w_t = prices(k[t+1], par)
            push!(rpath, r_t); push!(wpath, w_t)
        end
        # compute choices: at time t the working generation faces wage w_t and expects r_{t+1}
        Lseq = zeros(T+1); sseq = zeros(T+1)
        for t in 0:T-1  # t = 0..T-1 (we don't need to choose for terminal generation beyond T-1)
            r_next = (t+2 <= T) ? rpath[t+2] : ss.r  # r_{t+2} expectation; pin to SS at end
            Lopt, sopt, _, _, _ = household_choices(wpath[t+1], r_next, par)
            Lseq[t+1] = Lopt
            sseq[t+1] = sopt
        end
        # update k: k_{t+1} = s_t / (1+n)
        knew = copy(k)
        for t in 0:T-1
            knew[t+2] = sseq[t+1] / (1.0 + par.n)
        end
        knew[end] = ss.k

        # damped update (keep k[1]=K0 fixed)
        maxdiff = 0.0
        for idx in 2:length(k)
            k_old = k[idx]
            k[idx] = (1.0 - θ)*k[idx] + θ*knew[idx]
            maxdiff = max(maxdiff, abs(k[idx] - k_old))
        end
        if par.verbose && it%200==0
            @info "TPI iter" it maxdiff
        end
        if maxdiff < par.tol
            break
        end
    end

    # final recomputation of prices and choices
    rpath = Float64[]; wpath = Float64[]; Lseq = zeros(T+1); sseq = zeros(T+1)
    for t in 0:T
        r_t, w_t = prices(k[t+1], par)
        push!(rpath, r_t); push!(wpath, w_t)
    end
    for t in 0:T-1
        r_next = (t+2 <= T) ? rpath[t+2] : rpath[end]
        Lopt, sopt, _, _, _ = household_choices(wpath[t+1], r_next, par)
        Lseq[t+1] = Lopt
        sseq[t+1] = sopt
    end

    return TransitionPath(k, rpath, wpath, Lseq, sseq)
end

################################################################################
# 7) Utilities / Printing
################################################################################
function pretty_print(ss::SteadyState)
    println("
— 3-Period Steady State —")
    println(@sprintf("k* = %.6f", ss.k))
    println(@sprintf("r* = %.6f", ss.r))
    println(@sprintf("w* = %.6f", ss.w))
    println(@sprintf("y* = %.6f", ss.y))
    println(@sprintf("L* = %.6f", ss.L))
    println(@sprintf("s* = %.6f", ss.s))
    println(@sprintf("c1* = %.6f", ss.c1))
    println(@sprintf("c2* = %.6f", ss.c2))
    println(@sprintf("c3* = %.6f", ss.c3))
end

function pretty_print(tp::TransitionPath)
    T = length(tp.k) - 1
    println("
— Transition Path — (t = 0..$T)")
    println(@sprintf("k0=%.6f  kT=%.6f", tp.k[1], tp.k[end]))
    println(@sprintf("First 5 k: %s", join(@view(tp.k[1:min(end,5)]), ", ")))
    println(@sprintf("First 5 L: %s", join(@view(tp.L[1:min(end,5)]), ", ")))
    println(@sprintf("First 5 s: %s", join(@view(tp.s[1:min(end,5)]), ", ")))
end

end # module OLG_Edu

# Quick demo (uncomment to run):
# using .OLG_Edu
# par = OLG_Edu.Params(phi1=0.3, edu_rate=0.04)
# ss = OLG_Edu.steady_state(par)
# OLG_Edu.pretty_print(ss)
# tp = OLG_Edu.transition_path(0.15, 60, par)
# OLG_Edu.pretty_print(tp)
