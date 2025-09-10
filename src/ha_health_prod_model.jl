# Heterogeneous-agents model with binary health and productivity
# Filename: ha_health_prod_model.jl
# - Agents live up to A_max periods (ages 1..A_max)
# - Two binary Markov states: productivity (z in {low, high}) and health (h in {good, bad})
# - Productivity affects labor income via wage * z * labor
# - Health affects disutility of labor and mortality probability by age
# - Interest rate r is exogenous
# - Agents choose assets a' (next period) and labor l each period
# - Labor is chosen intratemporally from first-order condition given consumption
# - No borrowing (a' >= a_min). No explicit bequest motive; utility after death = 0
# - Solve by backward induction (finite horizon) with interpolation in asset grid

using LinearAlgebra
using Statistics
using Random
using Distributions
using Printf
using Roots
using Plots 

try
    using Interpolations
catch
    @warn "Install Interpolations.jl for faster interpolation (optional)."
end

##########################
# 0. High-level choices
##########################
A_max = 100            # maximum age / lifespan (periods)
A_retire = 66          # optional: retirement age (not used here)
r = 0.03               # exogenous interest rate
β = 0.96               # discount factor
γ = 2.0                # CRRA for consumption
φ = 1.0                # labor disutility curvature: disutil ~ l^{1+φ}/(1+φ)
ψ_good = 1.0           # scale of disutility when healthy
ψ_bad = 2.0            # scale when unhealthy (higher disutility)
w = 1.0                # wage (can be age-dependent later)

# asset grid
na = 500
a_min = 0.0
a_max = 200.0
# use denser grid near zero
a_grid = [a_min; collect(range(1e-6, 1.0, length=50)); collect(range(1.0,10.0,length=150)); collect(range(10.0,a_max,length=na-201))]
na = length(a_grid)

##########################
# 1. Binary Markov processes
##########################
# Productivity: low(1), high(2)
π_z = [0.9 0.1; 0.1 0.9]    # rows: from-state, cols: to-state
z_vals = [0.6, 1.4]         # productivity multipliers

# Health: good(1), bad(2)
π_h = [0.95 0.05; 0.2 0.8]
# disutility scale by health
ψ_h = [ψ_good, ψ_bad]

# Joint transition for (z,h) assuming independence between processes
# indices: s = 1..4 mapping (z,h): (1,1)->1, (2,1)->2, (1,2)->3, (2,2)->4
states = [(1,1),(2,1),(1,2),(2,2)]
s = length(states)
Π = zeros(s,s)
for i in 1:s
    (zi,hi) = states[i]
    for j in 1:s
        (zj,hj) = states[j]
        Π[i,j] = π_z[zi,zj] * π_h[hi,hj]
    end
end
# map state -> (z, h) values
z_of = [z_vals[st[1]] for st in states]
h_of = [st[2] for st in states]

##########################
# 2. Mortality / survival schedule by age and health
##########################
# We'll use a simple Gompertz-like hazard: m(a) = A * exp(B * (a-20))
# Then scale by health: bad health multiplies mortality.
A_gomp = 1e-5
B_gomp = 0.085
age_vec = collect(1:A_max)
mort_base = [A_gomp * exp(B_gomp * (age - 20)) for age in age_vec]
# ensure probabilities in (0,1)
mort_base = clamp.(mort_base, 0.0, 0.999)

# health multiplier
mort_mult = [1.0, 3.0]   # bad health triples mortality
# survival probability s(age, health) = 1 - mortality
surv = Array{Float64}(undef, A_max, 2)
for a in 1:A_max
    for h in 1:2
        m = mort_base[a] * mort_mult[h]
        surv[a,h] = max(0.0, 1.0 - m)
    end
end

##########################
# 3. Utility and intratemporal labor FOC
##########################
function u(c)
    if c <= 0
        return -1e20
    end
    if isapprox(γ,1.0; atol=1e-12)
        return log(c)
    else
        return (c^(1-γ) - 1) / (1 - γ)
    end
end

function mup(c)  # marginal utility
    return c^(-γ)
end

# Given (a, a', z, h), find optimal labor l in [0,1] solving FOC
# FOC: mup(c) * w * z = ψ_h * l^φ
# where c = (1+r)*a + w*z*l - a'
function optimal_l(a, ap, z, ψ; w=w)
    # If consumption nonpositive for all l in [0,1], infeasible -> return NaN
    # Define function f(l) = mup(c(l)) * w*z - ψ * l^φ
    c_at_l(l) = (1 + r) * a + w * z * l - ap
    if c_at_l(0.0) <= 0 && c_at_l(1.0) <= 0
        return NaN
    end
    f(l) = begin
        c = c_at_l(l)
        if c <= 0
            return 1e6  # large positive so root finder won't pick infeasible region
        end
        return mup(c) * w * z - ψ * l^φ
    end
    # check sign at endpoints
    f0 = f(0.0)
    f1 = f(1.0)
    if f0 < 0 && f1 < 0
        # marginal benefit < marginal cost everywhere => corner at l=0
        return 0.0
    elseif f0 > 0 && f1 > 0
        # marginal benefit > marginal cost everywhere => corner at l=1 (if feasible)
        if c_at_l(1.0) > 0
            return 1.0
        else
            # infeasible at l=1; pick highest feasible l with positive c
            # try bisection on l to find feasible region where c>0
            return max(0.0, min(1.0, (ap - (1+r)*a) / (w*z) ))
        end
    else
        # root exists between 0 and 1
        root = find_zero(f, (0.0, 1.0), Bisection(), atol=1e-8)
        return clamp(root, 0.0, 1.0)
    end
end

##########################
# 4. Backward induction: V[a_ind, state_index, age]
##########################
# For memory, we keep only V_next and V_curr; store policy functions for all ages
V_next = zeros(s, na)
V_curr = similar(V_next)
policy_ap_idx = Array{Int}(undef, A_max, s, na)
policy_l = Array{Float64}(undef, A_max, s, na)

# interpolation function helper using simple linear interpolation
function interp_vec(xgrid, vvec, x)
    # assume x within [xgrid[1], xgrid[end]]
    if x <= xgrid[1]
        return vvec[1]
    elseif x >= xgrid[end]
        return vvec[end]
    end
    i = searchsortedfirst(xgrid, x)
    if xgrid[i] == x
        return vvec[i]
    end
    i0 = i-1
    w = (x - xgrid[i0]) / (xgrid[i] - xgrid[i0])
    return (1-w)*vvec[i0] + w*vvec[i]
end

@printf("Starting backward induction for A_max=%d, na=%d, nstates=%d\n", A_max, na, s)
for age in A_max:-1:1
    @printf(" Solving age %3d\n", age)
    for istate in 1:s
        z = z_of[istate]
        h_ind = h_of[istate]
        ψ = ψ_h[h_ind]
        survival = surv[age, h_ind]
        for ia in 1:na
            a = a_grid[ia]
            best_val = -1e20
            best_idx = 1
            best_l = 0.0
            # choose a' on grid
            for iap in 1:na
                ap = a_grid[iap]
                # compute optimal labor for this (a, ap, z, ψ)
                lstar = optimal_l(a, ap, z, ψ)
                if isnan(lstar)
                    continue
                end
                c = (1 + r) * a + w * z * lstar - ap
                if c <= 0
                    continue
                end
                flow = u(c) - ψ * (lstar^(1+φ)) / (1+φ)
                # expected continuation value across next states
                if age == A_max
                    cont = 0.0
                else
                    cont = 0.0
                    for jstate in 1:s
                        # interpolate V_next[jstate, :] at ap
                        Vnext_ap = interp_vec(a_grid, V_next[jstate, :], ap)
                        cont += Π[istate, jstate] * Vnext_ap
                    end
                    # mortality: if die, value is 0; if survive, weight continuation by survival prob
                    cont *= survival
                end
                val = flow + β * cont
                if val > best_val
                    best_val = val
                    best_idx = iap
                    best_l = lstar
                end
            end
            V_curr[istate, ia] = best_val
            policy_ap_idx[age, istate, ia] = best_idx
            policy_l[age, istate, ia] = best_l
        end
    end
    # roll forward
    V_next .= V_curr
end

@printf("Backward induction complete.\n")

##########################
# 5. Simulate a cohort forward (micro-simulation)
##########################
Tsim = A_max
Nsim = 20000
rng = MersenneTwister(2025)
# initialize agents at age 1 with zero assets, draw initial states from stationary distribution
# compute stationary distribution of joint Markov
eigvals, eigvecs = eigen(Π')
stat = abs.(eigvecs[:, findall(isapprox.(eigvals, 1.0; atol=1e-8))[1]])
stat = stat / sum(stat)

# sample initial states
state_dist = Categorical(vec(stat))
state_sim = rand(rng, state_dist, Nsim)
age_sim = ones(Int, Nsim)
asset_sim = fill(a_grid[1], Nsim)  # start at lowest asset
cons_sim = zeros(Nsim, Tsim)
lab_sim = zeros(Nsim, Tsim)
asset_path = zeros(Nsim, Tsim)

for t in 1:Tsim
    for i in 1:Nsim
        age = t
        ist = state_sim[i]
        # find current asset index (nearest)
        ai = findmin(abs.(a_grid .- asset_sim[i]))[2]
        ap_idx = policy_ap_idx[age, ist, ai]
        ap = a_grid[ap_idx]
        l = policy_l[age, ist, ai]
        c = (1 + r) * asset_sim[i] + w * z_of[ist] * l - ap
        cons_sim[i, t] = c
        lab_sim[i, t] = l
        asset_path[i, t] = asset_sim[i]
        # draw next state
        # If at final age, they die; else transition according to Π and mortality
        if age == A_max
            # terminal: set next asset to NaN
            asset_sim[i] = NaN
            state_sim[i] = ist
        else
            # draw next joint state
            state_sim[i] = rand(rng, Categorical(vec(Π[ist, :])))
            # apply survival: if die, mark as dead and stop tracking
            h_ind = h_of[state_sim[i]]
            survprob = surv[age, h_ind]
            if rand(rng) > survprob
                # agent dies: we fill remaining with NaN and remove from sim
                # for simplicity, set asset to NaN
                asset_sim[i] = NaN
                state_sim[i] = 0
            else
                asset_sim[i] = ap
            end
        end
    end
end

@printf("Simulation complete (N=%d, T=%d). Sample means at midlife (age=45):\n", Nsim, Tsim)
mid = min(Tsim, 45)
alive = .!isnan.(asset_path[:, mid])
@printf(" Mean assets: %.3f, Mean consumption: %.3f, Mean labor: %.3f (at age %d)\n", mean(asset_path[alive, mid]), mean(cons_sim[alive, mid]), mean(lab_sim[alive, mid]), mid)

##########################
# 6. Outputs saved or available in memory
# - policy_ap_idx[age, state, a_index]
# - policy_l[age, state, a_index]
# - V_curr/V_next final values for age=1 value function
# - simulation arrays: cons_sim, lab_sim, asset_path
##########################

# End of file
