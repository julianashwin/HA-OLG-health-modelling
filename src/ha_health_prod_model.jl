if occursin("jashwin", pwd())
    cd("C://Users/jashwin/Documents/GitHub/HA-OLG-health-modelling/")
else
    cd("/Users/julianashwin/Documents/GitHub/HA-OLG-health-modelling/")
end


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
using Statistics, Random, Distributions
using Printf, Plots, FreqTables
using Interpolations, Roots
using Filesystem: mkpath

##########################
# 0. High-level choices
##########################
A_max = 100            # maximum age / lifespan (periods)
A_retire = 66
r = 0.03
β = 0.96
γ = 2.0
φ = 1.0
ψ_good = 1.0
ψ_bad = 2.0
w = 1.0

# asset grid
na = 500
a_min = 0.0
a_max = 200.0
a_grid = [a_min; collect(range(1e-6, 1.0, length=50)); collect(range(1.0,10.0,length=150)); collect(range(10.0,a_max,length=na-201))]
na = length(a_grid)

##########################
# 1. Binary Markov processes
##########################
π_z = [0.9 0.1; 0.1 0.9]
z_vals = [0.6, 1.4]

π_h = [0.95 0.05; 0.2 0.8]
ψ_h = [ψ_good, ψ_bad]

# Joint states: index -> (z_index, h_index)
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

# maps
z_of = [z_vals[st[1]] for st in states]   # productivity multiplier (float)
h_of = [st[2] for st in states]           # health index (1 or 2)

##########################
# 2. Mortality / survival schedule
##########################
A_gomp = 1e-4
B_gomp = 0.085
age_vec = collect(1:A_max)
mort_base = [A_gomp * exp(B_gomp * (age - 20)) for age in age_vec]
mort_base = clamp.(mort_base, 0.0, 0.999)

# health multiplier
mort_mult = [1.0, 3.0]
surv = Array{Float64}(undef, A_max, 2)
for a in 1:A_max
    for h in 1:2
        m = mort_base[a] * mort_mult[h]
        surv[a,h] = max(0.0, 1.0 - m)
    end
end

# small quick plot to inspect (optional)
# plot(mort_base); savefig("figures/ha_health_prod/mort_base.png")
# plot([surv[:,1] surv[:,2]], label=["healthy" "unhealthy"]); savefig("figures/ha_health_prod/surv.png")

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

function mup(c)
    return c^(-γ)
end

function optimal_l(a, ap, z, ψ; w=w)
    c_at_l(l) = (1 + r) * a + w * z * l - ap
    if c_at_l(0.0) <= 0 && c_at_l(1.0) <= 0
        return NaN
    end
    f(l) = begin
        c = c_at_l(l)
        if c <= 0
            return 1e6
        end
        return mup(c) * w * z - ψ * l^φ
    end
    f0 = f(0.0)
    f1 = f(1.0)
    if f0 < 0 && f1 < 0
        return 0.0
    elseif f0 > 0 && f1 > 0
        if c_at_l(1.0) > 0
            return 1.0
        else
            return max(0.0, min(1.0, (ap - (1+r)*a) / (w*z) ))
        end
    else
        root = find_zero(f, (0.0, 1.0), Bisection(), atol=1e-8)
        return clamp(root, 0.0, 1.0)
    end
end

##########################
# 4. Backward induction
##########################
V_next = zeros(s, na)
V_curr = similar(V_next)
policy_ap_idx = Array{Int}(undef, A_max, s, na)
policy_l = Array{Float64}(undef, A_max, s, na)

function interp_vec(xgrid, vvec, x)
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
            for iap in 1:na
                ap = a_grid[iap]
                lstar = optimal_l(a, ap, z, ψ)
                if isnan(lstar)
                    continue
                end
                c = (1 + r) * a + w * z * lstar - ap
                if c <= 0
                    continue
                end
                flow = u(c) - ψ * (lstar^(1+φ)) / (1+φ)
                if age == A_max
                    cont = 0.0
                else
                    cont = 0.0
                    for jstate in 1:s
                        Vnext_ap = interp_vec(a_grid, V_next[jstate, :], ap)
                        cont += Π[istate, jstate] * Vnext_ap
                    end
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
    V_next .= V_curr
end

@printf("Backward induction complete.\n")

##########################
# 5. Simulation (with health and productivity paths recorded)
##########################
Tsim = A_max
Nsim = 2000
rng = MersenneTwister(2025)

eigvals, eigvecs = eigen(Π')
stat = abs.(eigvecs[:, findall(isapprox.(eigvals, 1.0; atol=1e-8))[1]])
stat = stat / sum(stat)

state_dist = Categorical(vec(stat))
# state_sim contains joint-state indices 1..4
state_sim = rand(rng, state_dist, Nsim)

asset_sim = fill(a_grid[1], Nsim)
alive = trues(Nsim)                     # track who's alive

cons_sim  = fill(NaN, Nsim, Tsim)
lab_sim   = fill(NaN, Nsim, Tsim)
asset_path = fill(NaN, Nsim, Tsim)

# NEW: record health and productivity histories as indices (1 or 2)
health_sim = fill(0, Nsim, Tsim)   # int matrix (1 or 2) for health index
prod_sim   = fill(0, Nsim, Tsim)   # int matrix (1 or 2) for productivity index

age_death = fill(0, Nsim)

for t in 1:Tsim
    for i in 1:Nsim
        if !alive[i]
            continue
        end
        age = t
        ist = state_sim[i]
        zi, hi = states[ist]      # zi in {1,2}, hi in {1,2}
        zval = z_vals[zi]
        # record indices (not floats)
        health_sim[i,t] = hi
        prod_sim[i,t]   = zi

        ai = findmin(abs.(a_grid .- asset_sim[i]))[2]
        ap_idx = policy_ap_idx[age, ist, ai]
        ap = a_grid[ap_idx]
        l = policy_l[age, ist, ai]
        c = (1 + r) * asset_sim[i] + w * zval * l - ap

        cons_sim[i,t]  = c
        lab_sim[i,t]   = l
        asset_path[i,t] = asset_sim[i]

        if age == A_max
            alive[i] = false
            age_death[i] = age
            asset_sim[i] = NaN
        else
            next_state = rand(rng, Categorical(vec(Π[ist, :])))
            survprob = surv[age, hi]   # survival prob depends on current age & health index
            if rand(rng) > survprob
                alive[i] = false
                age_death[i] = age
                asset_sim[i] = NaN
                state_sim[i] = 0       # sentinel, will be ignored because alive=false
            else
                state_sim[i] = next_state
                asset_sim[i] = ap
            end
        end
    end
end

@printf("Simulation complete.\n")

##########################
# 6. Plots of key outputs (handle NaNs)
##########################

# helper safe mean that returns NaN if no valid observations
safemean(v) = begin
    vals = filter(!isnan, v)
    isempty(vals) ? NaN : mean(vals)
end

# create output dir if needed
outdir = "figures/ha_health_prod"
mkpath(outdir)

# (optional) frequency of final joint states among still-alive agents: ignore zeros/sentinels
alive_final_states = state_sim[state_sim .> 0]  # this includes last-drawn states, but may include dead -> sentinel removed
# If you want counts by state at a given age, better compute during simulation. For quick check:
try
    freq = freqtable(alive_final_states)
    @show freq
catch
    @warn "freqtable failed (maybe no nonzero states present)"
end

# Aggregate means
mean_assets = [safemean(asset_path[:,t]) for t in 1:Tsim]
mean_cons   = [safemean(cons_sim[:,t])   for t in 1:Tsim]
mean_lab    = [safemean(lab_sim[:,t])    for t in 1:Tsim]

p1 = plot(age_vec, mean_assets, xlabel="Age", ylabel="Assets", title="Mean Assets (All Agents)", legend=false)
p2 = plot(age_vec, mean_cons,   xlabel="Age", ylabel="Consumption", title="Mean Consumption (All Agents)", legend=false)
p3 = plot(age_vec, mean_lab,    xlabel="Age", ylabel="Labor", title="Mean Labor (All Agents)", legend=false)
plot(p1,p2,p3, layout=(3,1), size=(700,900))
savefig(joinpath(outdir, "lifecycle_profiles.pdf"))

# By health (good=1, bad=2)
colors = [:blue, :red]
labels = ["Healthy","Unhealthy"]
ph1 = plot(title="Assets by Health", xlabel="Age", ylabel="Assets")
ph2 = plot(title="Consumption by Health", xlabel="Age", ylabel="Consumption")
ph3 = plot(title="Labor by Health", xlabel="Age", ylabel="Labor")
for h in 1:2
    aset = [ safemean(asset_path[health_sim[:,t] .== h, t]) for t in 1:Tsim ]
    cons = [ safemean(cons_sim[health_sim[:,t] .== h, t])   for t in 1:Tsim ]
    lab  = [ safemean(lab_sim[health_sim[:,t] .== h, t])    for t in 1:Tsim ]
    plot!(ph1, age_vec, aset, label=labels[h], color=colors[h])
    plot!(ph2, age_vec, cons, label=labels[h], color=colors[h])
    plot!(ph3, age_vec, lab,  label=labels[h], color=colors[h])
end
plot(ph1,ph2,ph3, layout=(3,1), size=(700,900))
savefig(joinpath(outdir, "profiles_by_health.png"))

# By productivity (1=low, 2=high)
labels_prod = ["Low Prod","High Prod"]
pp1 = plot(title="Assets by Productivity", xlabel="Age", ylabel="Assets")
pp2 = plot(title="Consumption by Productivity", xlabel="Age", ylabel="Consumption")
pp3 = plot(title="Labor by Productivity", xlabel="Age", ylabel="Labor")
for j in 1:2
    aset = [ safemean(asset_path[prod_sim[:,t] .== j, t]) for t in 1:Tsim ]
    cons = [ safemean(cons_sim[prod_sim[:,t] .== j, t])   for t in 1:Tsim ]
    lab  = [ safemean(lab_sim[prod_sim[:,t] .== j, t])    for t in 1:Tsim ]
    plot!(pp1, age_vec, aset, label=labels_prod[j])
    plot!(pp2, age_vec, cons, label=labels_prod[j])
    plot!(pp3, age_vec, lab,  label=labels_prod[j])
end
plot(pp1,pp2,pp3, layout=(3,1), size=(700,900))
savefig(joinpath(outdir, "profiles_by_productivity.png"))

@printf("Plots saved in %s\n", outdir)
# End of file
