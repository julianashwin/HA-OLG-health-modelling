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
using Distributed, SharedArrays

# Check if we need to add processes
if nprocs() == 1
    addprocs(4)  # Add 4 worker processes
end

@everywhere using LinearAlgebra, Statistics, SharedArrays, Interpolations, Roots

function check_everywhere(var::Symbol)
    results = Dict()
    for p in workers() ∪ [myid()]
        results[p] = @fetchfrom p isdefined(Main, var)
    end
    return results
end


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

# asset grid: finer grid closer to zero
function create_asset_grid(a_min, a_max, na_target)
    # More careful construction to avoid overlaps
    # Dense near zero, medium density in middle, coarse at high values
    n1, n2, n3 = 50, 150, na_target - 50 - 150
    # Segment 1: a_min to 1.0
    seg1 = range(a_min, 1.0, length=n1+1)[1:end-1]  # exclude endpoint
    # Segment 2: 1.0 to 10.0  
    seg2 = range(1.0, 10.0, length=n2+1)[1:end-1]   # exclude endpoint
    # Segment 3: 10.0 to a_max
    seg3 = range(10.0, a_max, length=n3)          # include endpoint

    return [collect(seg1); collect(seg2); collect(seg3)]
end

na = 500
a_min = 0.0
a_max = 200.0
a_grid = create_asset_grid(a_min, a_max, na)
@assert na == length(a_grid) "Asset grid is the wrong length"
@assert length(a_grid) == length(unique(a_grid)) "Asset grid has duplicates!"


# Helper function to send variables to workers
function sendto(processes; kwargs...)
    for p in processes
        for (var, val) in kwargs
            @spawnat p eval(:($(Symbol(var)) = $(val)))
        end
    end
end

# Send the parameter values
sendto(workers(), r=r)
sendto(workers(), β=β) 
sendto(workers(), γ=γ)
sendto(workers(), φ=φ)
sendto(workers(), w=w)
sendto(workers(), a_grid=a_grid)



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


# Send the maps and distributions
sendto(workers(), z_of=z_of)
sendto(workers(), h_of=h_of) 
sendto(workers(), ψ_h=ψ_h) 
sendto(workers(), Π=Π) 
sendto(workers(), s=s) 


##########################
# 2. Mortality / survival schedule by age and health (Gompertz--Makeham)
##########################
# Gompertz--Makeham hazard: m(a) = C + A * exp(B * (a - offset))
C_makeham = 0.0005       # Makeham constant (age-independent baseline)
B_gomp    = 0.0966        # Gompertz slope (kept from prior)
A_gomp   =  0.3319  # calibrated Gompertz scale

age_vec = collect(1:A_max)
mort_base = [C_makeham + A_gomp * exp(B_gomp * (age - 97.6)) for age in age_vec]
# keep hazards in (0,1)
mort_base = clamp.(mort_base, 0.0, 0.999)



# compute implied life expectancy (discrete ages, death probability at each age)
surv_uncond = ones(Float64, A_max)
for t in 2:A_max
    surv_uncond[t] = surv_uncond[t-1] * (1.0 - mort_base[t-1])
end
prob_die = similar(mort_base)
for t in 1:A_max
    prob_die[t] = surv_uncond[t] * mort_base[t]
end
# survivors beyond last age are treated as dying at age A (consistent with simulation)
exp_age = sum((1:A_max) .* prob_die) + surv_uncond[end] * A_max
@printf("Implied life expectancy (ages 1..%d, terminal death at %d): %.4f\n", A_max, A_max, exp_age)

plot(mort_base)
plot!(surv_uncond)

# health multiplier (unchanged)
mort_mult = [1.0, 3.0]   # bad health triples mortality

# survival probability s(age, health) = 1 - mortality
surv = max.(0.0, 1.0 .- mort_base .* (mort_mult')) 
@everywhere surv = $surv

# Mortality plots (uncomment to view)
# plot(age_vec, mort_base, xlabel="Age", ylabel="Hazard m(a)", title="Gompertz-Makeham hazard")
# plot(age_vec, surv[:,1], label="Healthy"); plot!(age_vec, surv[:,2], label="Unhealthy", xlabel="Age", ylabel="Survival prob")


##########################
# 3. Utility and intratemporal labor FOC
##########################
@everywhere function u(c)
    if c <= 0
        return -1e20
    end
    if isapprox(γ,1.0; atol=1e-12)
        return log(c)
    else
        return (c^(1-γ) - 1) / (1 - γ)
    end
end

@everywhere function mup(c)
    return c^(-γ)
end

@everywhere function optimal_l(a, ap, z, ψ; w=w)
    c_at_l(l) = (1 + r) * a + w * z * l - ap

    # Add bounds checking
    @assert 0 ≤ a "Negative assets not allowed"
    @assert 0 ≤ ap "Negative next-period assets not allowed"
    
    if c_at_l(0.0) ≤ 0 && c_at_l(1.0) ≤ 0
        return 0.0  # Return 0 instead of NaN
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
@everywhere function update_interpolators!(V_interpolators, a_grid, V_next)
    for j in 1:s
        V_interpolators[j] = LinearInterpolation(a_grid, V_next[j,:], 
                                                extrapolation_bc=Line())
    end
end


# Initialise value and policy functions
V_next = SharedArray{Float64}(s, na)
V_curr = SharedArray{Float64}(s, na) 
policy_ap_idx = SharedArray{Int}(A_max, s, na)
policy_l = SharedArray{Float64}(A_max, s, na)

# Initialize with zeros
fill!(V_next, 0.0)
fill!(V_curr, 0.0)


# Pre-create interpolation objects for each state
V_interpolators = Vector{Any}(undef, s)
# Initialize with zeros for the first iteration
update_interpolators!(V_interpolators, a_grid, V_next)
@everywhere V_interpolators = $V_interpolators


check_everywhere(:V_interpolators)


@printf("Starting backward induction for A_max=%d, na=%d, nstates=%d\n", A_max, na, s)
@printf("Running on %d processes\n", nprocs())
for age in A_max:-1:1
    @printf(" Solving age %3d\n", age)    
    # Send current age to all workers
    sendto(workers(), current_age=age)
    
    @distributed for istate in 1:s
        z = z_of[istate]
        h_ind = h_of[istate]
        ψ = ψ_h[h_ind]
        survival = surv[age, h_ind]

        for ia in 1:length(a_grid)
            a = a_grid[ia]
            best_val = -1e20
            best_idx = 1
            best_l = 0.0

            for iap in 1:length(a_grid)
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
                if current_age == A_max
                    cont = 0.0
                else
                    cont = 0.0
                    for jstate in 1:s
                        # Use the pre-built interpolator
                        Vnext_ap = V_interpolators[jstate](ap)
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

    # Synchronize: wait for all workers to finish
    @sync @distributed for i in 1:s
        nothing  # Just to ensure synchronization
    end
    
    # Update for next iteration
    V_next .= V_curr
    # UPDATE THE INTERPOLATORS with new V_next values
    update_interpolators!(V_interpolators, a_grid, V_next)
    @everywhere V_interpolators = $V_interpolators
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

cons_sim = Matrix{Float64}(undef, Nsim, Tsim)
lab_sim = Matrix{Float64}(undef, Nsim, Tsim)
health_sim = Matrix{Int8}(undef, Nsim, Tsim)  # Int8 sufficient for 1/2 values
prod_sim = Matrix{Int8}(undef, Nsim, Tsim)

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
