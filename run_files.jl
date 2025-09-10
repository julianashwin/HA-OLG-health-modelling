if occursin("jashwin", pwd())
    cd("C://Users/jashwin/Documents/GitHub/HA-OLG-health-modelling/")
else
    cd("/Users/julianashwin/Documents/GitHub/HA-OLG-health-modelling/")
end

using Plots

include("src/OLG_Edu.jl")
using .OLG_Edu

par = OLG_Edu.Params(phi1=0.3, edu_rate=0.04)
ss = OLG_Edu.steady_state(par)
OLG_Edu.pretty_print(ss)

tp = OLG_Edu.transition_path(0.15, 80, par)
OLG_Edu.pretty_print(tp)_Edu






include("src/OLG_Lifecycle.jl")
using .OLG_Lifecycle

par = OLG_Lifecycle.Params(S=100, Na=200, amax=80.0, χ=2.0, η=1.0, σ=2.0, β=0.96)
ss = OLG_Lifecycle.steady_state(par)
OLG_Lifecycle.pretty_print(ss)

# simulate one household from a0=0
a_path, l_path = OLG_Lifecycle.simulate_household(0.0, ss, par)
println("Labor supply ages 1..100 (sample):")
plot(l_path)
