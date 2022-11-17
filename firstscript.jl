using DifferentialEquations
using OrdinaryDiffEq
using Noise
using Plots

function msd(ddu,du,u,p,t)
    ddu[1]= -u[1]*p[1]-du[1]*p[2]
end

function spring()
    initialvel=[0]
    initialdisp=[9]
    tspan = (0.0,8)
    p=[1,0]
    prob=SecondOrderODEProblem(msd,initialvel,initialdisp,tspan,p)
    sol = solve(prob)
    return sol
end

sol=spring()

my_time_range = 0:0.01:8

disp = getindex.(sol.(my_time_range),1)


plot(my_time_range,disp)

disp .+= 0.5*randn(size(disp)) + 0.8*my_time_range

plot!(my_time_range,disp)
