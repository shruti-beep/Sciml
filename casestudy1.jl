using DifferentialEquations
using OrdinaryDiffEq
using Flux
using Flux: train!
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

t1 = 0:0.01:8
dataplot=sol(t1)

pos= [state[2]  for state in dataplot]
vel= [state[1]  for state in dataplot]

t2=0:0.5:8
dataset=sol(t2)

posdata=[state[2]  for state in dataset]
veldata= [state[1]  for state in dataset]



plot(t1,vel)
scatter!(t2,veldata)

veldata.+= 0.5*randn(size(veldata)) + 0.8*t2

scatter!(t2,veldata)

NNvel =Chain(x ->[x],
 Dense(1,32,tanh)
 Dense(32,1),
 first)

predict()

loss() = Flux.Losses.mse(predict(x), y)

loss()

opt = Descent()
data = [(x_train, y_train)]
parameters = Flux.params(predict)

for epoch in 1:200
    train!(loss, parameters, data, opt)
end

loss(x_train, y_train)

plot(x_test', actual.(x_test)')
plot!(x_test', predict(x_test)')

