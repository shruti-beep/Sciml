using Flux
using Statistics
using Plots

NNODE = Chain(x -> [x], # Take in a scalar and transform it into an array
           Dense(1,32,tanh),
           Dense(32,1),
           first) # Take first value, i.e. return a scalar
NNODE(1.0)

g(t) = t*NNODE(t) + 1f0


ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t+ϵ)-g(t))/ϵ) - cos(2π*t)) for t in 0:0.01:1.0)

ttrain=0:0.01:1.0
utrain=cos.(2π*ttrain)


opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(loss())
  end
end
display(loss())
Flux.train!(loss, Flux.params(NNODE), data, opt; cb=cb)

t = 0:0.001:1.0
plot(t,g.(t),label="NN")
plot!(t,1.0 .+ sin.(2π.*t)/2π, label = "True Solution")
scatter!(ttrain,utrain)