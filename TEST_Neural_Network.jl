
using Flux
using Flux: train!
using Plots

actual(x)=sin(x)

x_train, x_test = hcat(0:0.1:5...), hcat(6:0.1:10...)
y_train, y_test = actual.(x_train),actual.(x_test)

predict = Dense(1 => 1)

predict(x_train)

loss(x, y) = Flux.Losses.mse(predict(x), y)

loss(x_train, y_train)

opt = Descent()
data = [(x_train, y_train)]
parameters = Flux.params(predict)

for epoch in 1:200
    train!(loss, parameters, data, opt)
end

loss(x_train, y_train)

plot(x_test', actual.(x_test)')
plot!(x_test', predict(x_test)')

