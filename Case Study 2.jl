using Flux
using Statistics
using DifferentialEquations
using Plots
using Random

m=1
k=1
b=0.1

e=2.7182

ωn=sqrt(k)
ζ=b/(2*ωn)

function msdreal!(du,u,p,t)
    du[1] = u[2]
    du[2] = -(k)*u[1]-(b)*u[2]
end

initial = [1, 0]             # [pos,vel]
tspan = (0.0,60)             # time interval
  

probreal = ODEProblem(msdreal!,initial,tspan)
solreal = solve(probreal)

# Generate the training dataset
data_cycles=5
n=50# #the number of position data readings taken during this period

data_t_end=2*pi*(1/sqrt(k))*data_cycles  #the time period for which the position data has been measured
step_time=data_t_end/(n-1)

t_plot_data=data_t_end+step_time:step_time:data_t_end+pred_t_end+step_time
plot_position_ = [state[1] for state in solreal(t_plot_data)]
plot_velocity_r = [state[2] for state in solreal(t_plot_data)]

meas_t=0:step_time:data_t_end
meas_position_r = [state[1] for state in solreal(meas_t)]
meas_velocity_r = [state[2] for state in solreal(meas_t)]

E_data =0.5(e.^(-b.*(meas_t)))
pred_pos=((E_data./0.5).-(meas_velocity_r.^2))./k


E_pred_kinetic= 0.5*(m.*(meas_velocity_r.^2))
E_pred_potential=(0.5*(k.*(meas_position_r.^2)))
E_pred_total= 0.5*(m.*(meas_velocity_r.^2)).+(0.5*(k.*(meas_position_r.^2)))
plot(meas_t,E_data,label="E(t)")
plot!(meas_t,E_pred_kinetic,label="Kinetic Energy")
plot!(meas_t,E_pred_potential,label="Potential Energy")
plot!(meas_t,E_pred_total,label="Total  Kinetic and Potential Energy",xlabel="Time",ylabel="Energy" )
title!("E(t) as an approximation of total energy in a damped system", titlefontsize=10)

plot(meas_t,meas_position_r)
plot!(meas_t, pred_pos)



# Convert your data to Float32
pos_data = Float32.(meas_position_r)
vel_data = Float32.(meas_velocity_r)

window_size = Int(ceil(2n/data_cycles)) #The number of points of history the NN has

# Create a vector of vectors from the sliding window
X_data = [pos_data[i:i+window_size-1] for i in 1:(length(pos_data)-window_size)]
Y_data = pos_data[window_size+1:end]
V_data= vel_data[window_size+1:end]
E_data= E_data[window_size+1:end]


train_ratio = 0.5
train_size = Int(round(size(X_data, 1) * train_ratio))

#######


val_t=0.5*step_time:step_time:data_t_end+0.5*step_time
val_position_r = [state[1] for state in solreal(val_t)]
val_velocity_r = [state[2] for state in solreal(val_t)]

E_val =0.5(e.^(-b.*(val_t)))


# Convert your data to Float32
pos_val = Float32.(val_position_r)
vel_val = Float32.(val_velocity_r)

idx=randperm(length(X_data))

# Create a vector of vectors from the sliding window
X_val = [pos_val[i:i+window_size-1] for i in 1:(length(pos_val)-window_size)][idx]
Y_val = pos_val[window_size+1:end][idx]
V_val= vel_val[window_size+1:end][idx]
E_val= E_val[window_size+1:end][idx]

####
# this loss_equal is no longer used 
# function loss_equal()
#     loss=0
#     for i in 1:length(X_data)
#         loss=loss+NN(X_data[i])
#     end
#     return abs(sum(loss))*(1/length(X_data))
# end

loss_data()=Flux.Losses.mse(NN.(X_data) , Y_data)

loss_con()=Flux.Losses.mse(E_data,0.5 * (k.*NN.(X_data).^2 .+ m.*V_data.^2))
loss()=0.9*loss_data()+0.1*loss_con()


# Create the RNN model

NN = Chain( x-> x',
    LSTM(1,50),
    Dense(50,50,tanh),
    Dense(50,30,tanh),
    Dense(30,1),
    first
    )


loss()

opt = ADAM()
ps = Flux.params(NN)
data=Iterators.repeated((),100)
iter=0
train_val_store=Vector[]
cb = function () 
    Flux.reset!(NN) # Reset RNN
    global iter+=1
    loss_data_val=Flux.Losses.mse(NN.(X_val) , Y_val)
    loss_con_val=Flux.Losses.mse(E_val,0.5 * (k.*NN.(X_val).^2 .+ m.*V_val.^2))
    val_loss=0*loss_data_val+1*loss_con_val

    train_val=[loss(),val_loss]
    display(train_val)
    push!(train_val_store,train_val)
end


Flux.train!(loss, ps ,data,opt;cb=cb ) # Update parameters


pred_cycles=4
    
pred_t_end=2*pi*(1/sqrt(k))*pred_cycles 

t_plot=data_t_end+step_time:step_time:data_t_end+pred_t_end-step_time
plot_position_r = [state[1] for state in solreal(t_plot)]
plot_velocity_r = [state[2] for state in solreal(t_plot)]

numpreds=Int(pred_t_end÷(data_t_end/(n-1)))
test_data=pos_data[end-(window_size-1):end] #capturing the last few points of measured day to feed into NN for the prediction of the next point
    
for i in 1:numpreds
    test=test_data[end-(window_size-1):end]
    push!(test_data,NN(test))
end

t_fine=0:0.1:(data_t_end+pred_t_end-step_time)
fine_position_r = [state[1] for state in solreal(t_fine)]

NN_composed=Float64.(test_data[end-(numpreds-1):end-1])
NN_con=Float64.(test_data[end-(numpreds-1):end-1])
NN_data=Float64.(test_data[end-(numpreds-1):end-1])



Flux.Losses.mse(NN_2,plot_position_r)

# Flux.Losses.mse(NN_con5,plot_position_r)
# Flux.Losses.mse(NN_data,plot_position_r)
# Flux.Losses.mse(NN_con,plot_position_r)

# plot_1=0.5*((k*NN_composed_data2.^2)+(m*plot_velocity_r.^2))
# plot_2=sqrt.((e.^(-b.*(t_plot)))-(m*plot_velocity_r.^2))

plot(meas_t,meas_position_r,seriestype=:scatter,xlabel="Time",ylabel="Velocity",label="Position Measurements",markersize=3)
plot!(t_fine,fine_position_r,xlabel="Time",ylabel="Position",label="True Position",xlims=(0,50),linecolor="orange")

plot!(t_plot,NN_comp5,label="NNPhysics predictions",color="red",seriestype=:scatter )
title!("Position Time plot of CPINN when different values of a are used",legend=:topright,size=((600,400)),titlefontsize=10)
plot!(t_plot,NN_composed,seriestype=:scatter,label="Predicted Position (CPINN)",color="green")
plot!(t_plot,NN_con,seriestype=:scatter,label="Predicted Position (NNPhysics)",color="red")
plot!(t_plot,NN_data,seriestype=:scatter,label="Predicted Position (NNData)",color="purple")
plot!(t_fine,fine_position_r,xlabel="Time",ylabel="Position",label="True Position",xlims=(0,50),linecolor="orange")

# plot(t_plot, plot_1)
# plot!(t_plot,plot_2,)

plot(0:(length(train_val_store)-1),[vec[2] for vec in train_val_store],xlabel="Number of Training Cycles",ylabel="Loss(MSE)", label="Training loss")
plot!(0:(length(train_val_store)-1),[vec[1] for vec in train_val_store],xlabel="Number of Epochs",ylabel="Loss(MSE)", label="Validation loss")
title!("Loss curve for NNPhysics", titlefontsize=10)

############################## PREVIOUS VERSIONS OF CODE  ######################################

using Flux
using Statistics
using DifferentialEquations
using Plots

k = 3
breal = 0.2


function msdreal!(du,u,p,t)
    du[1] = u[2]
    du[2] = -(k)*u[1]-(breal)*u[2]
end

initial = [1, 0]             # [vel,pos]
tspan = (0.0,25)             # time interval
  

probreal = ODEProblem(msdreal!,initial,tspan)
solreal = solve(probreal)

# Generate the training dataset
data_cycles=3
n=40            #the number of position data readings taken during this period

data_t_end=2*pi*(1/sqrt(k))*data_cycles  #the time period for which the position data has been measured
step=t_end/(n-1)

meas_t=0:step:data_t_end
meas_position_r = [state[2] for state in solreal(meas_t)]

t = 0:0.1:20
data_position_r = [state[2] for state in solreal(t)]


# Convert your data to Float32
pos_data = Float32.(meas_position_r)

window_size = Int(ceil(2n/3)) #The number of points of history the NN has

# Create a vector of vectors from the sliding window
X_data = [pos_data[i:i+window_size-1] for i in 1:(length(pos_data)-window_size)]
Y_data = pos_data[window_size+1:end]

# Define a loss function

loss_data()=Flux.Losses.mse(NN.(X_data) , Y_data)

function loss_equal()
    loss=0
    for i in 1:length(X_data)
        loss=loss+NN(X_data[i])
    end
    return abs(sum(loss))*(1/length(X_data))
end

position_datastore=[]
for i in 1:10

    loss()=(i*0.1)*loss_data()+(1-i*0.1)*loss_equal()

    # Create the RNN model

    NN = Chain( x-> x',
    RNN(1,100),
    RNN(100,100),
    Dense(100,100,tanh),
    Dense(100,100,tanh),
    Dense(100,1),
    first
    )

    loss()

    opt = ADAM()
    ps = Flux.params(NN)
    data=Iterators.repeated((),500)


    iter=0
    losses=Float64[]
    cb = function ()
        Flux.reset!(NN) # Reset RNN
        global iter+=1
        push!(losses,loss())
        if iter % 100==0
            display(loss())
        end
    end

    Flux.train!(loss, ps ,data,opt;cb=cb ) # Update parameters
    
    pred_cycles=2
    
    pred_t_end=2*pi*(1/sqrt(k))*pred_cycles 
    t_plot=data_t_end+step:step:data_t_end+pred_t_end+step
    numpreds=Int(pred_time_end÷(data_t_end/(n-1)))
    test_data=pos_data[end-(window_size-1):end] #capturing the last few points of measured day to feed into NN for the prediction of the next point
    
    for i in 1:numpreds
        test=test_data[end-(window_size-1):end]
    push!(test_data,NN(test))
    end
    test_data=Float64.(test_data[end-(numpreds-1):end])
    push!(position_datastore,test_data)
end


i=
plot(t_plot,position_datastore[i],seriestype=:scatter,label="Predicted Position")
plot!(t,data_position_r, label="Real Position")
plot!(meas_t,meas_position_r,seriestype=:scatter,xlabel="Time",ylabel="Position",label="Position Measurements", legend=:outertopleft,size=(900,400) )

plot(0:(length(losses)-1),losses,xlabel="Number of Training Cycles",ylabel="Loss", label="Training loss")
title!("Loss curve for the function F= -x + 2xsin(2x) using loss_composed() [0.5,0.5]", titlefontsize=10,ylabel="Loss", label="Training loss")


