## In this case study , PINNs are used to discover the non linear extension in a spring.
## limited data is available to describes the non linear force , and a model force (which follows hookes law )
## is said to be a good approximation of the non linear force.
## Both the model and the data are used in a composed loss function to discover the non linear function

#importing modules
using DifferentialEquations
using Flux
using Flux: train!
using Plots
using Random
using Interpolations

#defining spring constant
k=1
                
#generating the model force data
forcemodel(dx,x,k,t) = -k*x
probmodel = SecondOrderODEProblem(forcemodel,1.0,0.0,(0.0,20.0),k)
solmodel = solve(probmodel)

#generating the real (or true) force data
forcereal(dx,x,k,t) = -k*x .+ 2*sin.(2x)
probreal = SecondOrderODEProblem(forcereal,1.0,0.0,(0.0,20.0),k)
solreal = solve(probreal)

#generate time data
t = 0:0.1:19.9

#generate model training data to 
t_gen=0:0.5:19.9

gen_position_r = [state[2] for state in solreal(t_gen)]
gen_force_r = [forcereal(state[1],state[2],k,t) for state in solreal(t_gen)]
gen_force_m = [forcemodel(state[1],state[2],k,t) for state in solreal(t_gen)]


#generate the data training dataset
t_meas=0:(5):20
t_meas=[6,7,9,2,0,3]
t_meas=[9,7,9,10,2,0,3,4]
t_meas=[5,8,10,13,15,6,7,1]
t_meas=[4,11,2,16,14,7,15,8]

meas_position_r = [state[2] for state in solreal(t_meas)]
meas_force_r = [forcereal(state[1],state[2],k,t) for state in solreal(t_meas)]

#test data is all the values not used for training
t_test = setdiff(t,t_gen)

test_position_r = [state[2] for state in solreal(t_test)]
test_force_r = [forcereal(state[1],state[2],k,t) for state in solreal(t_test)]
test_force_m = [forcemodel(state[1],state[2],k,t) for state in solreal(t_test)]

# Plots to show problem set up
plot(meas_position_r,meas_force_r,seriestype=:scatter,xlabel="Position",ylabel="Force",label="Force Measurements")
plot!(gen_position_r,gen_force_m,label="Model Force")
plot!(gen_position_r,gen_force_r,label="Real Force")
title!("Force-Position plot for the function F= -x + 2sin(2x) ", titlefontsize=10)


# plot(t_meas,meas_position_r,seriestype=:scatter)
# plot!(t_gen, gen_position_r)

# plot(t_meas,meas_force_r,seriestype=:scatter,xlabel="Time",ylabel="Force",label="Force Measurements")
# plot!(t_gen,gen_force_m,label="Model Force")
# plot!(t_gen,gen_force_r,label="Real Force")
# title!("Force-Time plot for the function F= -x + 2sin(2x) ", titlefontsize=10,label="Measured Force",legend=:topright)

#data generation

dataset_meas=hcat(meas_position_r,meas_force_r)
dataset_gen=hcat(gen_position_r,gen_force_m,gen_force_r)

dataset_meas = dataset_meas[shuffle(1:end), :]
dataset_gen = dataset_gen[shuffle(1:end), :]

#splitting data into training and validation datasets
train_ratio = 0.5

train_size_meas = Int(round(size(dataset_meas, 1) * train_ratio))
train_meas = dataset_meas[1:train_size_meas, :]
val_meas = dataset_meas[train_size_meas+1:end, :]

train_size_gen = Int(round(size(dataset_gen, 1) * train_ratio))
train_gen = dataset_gen[1:train_size_gen, :]
val_gen = dataset_gen[train_size_gen+1:end, :]

#definining loss functions
loss_data() = Flux.Losses.mse(NNForce.(train_meas[:,1]) , train_meas[:,2])
    
loss_model() = Flux.Losses.mse(NNForce.(train_gen[:,1]) , train_gen[:,2] )

#creating composed loss function
loss()=0.7*loss_data()+0.3*loss_model()


#creating neural networom architecture
NNForce = Chain(x -> [x],
           Dense(1 => 100,tanh),
           Dense(100 =>200,tanh),
           Dense(200=>200,tanh),
           Dense(200=>100,tanh),
           Dense(100 => 1),
           first)

#setting up the training and callback function
opt = Adam()
data = Iterators.repeated((), 1000)
iter = 0
train_losses=Float64[]
val_losses=Float64[]
test_losses=Float64[]
cb = function () #callback function to observe training
    global iter += 1
    train_loss_data=Flux.Losses.mse(NNForce.(train_meas[:,1]) , train_meas[:,2])
    train_loss_model=Flux.Losses.mse(NNForce.(train_gen[:,1]) , train_gen[:,2])
    #train_loss=0.7*train_loss_data+0.3*train_loss_model
    train_loss=train_loss_data
    push!(train_losses,train_loss)
    val_loss_data=Flux.Losses.mse(NNForce.(val_meas[:,1]) , val_meas[:,2])
    val_loss_model=Flux.Losses.mse(NNForce.(val_gen[:,1]) , val_gen[:,2])
    #val_loss=0.7*val_loss_data+0.3*val_loss_model
    val_loss=val_loss_data
    push!(val_losses,val_loss)
    test_loss=Flux.Losses.mse(NNForce.(test_position_r), test_force_r)
    push!(test_losses,test_loss)
    if iter % 3 == 0
        train_val=(train_loss,val_loss)
        display(train_val)
    end

end
display(loss())

#training the model
Flux.train!(loss, Flux.params(NNForce), data, opt; cb=cb)


#plotting the loss curve
plot(0:((length(train_losses)-1)),train_losses,xlabel="Number of Epochs",ylabel="Loss(MSE)", label="Training loss")
plot!(0:((length(val_losses)-1)),val_losses,xlabel="Number of Epochs",ylabel="Loss (MSE)", label="Validation loss")
plot!(0:((length(test_losses)-1)),test_losses,xlabel="Number of Epochs",ylabel="Loss (MSE)", label="Test loss")
title!("Loss curve for NNdata 3", titlefontsize=10,yticks=0:0.5:8,ylims = (0, 8), legend= false)

#testing the trained network on test data
learned_force_1= NNForce.(test_position_r)


#finding the mse between the NN predictions and the true force  - test loss
mse_test=Flux.Losses.mse(NNForce.(test_position_r), test_force_r)

#plotting results on a force-time plot
plot(t_meas,meas_force_r,seriestype=:scatter,xlabel="t",ylabel="Force",label="Measured Force",legend=:outertopleft,size=((900,400)))
plot!(t_test,test_force_r,xlabel="t",ylabel="Force",label="Real Force",linecolor="green")
plot!(t_test,test_force_m,xlabel="t",label="Model Force",linecolor="blue")
plot!(t_test,learned_force_1,label="Predicted Force (loss_data)", linecolor="purple")
plot!(t_test,learned_force_model,label="Predicted Force (loss_model)",linecolor="orange")
plot!(t_test,learned_force_composed,label="Predicted Force (loss_composed 50-50)",linecolor="red")
title!("Force-Time plot for the function F= -x + 2sin(2x) ", titlefontsize=10, legend=false,size=[600,400])


#sorting values before plotting for a smoother curve
test_position_r_sort=sort(test_position_r)
learned_force_1_sort= NNForce.(test_position_r_sort)
A=hcat(test_position_r,test_force_r,)
A=A[sortperm(A[:, 1]), :]
mse_test=Flux.Losses.mse(learned_force_composed_2, A[:,2])

#plotting results on a position- force plot
plot(meas_position_r[1:4],meas_force_r[1:4],seriestype=:scatter,xlabel="Position",ylabel="Force",label="Force Measurements Training",markercolor="blue")#size=((900,400)))
plot!(meas_position_r[5:8],meas_force_r[5:8],seriestype=:scatter,xlabel="Position",ylabel="Force",label="Force Measurements Validation",markercolor="light blue")
plot!(A[:,1],A[:,2],label="Real Force",linecolor="green")
plot!(test_position_r,test_force_m,label="Model Force",linecolor="blue")
plot!(test_position_r_sort,learned_force_1_sort,label="NNdata 1", linecolor="purple")
#plot!(test_position_r_sort,learned_force_model,label="Predicted Force (loss_model)",linecolor="orange")
plot!(test_position_r_sort,learned_force_composed_2,label="PINN 1",linecolor="red")
title!("Force-Position plot for the function F= -x + 2sin(2x) Overfitted", titlefontsize=10,)




##################### EXTRA CODE USED FOR ANALYSIS#################################



using DifferentialEquations
using Flux
using Flux: train!
using Plots
using Random

k=1
                                                            
forcemodel(dx,x,k,t) = -k*x
probmodel = SecondOrderODEProblem(forcemodel,1.0,0.0,(0.0,20.0),k)
solmodel = solve(probmodel)


forcereal(dx,x,k,t) = -k*x .+ 2*sin.(2x)
probreal = SecondOrderODEProblem(forcereal,1.0,0.0,(0.0,20.0),k)
solreal = solve(probreal)

t = 0:0.1:19.9
data_position_r = [state[2] for state in solreal(t)]
data_force_r = [forcereal(state[1],state[2],k,t) for state in solreal(t)]
data_force_m = [forcemodel(state[1],state[2],k,t) for state in solreal(t)]

plot_t = 0:0.01:20

plot_position_r = [state[2] for state in solreal(plot_t)]
plot_force_r = [forcereal(state[1],state[2],k,t) for state in solreal(plot_t)]
plot_force_m = [forcemodel(state[1],state[2],k,t) for state in solreal(plot_t)]


plot(plot_position_r,plot_force_r,label="Real Force",xlabel="Position",ylabel="Force",legend=:outertopleft,size=((900,400)))
plot!(plot_position_r,plot_force_m,label="Model Force")


data_losses=Float64[]
position_datastore=[]
force_datastore=[]
learnedforce=[]
t_meas= Set([0, 3, 5, 18])
t_meas_data=[0,3,5,18,11,4,8,7,9,19,2,1,10,15,17,13,12,14]

for i in 4:10
    
    meas_position_r = [state[2] for state in solreal(collect(t_meas))]
    push!(position_datastore,meas_position_r)
    meas_force_r = [forcereal(state[1],state[2],k,t) for state in solreal(collect(t_meas))]
    push!(force_datastore,meas_force_r)
    
    
    dataset_meas=hcat(meas_position_r,meas_force_r)
    dataset_data=hcat(data_position_r,data_force_m,data_force_r)
    
    
    dataset_meas = dataset_meas[shuffle(1:end), :]
    dataset_data = dataset_data[shuffle(1:end), :]
    
    train_ratio = 0.7
    
    train_size_meas = Int(round(size(dataset_meas, 1) * train_ratio))
    train_meas = dataset_meas[1:train_size_meas, :]
    val_meas = dataset_meas[train_size_meas+1:end, :]
    
    train_size_data = Int(round(size(dataset_data, 1) * train_ratio))
    train_data = dataset_data[1:train_size_data, :]
    val_data = dataset_data[train_size_meas+1:end, :]
    
    
    
    loss_data() = Flux.Losses.mse(NNForce.(train_meas[:,1]) , train_meas[:,2])
    
    loss_model() = Flux.Losses.mse(NNForce.(train_data[:,1]) , train_data[:,2] )
    

    function loss_equal()
        loss=0
        for i in 1:length(train_data[:,1])
            loss=loss+NNForce(train_data[i,1])
        end
        return abs(sum(loss))
    end

    if i<8
        loss()=(0.2+0.1*i)*loss_data()+(0.8-0.1*i)*loss_model()
    else
        loss()=loss_data()
    end

    NNForce = Chain(x -> [x],
        Dense(1 => 100,tanh),
        Dense(100 =>200,tanh),
        Dense(200=>100,tanh),
        Dense(100 => 1),
        first)
            
    opt = Adam()
    data = Iterators.repeated((), 150)
    iter = 0
    lossreal = 0
 
    cb = function () #callback function to observe training
        global iter += 1
        if iter % 20 == 0
            global lossreal=Flux.Losses.mse(NNForce.(val_data[i,1]) , val_data[i,3] )
            display(lossreal)
        end
    end

    Flux.train!(loss, Flux.params(NNForce), data, opt; cb=cb)
    push!(data_losses,lossreal)
 

    learned_force= NNForce.(plot_position_r)
    push!(learnedforce,learned_force)

    push!(t_meas,t_meas_data[i+1])

end


plot(4:10,data_losses,seriestype=:scatter,xlabel="Number of Data points",ylabel="Loss",xticks=0:20)
title!("Loss value decreases with increasing data points (loss_data)", titlefontsize=10)


plot(position_datastore[i],force_datastore[i],seriestype=:scatter,xlabel="Position",ylabel="Force",label="Force Measurements",legend=:outertopleft,size=((900,400)))
plot(plot_position_r,plot_force_r,label="Real Force",linecolor=:lightgreen)
plot!(plot_position_r,plot_force_m,label="Model Force",linecolor=:red)
plot!(plot_position_r,learnedforce[i],label="Datapoints= "*string(i+3),linecolor=:blue,alpha=-0.1+(0.2*i))
title!("Force-Position plot changes as no.datapoints increase- changing loss weights",titlefontsize=10)

plot(0:4:((length(train_losses)-1)*4),train_losses,xlabel="Number of Training Cycles",ylabel="Loss", label="Training loss")
plot!(0:4:((length(val_losses)-1)*4),val_losses,xlabel="Number of Training Cycles",ylabel="Loss", label="Validation loss")
title!("Loss curve for the function F= -x + 2xsin(2x) using loss_data() ", titlefontsize=10)

learned_force_composed= NNForce.(plot_position_r)
learned_force_final = plot_force_m.+learned_force


plot(t_meas,meas_force_r,seriestype=:scatter,xlabel="t",ylabel="Mean of Variances",label="Measured Force",legend=:outertopleft,size=((900,400)))
plot(plot_t,plot_force_r,xlabel="t",ylabel="Force",label="Real Force",legend=:outertopleft,size=((900,400)))
plot!(plot_t,plot_force_m,xlabel="t",label="Model Force")
plot!(plot_t,learned_force_data,label="Predicted Force (loss_data)")
plot!(plot_t,learned_force_model,label="Predicted Force (loss_model)")
plot!(plot_t,learned_force_composed,label="Predicted Force (loss_composed)")
title!("Mean of variance of Neural network predictions as no. datapoints increase ", titlefontsize=10)

plot(meas_position_r,meas_force_r,seriestype=:scatter,xlabel="Position",ylabel="Force",label="Force Measurements",legend=:outertopleft,size=((900,400)))
plot(plot_position_r,plot_force_r,label="Real Force",legend=:outertopleft,size=((900,400)))
plot!(plot_position_r,plot_force_m,label="Model Force")
plot!(plot_position_r,learned_force_data,label="Predicted Force (loss_data)")
plot!(plot_position_r,learned_force_model,label="Predicted Force (loss_model)")
plot!(plot_position_r,learned_force_composed,label="Predicted Force (loss_composed)")
title!("Force-Position plot for the function F= -x + 2sin(2x) ", titlefontsize=10)


