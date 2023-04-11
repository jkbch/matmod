

using GLPK, Cbc, JuMP, SparseArrays

include("Antons.jl")

function constructA(H,K)
   A = zeros(Float64,length(H),length(H))

   for i in 3:length(H)-2
    A[i-2,i] = K[3];
    A[i-1,i] = K[2];
    A[i,i] = K[1];
    A[i+1,i] = K[2];
    A[i+2,i] = K[3];
    end

    A[length(H)-2,length(H)] = K[3];
    A[length(H)-1,length(H)] = K[2];
    A[length(H),length(H)] = K[1];
    A[1,1] = K[1];
    A[2,1] = K[2];
    A[3,1] = K[3];
  
   return A
end


function solveIP(H, K)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructA(H,K)

    JuMP.@variable(myModel, x[1:h], Bin )
    JuMP.@variable(myModel, R[1:h] >= 0 )

    JuMP.@objective(myModel, Min, sum(x[j] for j=1:h) )

    JuMP.@constraint(myModel, [j=1:h],R[j] >= H[j] + 10 )
    JuMP. @constraint(myModel, [i=1:h],R[i] == sum(A[i,j]*x[j] for j=1:h) )

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
        return JuMP.value.(x)
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end
end

function solveIP4(H, K, display) # Stkålet fra Lukas
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructMatrix(H,K)

    @variable(myModel, x[1:h], Bin )
    @variable(myModel, R[1:h] >= 0 )
    @variable(myModel, Z[1:h] >= 0)

    @objective(myModel, Min, sum(  Z[j] for j=1:h) )

    @constraint(myModel, [j=1:h] ,R[j] >= H[j] + 10 )
    @constraint(myModel, [i=1:h] ,R[i] == sum(A[i,j]*x[j] for j=1:h) )
    @constraint(myModel, [i=1:h] ,R[i] - H[i] - 10 <= Z[i])
    @constraint(myModel, [i=1:h] ,-R[i] + H[i] + 10 <= Z[i])


    optimize!(myModel)
    if display == true
        if termination_status(myModel) == MOI.OPTIMAL
            println("Objective value: ", JuMP.objective_value(myModel))
            println("x = ", JuMP.value.(x))
            println("R = ", JuMP.value.(R))
        else
            println("Optimize was not succesful. Return code: ", termination_status(myModel))
        end
    end
    return JuMP.value.(x), JuMP.value.(R)
end
function solveIPSmoothFlow(H, K) # Løser opgave 5
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructA(H,K)

    JuMP.@variable(myModel, x[1:h], Bin )
    JuMP.@variable(myModel, R[1:h] >= 0 )
    JuMP.@variable(myModel, Z[1:h] >= 0 )

    JuMP.@objective(myModel, Min, sum(Z[j] for j=1:h) )

    JuMP.@constraint(myModel, [j=1:h],R[j] >= H[j] + 10 )
    JuMP.@constraint(myModel, [i=1:h],R[i] == sum(A[i,j]*x[j] for j=1:h) )
    JuMP.@constraint(myModel, [j=1:h],Z[j]  >= R[j]-H[j]-10 )
    JuMP.@constraint(myModel, [j=1:h],Z[j]  >= -(R[j]-H[j]-10) )

    JuMP.@constraint(myModel, [j=2:h-1],x[j] <= 1-x[j-1])
    #JuMP.@constraint(myModel, [j=2:h-1],x[j] <= 1-x[j+1])
    JuMP.@constraint(myModel, x[2] <= 1-x[1])
    JuMP.@constraint(myModel, x[h-1] <= 1-x[h])

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
        return JuMP.value.(x)
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end
end

function solveIPDialYield(H, K1,K2,K3) # Forsøger at løse opgave 6
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructA(H,K1)
    B = constructA(H,K2)
    C = constructA(H,K3)

    JuMP.@variable(myModel, x[1:h], Bin )
    JuMP.@variable(myModel, y[1:h], Bin )
    JuMP.@variable(myModel, z[1:h], Bin )
    JuMP.@variable(myModel, R[1:h] >= 0 )
    JuMP.@variable(myModel, Z[1:h] >= 0 )

    JuMP.@objective(myModel, Min, sum(Z[j] for j=1:h) )

    JuMP.@constraint(myModel, [j=1:h], 1 >= x[j]+y[j]+z[j])
    JuMP.@constraint(myModel, [j=2:h], x[j]+y[j]+z[j]  <= 1-x[j-1] -y[j-1] -z[j-1])

    JuMP.@constraint(myModel, [i=1:h],R[i] == sum(A[i,j]*x[j] + B[i,j]*y[j] + C[i,j]*z[j] for j=1:h) )

    JuMP.@constraint(myModel, [j=1:h],R[j] >= H[j] + 10 )
    JuMP.@constraint(myModel, [j=1:h],Z[j]  >= R[j]-H[j]-10 )
    JuMP.@constraint(myModel, [j=1:h],Z[j]  >= -(R[j]-H[j]-10) )

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
        return JuMP.value.(x)
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end
end

H = yvals_cubic
K1 = [300,140,40]
K2 = [500,230,60]
K3 = [1000,400,70]

#A = constructA(H,K1)
#Bombs = solveIPDialYield(H,K1,K2,K3)
Bombs = solveIPSmoothFlow(H,K1)
y_bomb = []
x_bomb = []
for i in eachindex(Bombs)
    if Bombs[i] == 1
        append!(y_bomb,H[i])
        append!(x_bomb,xvals[i])
    end
end

plot(accDistance, matrix[:,3], linewidth=4, label="Data")
scatter!(x_bomb,y_bomb)
sum(Bombs)