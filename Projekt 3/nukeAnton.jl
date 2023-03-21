

using GLPK, Cbc, JuMP, SparseArrays

include("Antons.jl")

H = ys


K = [
300 140 40
]


function constructA(H,K)
   A = zeros(Float64,length(H),length(H))

   for i in 3:length(H)-2
    A[i-2,i] = K[2];
    A[i-1,i] = K[1];
    A[i,i] = K[0];
    A[i+1,i] = K[1];
    A[i+2,i] = K[2];
    end

    A[length(H)-2,length(H)] = K[2];
    A[length(H)-1,length(H)] = K[1];
    A[length(H),length(H)] = K[0];
    A[1,1] = K[0];
    A[2,1] = K[1];
    A[3,1] = K[2];
  
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
function solveIPSmoothFlow(H, K)
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
    JuMP.@constraint(myModel, [j=2:h-1],x[j] <= 1-x[j+1])
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
function solveIPDialYield(H, K1,K2,K3)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructA(H,K1)
    B = constructA(H,K2)
    C = constructA(H,K3)

    JuMP.@variable(myModel, x[1:h], Bin )
    JuMP.@variable(myModel, type1[1:h], Bin )
    JuMP.@variable(myModel, type2[1:h], Bin )
    JuMP.@variable(myModel, type3[1:h], Bin )
    JuMP.@variable(myModel, R[1:h] >= 0 )
    JuMP.@variable(myModel, Z[1:h] >= 0 )

    JuMP.@objective(myModel, Min, sum(Z[j] for j=1:h) )

    JuMP.@constraint(myModel, [j=1:h],R[j] >= H[j] + 10 )
    JuMP.@constraint(myModel, [i=1:h],R[i] == sum(A[i,j]*x[j]*type1[j]+A[i,j]*x[j]*type2[j]+A[i,j]*x[j]*type3[j] for j=1:h) )
    JuMP.@constraint(myModel, [j=1:h],Z[j]  >= R[j]-H[j]-10 )
    JuMP.@constraint(myModel, [j=1:h],Z[j]  >= -(R[j]-H[j]-10) )

    JuMP.@constraint(myModel, [j=2:h-1],x[j] <= 1-x[j-1])
    JuMP.@constraint(myModel, [j=2:h-1],x[j] <= 1-x[j+1])
    JuMP.@constraint(myModel, x[2] <= 1-x[1])
    JuMP.@constraint(myModel, x[h-1] <= 1-x[h])


    JuMP.@constraint(myModel, [j=1:h],x[j] == type1[j]+type2[j]+type3[j] )

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

K1 = [300,140,40]
K2 = [500,230,60]
K3 = [1000,400,70]

A = constructA(H,K)
Bombs = solveIPFlolYield(H,K)
y_bomb = []
x_bomb = []
for i in eachindex(Bombs)
    if Bombs[i] == 1
        append!(y_bomb,ys[i])
        append!(x_bomb,x[i])
    end
end

plot(accDist, matrix[:,3], linewidth=4, label="Data")
scatter!(x_bomb,y_bomb)