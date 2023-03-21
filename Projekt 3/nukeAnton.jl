

using GLPK, Cbc, JuMP, SparseArrays

include("Antons.jl")

H = ys


K = [
300 140 40
]


function constructA(H,K)
   A = zeros(Float64,length(H),length(H))

   for i in 3:length(H)-2
    A[i-2,i] = 40;
    A[i-1,i] = 140;
    A[i,i] = 300;
    A[i+1,i] = 140;
    A[i+2,i] = 40;
    end

    A[length(H)-2,length(H)] = 40;
    A[length(H)-1,length(H)] = 140;
    A[length(H),length(H)] = 300;
    A[1,1] = 300;
    A[2,1] = 140;
    A[3,1] = 40;
  
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

A = constructA(H,K)
Bombs = solveIPSmoothFlow(H,K)
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