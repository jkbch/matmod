

using GLPK, Cbc, JuMP, SparseArrays

include("Antons.jl")

H = matrix[:,3]


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
    return A
    A[length(H)-2,length(H)] = 40;
    A[length(H)-1,length(H)] = 140;
    A[length(H),length(H)] = 300;
    A[1,1] = 300;
    A[2,1] = 140;
    A[3,1] = 40;
   end
   return A
end


function solveIP(H, K)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructA(H,K)

    @variable(myModel, x[1:h], Bin )
    @variable(myModel, R[1:h] >= 0 )

    @objective(myModel, Min, sum(x[j] for j=1:h) )

    @constraint(myModel, [j=1:h],R[j] >= H[j] + 10 )
    @constraint(myModel, [i=1:h],R[i] == sum(A[i,j]*x[j] for j=1:h) )

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end
end

A = constructA(H,K)
solveIP(H,K)
