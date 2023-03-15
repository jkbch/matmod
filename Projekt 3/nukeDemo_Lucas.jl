using GLPK,  SparseArrays
using JuMP
include("Lucas_legetøj.jl")
function constructA(yvals,K)
    n = size(yvals,1)

    A = zeros(Float64,n,n)
    K2 = [K[3], K[2], K[1], K[2], K[3]]
    K3 = [K[3], K[2], K[1]]
    for i in 3:lastindex(yvals) - 2
        A[i,i-2:i+2] = K2
    end
    A[1,1:3] = K
    A[2,2:4] = K
    A[2,1] = K[2]
    A[end, end-2:end] = K3
    A[end-1, end-3: end-1] = K3
    A[end-1, end] = K[3]
    return A
end
K = [
300 140 40
]

function solveIP(yvals, K)
    h = length(yvals)
    myModel = Model(GLPK.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructA(yvals,K)

    @variable(myModel, x[1:h], Bin )
    @variable(myModel, R[1:h] >= 0 )

    @objective(myModel, Min, sum(x[j] for j=1:h) )

    @constraint(myModel, [j=1:h],R[j] >= yvals[j] + 10 )
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

solveIP(yvals,K)
