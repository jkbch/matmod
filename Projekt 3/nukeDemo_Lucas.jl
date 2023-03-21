using GLPK,  SparseArrays, Cbc
using JuMP
include("Lucas_legetøj.jl")
function constructMatrix(yvals,K)
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
K2 = [500 230 60]
K3 = [1000 400 70]
yvals = reshape(yvals, lastindex(yvals), 1)
print(size(yvals))

function solveIP(H, K, display)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructMatrix(H,K)

    @variable(myModel, x[1:h], Bin )
    @variable(myModel, R[1:h] >= 0 )

    @objective(myModel, Min, sum(  x[j] for j=1:h) )

    @constraint(myModel, [j=1:h],R[j] >= H[j] + 10 )
    @constraint(myModel, [i=1:h],R[i] == sum(A[i,j]*x[j] for j=1:h) )

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

function solveIP4(H, K, display)
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

function solveIP5(H, K, display)
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

    @constraint(myModel, [i=2:h-1], x[i] <= 1- x[i-1])
    @constraint(myModel, [i=2:h-1], x[i] <= 1- x[i+1])
    @constraint(myModel, x[2] <= 1 - x[1] )
    @constraint(myModel, x[h-1] <= 1 - x[h])

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

function solveIP6(H, K1,K2,K3, display)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructMatrix(H,K1)
    B = constructMatrix(H,K2)
    C = constructMatrix(H,K3)
    # Tilføjet flere bomber
    @variable(myModel, x[1:h], Bin )
    @variable(myModel, y[1:h], Bin )
    @variable(myModel, z[1:h], Bin )

    @variable(myModel, R[1:h] >= 0 )
    @variable(myModel, Z[1:h] >= 0)

    @objective(myModel, Min, sum(  Z[j] for j=1:h) )

    @constraint(myModel, [j=1:h] ,R[j] >= H[j] + 10 )
    # Tilføjet flere bomber
    @constraint(myModel, [i=1:h] ,R[i] == sum(A[i,j]*x[j] + B[i,j]*y[j] + C[i,j]*z[j] for j=1:h) )
    @constraint(myModel, [i=1:h] ,R[i] - H[i] - 10 <= Z[i])
    @constraint(myModel, [i=1:h] ,-R[i] + H[i] + 10 <= Z[i])

    @constraint(myModel, [i=2:h-1], x[i] + y[i] + z[i] <= 1- x[i-1] - y[i-1] - z[i-1] )
    @constraint(myModel, [i=2:h-1], x[i] + y[i] + z[i] <= 1- x[i+1] - y[i+1] - z[i+1])
    @constraint(myModel, x[2] + y[2] + z[2] <= 1 - x[1] - y[1] - z[1] )
    @constraint(myModel, x[h-1] + y[h-1] + z[h-1] <= 1 - x[h] - y[h] - z[h])

    @constraint(myModel, [i = 1:h], x[i] + y[i] + z[i] <= 1)

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

x, R = solveIP(yvals,K, false)

function iszero(num)
    num != 0
end

yvals = yvals_cubic

x_new, R_new = solveIP4(yvals, K, true)
# https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values
ybombs_new = vec(yvals .* x_new)
xbombs_new = vec(xvals .* x_new)
filter!(iszero, ybombs_new)
filter!(iszero, xbombs_new)
scatter!(xbombs_new, ybombs_new, markerstrokewidth = 1, label = "Anden gang")
x_5, R_5 = solveIP5(yvals, K, true)
#x_6, R_6 = solveIP6(yvals, K, K2, K3, true)

scatter(xvals,R .- yvals, label = "Basis")
scatter!(xvals, R_new .-yvals, label = "Opgave 4")
scatter!(xvals, R_5 .-yvals, label = "Opgave 5")
#scatter!(xvals, R_6 .-yvals, label = "Opgave 6")
