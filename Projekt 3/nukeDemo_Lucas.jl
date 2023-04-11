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
yvals = yvals .- yvals[1]

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
    return JuMP.value.(x), JuMP.value.(y), JuMP.value.(z), JuMP.value.(R)
end
x, R = solveIP(yvals, K, false)

function iszero(num)
    num != 0
end

yvals = yvals_cubic

x_4, R_4 = solveIP4(yvals, K, true)
x_5, R_5 = solveIP5(yvals, K, true)
ybombs_4 = vec(yvals .* x_4)
xbombs_4 = vec(xvals .* x_4)
filter!(iszero, ybombs_4)
filter!(iszero, xbombs_4)
ybombs = vec(yvals .* x)
xbombs = vec(xvals .* x)
filter!(iszero, ybombs)
filter!(iszero, xbombs)
ybombs_5 = vec(yvals .* x_5)
xbombs_5 = vec(xvals .* x_5)
filter!(iszero, ybombs_5)
filter!(iszero, xbombs_5)

scatter(xbombs[1:20], ybombs[1:20], label = "Basis case", markersize = 5)
xlabel!("Distance")
ylabel!("Height")
title!("Bombs placed in first interval", markersize = 5)
scatter!(xbombs_4[1:20], ybombs_4[1:20], label = "Optimize for channel smoothness", shape = [:xcross :o :utri], markersize = 7)
scatter!(xbombs_5[1:20], ybombs_5[1:20],  label = "Optimize for channel smoothness and placement", shape=[:diamond :o :utri])

scatter(xbombs[90:end], ybombs[90:end], label = "Basis case")
xlabel!("Distance")
ylabel!("Height")
title!("Bombs placed in last interval")
scatter!(xbombs_4[90:end], ybombs_4[90:end], label = "Optimize for channel smoothness", shape = [:xcross :o :utri], markersize = 6)
scatter!(xbombs_5[90:end], ybombs_5[90:end],  label = "Optimize for channel smoothness and placement", shape=[:diamond :o :utri])

scatter(xbombs, ybombs, label = "Basis case")
xlabel!("Distance")
ylabel!("Height")
title!("Bombs placed")
scatter!(xbombs_4, ybombs_4, label = "Optimize for channel smoothness", shape = [:xcross :o :utri], markersize = 6)
scatter!(xbombs_5, ybombs_5,  label = "Optimize for channel smoothness and placement", shape=[:diamond :o :utri])
#x_6, R_6 = solveIP6(yvals, K, K2, K3, true)



scatter(xvals,R .- yvals, label = "Basis case")
scatter!(xvals, R_4 .-yvals, label = "Optimize for channel smoothness")
scatter!(xvals, R_5 .-yvals, label = "Optimize for channel smoothness and placement")
title!("Dirt removed as a function of distance")
xlabel!("Distance")
ylabel!("Channel depth")

x_61, y_61, z_61,  R_61 = solveIP6(yvals[1:80], K, K2, K3, false)
x_61e = 0
y_61e = 0
z_61e = 0
x_62, y_62, z_62,  R_62 = solveIP6(yvals[82:160], K, K2, K3, false) # Ingen bombe til sidst
x_63, y_63, z_63,  R_63 = solveIP6(yvals[161:241], K, K2, K3, false) # Ingen bombe til sidst
x_64, y_64, z_64,  R_64 = solveIP6(yvals[242:318], K, K2, K3, false)
#scatter!(xvals[1:100], R_6 .-yvals[1:100], label = "Opgave 6", markerstrokewidth = 2)

x_6 = append!(x_61,x_61e)
x_6 = append!(x_6, x_62)
x_6 = append!(x_6, x_63)
x_6 = append!(x_6, x_64)

y_6 = append!(y_61, y_61e)
y_6 = append!(y_6, y_62)
y_6 = append!(y_6, y_63)
y_6 = append!(y_6, y_64)

z_6 = append!(z_61, z_61e)
z_6 = append!(z_6, z_62)
z_6 = append!(z_6, z_63)
z_6 = append!(z_6, z_64)

R_6 = append!(R_61, R_62)
R_6 = append!(R_6, R_6[lastindex(R_6)])
R_6 = append!(R_6, R_63)
R_6 = append!(R_6, R_64)

x_bombsx = vec(x_6 .*xvals)
y_bombsx = vec(x_6 .*yvals)
x_bombsy = vec(y_6 .*xvals)
y_bombsy = vec(y_6 .*yvals)
x_bombsz = vec(z_6 .*xvals)
y_bombsz = vec(z_6 .*yvals)
filter!(iszero, x_bombsx)
filter!(iszero, y_bombsx)
filter!(iszero, x_bombsy)
filter!(iszero, y_bombsy)
filter!(iszero, x_bombsz)
filter!(iszero, y_bombsz)

scatter(xvals,R .- yvals, label = "Basis case")
scatter!(xvals, R_4 .-yvals, label = "Optimize for channel smoothness")
scatter!(xvals, R_5 .-yvals, label = "Optimize for channel smoothness and placement")
scatter!(xvals, R_6 .-yvals, label = "Optimized for smoothness, placement and load")
title!("Dirt removed as a function of distance")
xlabel!("Distance")
ylabel!("Channel depth")

scatter(x_bombsx, y_bombsx, label = "Small Bombs", shape = [:diamond :o :utri], markersize = 6)
scatter!(x_bombsy, y_bombsy, label = "Medium Bombs", shape = [:hexagon :o :utri], markersize = 6)
scatter!(x_bombsz, y_bombsz, label = "Large Bombs", shape = [:xcross :o :utri], markersize = 6)
title!("Placement of  bombs of different types")
xlabel!("Distance from sea [km]")
ylabel!("Height [m]")


scatter(xbombs[90:end], ybombs[90:end], label = "Basis case")
xlabel!("Distance")
ylabel!("Height")
title!("Bombs placed in last interval")
scatter!(xbombs_4[90:end], ybombs_4[90:end], label = "Optimize for channel smoothness", shape = [:xcross :o :utri], markersize = 6)
scatter!(xbombs_5[90:end], ybombs_5[90:end],  label = "Optimize for channel smoothness and placement", shape=[:diamond :o :utri])
scatter!(x_bombsx[90:end], y_bombsx[90:end],  label = "Smoothness, Load size and placement [small]", shape=[:diamond :o :utri])
scatter!(x_bombsy[90:end], y_bombsy[90:end],  label = "Smoothness, Load size and placement [medium]", shape=[:diamond :o :utri])


scatter(xvals, yvals .- R .-10, label =false)
scatter(xvals, yvals .- R_4 .-10, label = false)
scatter(xvals, yvals .- R_5 .-10, label = false)
scatter(xvals, yvals .- R_6 .-10, label = false)
title!("Channel depth as a function of distance")
title!("Basis Case")
title!("With smoothness")
title!("Smoothness and placement")
title!("Smoothness, Placement and load size")
xlabel!("Distance [km]")
ylabel!("Channel depth [m]")