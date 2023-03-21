using DelimitedFiles, Plots, DataInterpolations, LinearAlgebra, GLPK, Cbc, JuMP, SparseArrays

function haversine_distance(lat1, lon1, lat2, lon2)
    R = 6371e3 # Radius of the earth in meters
    dLat = deg2rad(lat2 - lat1)
    dLon = deg2rad(lon2 - lon1) 
    a = sin.(dLat/2) .* sin.(dLat/2) +
        cos.(deg2rad(lat1)) .* cos.(deg2rad(lat2)) .* 
        sin.(dLon/2) .* sin.(dLon/2)
    c = 2 * atan.(sqrt.(a), sqrt.(1 .- a)) 
    d = R * c
    return d
end

function deg2rad(deg)
    return deg * (pi/180)
end

function constructA(H,K)
    n = length(H)
    A = diagm(fill(K[1], n))

    for (i, k) in enumerate(K[2:end])
        A += diagm(i => fill(k, n-i), -i => fill(k, n-i))
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

    @constraint(myModel, [j=1:h],R[j] >= H[j] + CHD )
    @constraint(myModel, [i=1:h],R[i] == sum(A[i,j]*x[j] for j=1:h) )

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end

    return (JuMP.value.(x), JuMP.value.(R))
end

function solveIP2(H, K)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructA(H,K)

    @variable(myModel, x[1:h], Bin )
    @variable(myModel, R[1:h] >= 0 )
    @variable(myModel, Z[1:h] >= 0)

    @objective(myModel, Min, sum(Z[j] for j=1:h) )

    @constraint(myModel, [j=1:h], R[j] >= H[j] + CHD )
    @constraint(myModel, [i=1:h], R[i] == sum(A[i,j]*x[j] for j=1:h) )
    @constraint(myModel, [i=1:h], R[i] - H[i] - CHD <= Z[i])
    @constraint(myModel, [i=1:h],-R[i] + H[i] + CHD <= Z[i])

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end

    return (JuMP.value.(x), JuMP.value.(R))
end

function solveIP3(H, K)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructA(H,K)

    @variable(myModel, x[1:h], Bin )
    @variable(myModel, R[1:h] >= 0 )
    @variable(myModel, Z[1:h] >= 0)

    @objective(myModel, Min, sum(Z[j] for j=1:h) )

    @constraint(myModel, [j=1:h], R[j] >= H[j] + CHD )
    @constraint(myModel, [i=1:h], R[i] == sum(A[i,j]*x[j] for j=1:h) )

    @constraint(myModel, [i=1:h], R[i] - H[i] - CHD <= Z[i])
    @constraint(myModel, [i=1:h],-R[i] + H[i] + CHD <= Z[i])

    @constraint(myModel, [i=1+1:h], x[i] <= 1 - x[i-1])
    @constraint(myModel, [i=1:h-1], x[i] <= 1 - x[i+1])

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end

    return (JuMP.value.(x), JuMP.value.(R))
end

function solveIP4(H, K1, K2, K3)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A1 = constructA(H, K1)
    A2 = constructA(H, K2)
    A3 = constructA(H, K3)

    @variable(myModel, x1[1:h], Bin )
    @variable(myModel, x2[1:h], Bin )
    @variable(myModel, x3[1:h], Bin )

    @variable(myModel, R[1:h] >= 0 )
    @variable(myModel, Z[1:h] >= 0)

    @objective(myModel, Min, sum(Z[j] for j=1:h) )

    @constraint(myModel, [j=1:h], R[j] >= H[j] + CHD )
    @constraint(myModel, [i=1:h], R[i] == sum(A1[i,j]*x1[j] + A2[i,j]*x2[j]+ A3[i,j]*x3[j] for j=1:h) )

    @constraint(myModel, [i=1:h], R[i] - H[i] - CHD <= Z[i])
    @constraint(myModel, [i=1:h],-R[i] + H[i] + CHD <= Z[i])

    @constraint(myModel, [i=1+1:h], x1[i] + x2[i] + x3[i] <= 1 - (x1[i-1] + x2[i-1] + x3[i-1]))
    @constraint(myModel, [i=1:h-1], x1[i] + x2[i] + x3[i] <= 1 - (x1[i+1] + x2[i+1] + x3[i+1]))

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end

    return (JuMP.value.(x), JuMP.value.(R))
end

channel_data = readdlm("channel_data.txt")

distances = pushfirst!(haversine_distance(
    channel_data[1:end - 1, 1], 
    channel_data[1:end - 1, 2],
    channel_data[2:end, 1], 
    channel_data[2:end, 2],
), 0)

heights = channel_data[1:end, 3] 

distances_acc = accumulate(+, distances)
f = CubicSpline(heights, distances_acc)

CHD = 10
#D = push!(collect(0:250:distances_acc[end]), distances_acc[end])
D = collect(0:250:distances_acc[end])
H = f.(D)

K1 = [300 140 40]
K2 = [500 230 60]
K3 = [1000 400 70]

x, R = solveIP4(H, K1, K2, K3)

scatter(D[x .== 1], H[x .== 1])
plot!(f)