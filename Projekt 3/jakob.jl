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

scatter(distances_acc, heights)
plot!(f)


function constructA(H,K)
    n = length(H)
    A = diagm(fill(K[1], n))

    for (i, k) in enumerate(K[2:end])
        A += diagm(i => fill(k, n-i), -i => fill(k, n-i))
    end

    return A
end

H = f.(0:250:(distances_acc[end]))
K = [300 140 40]
A = constructA(H,K)

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

solveIP(H,K)
