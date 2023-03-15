using DelimitedFiles, Plots, LinearSolve

function findDistanceKM(matrix)
    R = 6371
    a,b = size(matrix)
    dist = zeros(a,1)
    for i in 1:a-1
        phi1 = matrix[i,1] * pi/180
        phi2 = matrix[i+1,1] * pi/180
        lambda1 = matrix[i,2] * pi / 180
        lambda2 = matrix[i+1,2] * pi / 180
        delta_lambda = (lambda2 - lambda1)
        delta_phi = (phi2 - phi1)
        Hav = sin(delta_phi / 2)^2 + cos(phi1) * cos(phi2) * sin(delta_lambda/2)^2
        dist[i+1] = R * (2 * atan(sqrt(Hav), sqrt(1 - Hav)))
    end
    return dist
end
function accDist(dist)
    acc = zeros(1,lastindex(dist))
    for i in 2:lastindex(dist)
        acc[i] = acc[i-1] + dist[i]
    end
    return acc 
end

function PolyFit(xvals, yvals, degree)
    # Returns the coefficients of a polylonial of degree lenght(x) - 1
    a = length(xvals)
    X = ones(a,degree)
    for i = 2:degree
        X[:,i] = xvals.^(i-1)
    end
    prob = LinearProblem(X,yvals)
    sol = solve(prob)
    return sol.u
end
function evalPoly(xvals, poly)
    yvals = zeros(Float64, length(xvals))
    idx = 0
    for i in eachindex(xvals)
        idx += 1
        pot = 0
        for j in eachindex(poly)
            yvals[idx] += (xvals[i]^(j-1)) * poly[j]
            pot +=1
        end
    end
    return yvals
end 
function createxvals(distanceVector, distance)
    max = findmax(distanceVector)[1]
    npoints = floor(Int ,max / distance)
    xvals = zeros(Float64 , npoints+1)
    for i in eachindex(xvals)
        xvals[i] = 0.25 * (i-1)
    end
    append!(xvals, max)
    return xvals
end


matrix = readdlm("channel_data.txt")
dist = findDistanceKM(matrix);
height = matrix[:,3]
accumulatedDist = accDist(dist);
plot(accumulatedDist', height, label = "Data", linewidth = 4)
xlabel!("Distance (km)")
ylabel!("Height (m)")
title!("Height as function of distance")
coef = PolyFit(accumulatedDist, height, 9)
xvals = createxvals(accumulatedDist, 0.25)
yvals = evalPoly(xvals, coef)
plot!(xvals, yvals, label = "Eget Fit", linewidth = 4)

