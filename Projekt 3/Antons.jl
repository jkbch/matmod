using GLPK, Cbc, JuMP, SparseArrays, DelimitedFiles, Plots

function findDistance(matrix,R)
    # matrix - matrix of points we want to calculate the distance between
    # R - The radius of the sphere we are working on
    a,b = size(matrix);
    dist = zeros(a,1);

    for i in 2:a-1
        phi1 = matrix[i,1]*pi/180
        phi2 = matrix[i+1,1]*pi/180

        lambda1 = matrix[i,2]*pi/180
        lambda2 = matrix[i+1,2]*pi/180

        delta_phi = sqrt((phi1-phi2)^2)

        delta_lambda = sqrt((lambda1-lambda2)^2)

        havesine = sin(delta_phi/2)^2+cos(phi1)*cos(phi2)*sin(delta_lambda/2)^2;
        c = 2*atan(sqrt(havesine),sqrt(1-havesine))

        dist[i] = R*c
    end

    return dist

end

function accummulatedDist(dist,matrix)
    a,b = size(matrix);
    accDist = zeros(a,1)

    for i in 2:a
        accDist[i] = accDist[i-1]+dist[i]
    end-
    return accDist
end
matrix = readdlm("channel_data.txt")
#matrix[:,1] = matrix[:,1] .- matrix[1,1];
#matrix[:,2] = matrix[:,2] .- matrix[1,2];

R = 6371;
dist = findDistance(matrix,R);
accDist = accummulatedDist(dist,matrix);
matrix[:,3]

plot(matrix[:,3],accDist)
