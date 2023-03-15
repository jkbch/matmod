using DelimitedFiles, Plots, DataInterpolations

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

