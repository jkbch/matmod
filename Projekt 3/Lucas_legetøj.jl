using DelimitedFiles
#dataMatrix = [];
#open("channel_data.txt") do f
#    line = 0
#    while !eof(f)
#        data[data; readline(f)]
#    end
#end
#print(data)
matrix = readdlm("channel_data.txt")
matrix[:,1] = matrix[:,1] .- matrix[1,1];
matrix[:,2] = matrix[:,2] .- matrix[1,2];
print(matrix)
