function pic = addNoise(pic)
[n,m] = size(pic);
pic = pic + randn(n,m) .* pic / 1000 + randn(n,m) / 1000;