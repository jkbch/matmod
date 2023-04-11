% Anton document hvor han prøver at kører ting

A = load("testImage.mat","im");

%im = im;

%wood_log = A.im*100;

sample = downsample(wood_log,100);
sample = downsample(sample',100);

imshow(sample)
%imshow(wood_log)
axis on;
