% Anton document hvor han prøver at kører ting

%% Initialize and downsample
im = load("testImage.mat");

wood_log = im.im;
scale = 200;

sample = downsample(wood_log,100);
sample = downsample(sample',100);


%% Show image sample
imshow(sample)
%imshow(wood_log)
axis on;

%% Generate A matrix

N = 50;
%theta = [0,90];
A1 = paralleltomo(N);

%% Use the A matrix for something
sigma = 0.001;
[n,m] = size(A1);
noise = randn(n,1)*sigma;
x = reshape(sample,[],1);
b = A1*x+noise;

C = A1\b;

C = reshape(C,50,50);

imshow(C*scale)
axis on




