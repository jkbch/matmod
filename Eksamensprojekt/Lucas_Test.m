clear all
close all
clc
addpath('Matlabfunctions')
im = load('testImage.mat');
im = im.im;

factor = 100;
n = 5000 / factor;

im2 = downsample(im, factor);
im_down = downsample(im2', factor);

imshow(im_down * 200)
axis on
vec = 0:1:180;
[A,b,x, theta, p, d] = paralleltomo(5000 / factor,vec,[],[],0);
b

%%
b = A * reshape(im_down,[],1);

b = b + randn(length(b),1) .* b / 1000 + randn(length(b),1) / 1000;

X = mldivide(A,b);

%%
figure(1)
imagesc(reshape(X,5000 / factor, 5000 / factor)*10);


%%
I0 = 60; % keV
mu_wood = 0.1844;
mu_iron = 1.205;
mu_bismuth = 5.233;

%%
testImage = generateTestImage(n, true);
b_TestImage = A * reshape(testImage,[],1);

b_TestImage_noisy = imnoise(b_TestImage, "poisson");
X_TestImage = mldivide(A,b_TestImage_noisy);

figure(2)
imagesc(reshape(X_TestImage,5000 / factor, 5000 / factor)*10)

%% 

