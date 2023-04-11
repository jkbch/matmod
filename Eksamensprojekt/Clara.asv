clear all
close all

%Exam project 
load('testImage.mat','im');
%Img = imread('testImage.mat');


Img = downsample(transpose(downsample(im*200,100)),100);
%Img = imresize(im,1/100,'bilinear');

%figure;
%imshow(transpose(Img))
%axis on


A=paralleltomo(50);
x=reshape(Img,[],1);

b=A*x;

rand=(randn(length(b),1).*0.1);

b=b+rand;

%Reconstructing Img
C=A\b;
C=reshape(C,50,50);

imshow(C.')



