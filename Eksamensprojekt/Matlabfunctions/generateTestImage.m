function pic = generateTestImage(n, varargin)

if nargin == 1
    noise = false;
else 
    noise = varargin{1};
end
p_wood = 0.0008;
p_iron = 0.0037;
p_bismuth = 0.0540;

x = -n/2:n/2-1;
y = -n/2:n/2-1;
[xx, yy] = meshgrid(x,y);
pic = zeros(size(xx));
pic((xx.^2+yy.^2)<(n/2)^2) = p_wood; 

n_wood = pic(pic == p_wood);
bismuth = randi(length(n_wood),1);
iron = randi(length(n_wood),1);
while iron == bismuth
    iron = randi(length(n_wood),1);
end
disp(bismuth)
disp(iron)
v = pic(pic == p_wood);
v(bismuth) = p_bismuth;
v(iron) = p_iron;

pic(pic == p_wood) = v;
if noise
    pic  = addNoise(pic);
end