%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate nonuniform lighting using 2-dimensional 2nd degree polynomial
% Example: pol=[0 0 0 1 1 0 ];
% out=illumination([400 400],pol,0.5,[0 0]);
% (c) Jyrki Selinummi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Z=illumination(insize,center)
pol = [0 0 0 0 1 1];

center(1) = round(center(1)*insize(1));
center(2) = round(center(2)*insize(2));
[X,Y] = meshgrid(-insize(1)-center(1):2:insize(1)-center(1),...
    -insize(2)-center(2):2:insize(2)-center(2));

Z = pol(1)+pol(2).*X+pol(3).*Y+pol(4).*X.*Y+pol(5).*X.^2+pol(6).*Y.^2;

Z=Z-min(min(Z));
Z=1-Z/max(max(Z));
Z = Z(1:end-1,1:end-1);

