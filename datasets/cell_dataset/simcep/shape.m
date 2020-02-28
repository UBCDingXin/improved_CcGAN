%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SHAPE Function for random shape model (for details see manuscript)
% Input:  (1) shape parameter (alpha)
%         (2) shape parameter (beta)
%         (3) radius of object
% Output: (1) shape as a binary image
%
% (C) 22.2.2007 Antti Lehmussola
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[out] = shape(s1,s2,r);

step = 0.1;
t = (0:step:1)'*2*pi;

r1 = rand(size(t))-0.5;
r2 = rand(size(t))-0.5;
t1 = s1.*(2*rand(size(t))-1)+sin(t+s2.*(2*rand(size(t))-1));
t2 = s1.*(2*rand(size(t))-1)+cos(t+s2.*(2*rand(size(t))-1));
t1(end) = t1(1);
t2(end) = t2(1);

object = [t2';t1'];

pp_nuc = cscvn(object);
object = ppval(pp_nuc, linspace(0,max(pp_nuc.breaks),1000));
	
object = r*object;

object(1,:) = object(1,:) - min(object(1,:));
object(2,:) = object(2,:) - min(object(2,:));
object = round(object)+1;

I = zeros(max(round(object(1,:))),max(round(object(2,:))));
BW = roipoly(I,object(2,:),object(1,:));
out = BW;