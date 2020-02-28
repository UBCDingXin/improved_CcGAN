%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBCELL Class constructor for subcellular objects
% Input:  (1) coordinates for the objects
%         (2) index 
%         (3) radius of single subcellular object
%         (4) amount of subcellular objects
%         (5) the cytoplasm
% Output: (1) new subcellular object
%
% (C) 22.2.2007 Antti Lehmussola
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function c = subcell(coords,ind,radius,N,cell_obj)
% SUBCELL Subcell class constructor. 

shape_temp = zeros(size(getshape(cell_obj)));
object_temp = zeros(size(getshape(cell_obj)));

%The coordinates for cytoplasm
[y,x] = find(getshape(cell_obj) == 1); %With zero you will easily get outside structures

%Generate required amount of particles
for ind = 1:N
	
	%Shape and texture fixed here
	S = shape(0.1,0.1,radius);
	T = texture(size(S),3,1,4,0);
	
	%Uniformly distributed particles
	idx = randint(1,1,length(y))+1;
	y2 = y(idx);
	x2 = x(idx);
	
	object_temp = borders(object_temp,size(S),0);
	shape_temp = borders(shape_temp,size(S),0);
	
		
	%Update coordinates for padded image
	y2 = y2 + size(S,1);
	x2 = x2 + size(S,2);
	
	%Calculate location of the new object
	Y1 = y2-round(size(S,1)/2)+1;
	X1 = x2-round(size(S,2)/2)+1;
	Y2 = Y1 + size(S,1)-1;
	X2 = X1 + size(S,2)-1;
	
	%Insert new object
	shape_temp(Y1:Y2,X1:X2) = shape_temp(Y1:Y2,X1:X2) + S;
	object_temp(Y1:Y2,X1:X2) = object_temp(Y1:Y2,X1:X2) + T;
	
	%Remove padded borders
	shape_temp = borders(shape_temp,size(S),1);
	object_temp = borders(object_temp,size(S),1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All information inserted in the features struct are available through
% output of SIMCEP. Good place to store information for validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

features.coords = coords;
features.area = sum(S(:))/N; %Average size for particles

a = cellobj(coords,ind,shape_temp,object_temp,features);
c.param1 = 0;
c = class(c,'subcell',a);

%Function for border padding
function[out] = borders(input,B,opt);

if opt == 0
	out = [zeros(B(1),size(input,2)+2*B(2)); ...
		zeros(size(input,1),B(2)) input zeros(size(input,1),B(2));...
		zeros(B(1),size(input,2)+2*B(2))];
else
	out = input(B(1)+1:size(input,1)-B(1),B(2)+1:size(input,2)-B(2));
end

