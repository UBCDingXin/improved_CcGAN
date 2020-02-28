%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADDOBJECT Function for checking if the generated object is in valid place
% Input:  (1) Object from cellobj class
%         (2) Cell array defining location for each object
%         (3) Allowed amount of overlap
% Output: (1) Updated location cell array
%         (2) Parameter telling if the object location was valid
%
% (C) 22.2.2007 Antti Lehmussola
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[locations,status] = add_object(cellObj,locations,L)

Y = cellObj.coords(1);
X = cellObj.coords(2);

%Check if the coordinates are outside the image
if Y > size(locations,1) | Y < 1 | X > size(locations,2) | X < 1
	status = 0;
	return
end

template = zeros(size(locations));
template = borders(template,size(cellObj.shape),0);

%Update coordinates for padded image
Y = Y + size(cellObj.shape,1);
X = X + size(cellObj.shape,2);

%Calculate location of the new object
Y1 = Y-round(size(cellObj.shape,1)/2)+1;
X1 = X-round(size(cellObj.shape,2)/2)+1;
Y2 = Y1 + size(cellObj.shape,1)-1;
X2 = X1 + size(cellObj.shape,2)-1;

%Template for the new object
template(Y1:Y2,X1:X2) = cellObj.shape;
template = borders(template,size(cellObj.shape),1);
template = logical(template);

%Objects already in that location
objs = [locations{template}];
objs_uniq = unique(objs);

%Size of the object
sum_object = sum(sum(cellObj.shape));

%Go through all overlapping objects
for ind = 1:length(objs_uniq)
	
	%Is any of the objects overlapping too much with the current object
	R = sum(objs == objs_uniq(ind))/sum_object;
	if R > L
		status = 0;
		return
	end
	
	%Is the current object overlapping too much with any other objects 
	R = sum(objs == objs_uniq(ind))/sum([locations{:}] ==  objs_uniq(ind));
	if R > L
		status = 0;
		return
	end
	
end

%Update object into cell array
tmp = locations(template);
tmp = cellfun(@(x)cat(2,x,cellObj.index),tmp,'UniformOutput', false);
locations(template) = tmp;
status = 1;

%Function for border padding
function[out] = borders(input,B,opt);

if opt == 0
	out = [zeros(B(1),size(input,2)+2*B(2)); ...
		zeros(size(input,1),B(2)) input zeros(size(input,1),B(2));...
		zeros(B(1),size(input,2)+2*B(2))];
else
	out = input(B(1)+1:size(input,1)-B(1),B(2)+1:size(input,2)-B(2));
end