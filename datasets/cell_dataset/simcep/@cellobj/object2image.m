function[image,bw] = object2image(cellObj,image,bw)

Y = cellObj.coords(1);
X = cellObj.coords(2);

image = borders(image,size(cellObj.shape),0);
bw = borders(bw,size(cellObj.shape),0);

%Update coordinates for padded image
Y = Y + size(cellObj.shape,1);
X = X + size(cellObj.shape,2);

%Calculate location of the new object
Y1 = Y-round(size(cellObj.shape,1)/2)+1;
X1 = X-round(size(cellObj.shape,2)/2)+1;
Y2 = Y1 + size(cellObj.shape,1)-1;
X2 = X1 + size(cellObj.shape,2)-1;

image(Y1:Y2,X1:X2) = image(Y1:Y2,X1:X2) + cellObj.object;
bw(Y1:Y2,X1:X2) = bw(Y1:Y2,X1:X2) + cellObj.shape;

image = borders(image,size(cellObj.shape),1);
bw = borders(bw,size(cellObj.shape),1);

%Function for border padding
function[out] = borders(input,B,opt);

if opt == 0
	out = [zeros(B(1),size(input,2)+2*B(2)); ...
		zeros(size(input,1),B(2)) input zeros(size(input,1),B(2));...
		zeros(B(1),size(input,2)+2*B(2))];
else
	out = input(B(1)+1:size(input,1)-B(1),B(2)+1:size(input,2)-B(2));
end