%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTICS Function for generating optical aberrations
% Input:   (1) scale of one object
%          (2) variance which is used for defining different focus levels
%          (3) struct containing images for all objects
%          (4) struct containing binary images for all objects
% Output:  (1) struct for blurred images
%
% (C) Antti Lehmussola, 22.2.2007
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[blurred] = optics(R,v,image, bw)

blurred.nuclei = [];
blurred.cytoplasm = [];
blurred.subcell = [];

%Define the overall area of cell
if ~isempty(bw.nuclei)
	objects = bw.nuclei;
	if ~isempty(bw.cytoplasm)
		objects = objects | bw.cytoplasm;
		if ~isempty(bw.subcell)
			objects = objects | bw.subcell;
		end
	end
else
	objects = bw.cytoplasm;
	if ~isempty(bw.subcell)
		objects = objects | bw.subcell;
	end
end


L = bwlabel(objects);
D = zeros(size(objects));

%Define size of the gaussian kernel, here we use half of object scale
wl = round(R/2);

%Make size odd
if mod(wl,2) == 0
    wl = wl + 1;
end

    
%Generate depth information

%Depth bias
db = 0.0;

for ind = 1:max(L(:))
    r = db+randn*v;
    D(L == ind) = abs(r);
end

%Blur images
if v > 0
	if ~isempty(image.nuclei)
		blurred.nuclei = blurimage(image.nuclei,D,wl,.5,5);
	end
	
	if ~isempty(image.cytoplasm)
		blurred.cytoplasm = blurimage(image.cytoplasm,D,wl,.5,5);
	end
	
	if ~isempty(image.subcell)
		blurred.subcell = blurimage(image.subcell,D,wl,.5,5);
	end
else
	blurred = image;
end

