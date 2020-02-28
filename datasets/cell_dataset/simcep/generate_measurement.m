%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GENERATE_MEASUREMENT Function for generating errors based on the
% measurement system. 
% Input:  (1) struct with ideal images
%         (2) struct with binary images 
%         (3) measurement level parameters
%         (4) population level parameters
% Output: (1) ideal images corrupted with measurement errors
%
% (C) 22.2.2007 Antti Lehmussola
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[final] = generate_measurement(image,bw,measurement,population)
O_S = 15;
O_V = 0.5;

%Background autofluorescence
bkg_autofluor = autofluor(size(population.template),1,1,2,0);
ba = sum(bkg_autofluor(:).^2);

%Uneven illumination
bkg_ill = illumination(size(population.template),[measurement.misalign_x measurement.misalign_y]);
bi = sum(bkg_ill(:).^2);

if ~isempty(image.cytoplasm)
	ie = sum(image.cytoplasm(:).^2);
else
	ie = sum(image.nuclei(:).^2);
end

ill_scale = sqrt(measurement.illumscale*ie/bi);
autofluor_scale = sqrt(measurement.autofluorscale*ie/ba);

%Optical aberrations
blurred = optics(O_S,O_V,image,bw);

final.nuclei = [];
final.cytoplasm = [];
final.subcell = [];

%Add CCD noise
if ~isempty(blurred.nuclei)
	final.nuclei = imnoise(blurred.nuclei + ill_scale*bkg_ill + autofluor_scale*bkg_autofluor,....
		'gaussian',0,measurement.ccd);
end

if ~isempty(blurred.cytoplasm)
	final.cytoplasm = imnoise(blurred.cytoplasm + ill_scale*bkg_ill + autofluor_scale*bkg_autofluor,....
		'gaussian',0,measurement.ccd);
end

if ~isempty(blurred.subcell)
	final.subcell = imnoise(blurred.subcell + ill_scale*bkg_ill + autofluor_scale*bkg_autofluor,....
		'gaussian',0,measurement.ccd);
end
