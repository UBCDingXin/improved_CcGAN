%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMCEP - Simulation tool for fluorescent cell populations.
% Based on the manuscript "Computational framework for simulating
% fluorescence microscope images with cell populations"
%
% Input:   (1) scale of one object
%          (2) variance which is used for defining different focus levels
%          (3) struct containing images for all objects
%          (4) struct containing binary images for all objects
% Output:  (1) struct for blurred images
%
% (C) Antti Lehmussola, 22.2.2007
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[RGB, BW, features] = simcep(num_cells)

%Read simulation parameters
simcep_options;

%Generate cells
disp('Generating objects...')
[object] = generate_objects(cell_obj,population,num_cells);

%Generate ideal image of cell population
disp('Generating ideal image...')
[image,bw] = generate_image(object,population);

%Generate measurement errors
disp('Measurement errors...')
[final] = generate_measurement(image,bw,measurement,population);


%Final RGB and binary image and requested object features
RGB = zeros([size(population.template) 3]);
BW = RGB;

if ~isempty(final.cytoplasm)
	RGB(:,:,1) = final.cytoplasm;
	BW(:,:,1) = bw.cytoplasm;
	features.cytoplasm = getfeatures(object.cytoplasm);
end

if ~isempty(final.subcell)
	RGB(:,:,2) = final.subcell;
	BW(:,:,2) = bw.subcell;
	features.subcell = getfeatures(object.subcell);
end

if ~isempty(final.nuclei)
	RGB(:,:,3) = final.nuclei;
	BW(:,:,3) = bw.nuclei;
	features.nuclei = getfeatures(object.nuclei);
end

%Compression artefacts
q = round(100*(1-measurement.comp));
if q > 0
	imwrite(RGB,'simcep_compression.jpg', 'Quality', q);
	RGB = imread('simcep_compression.jpg');
end
