%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for population level simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the size of the simulated image and part of the image where cells
% will be simulated. For example, ones(500), results as a 500 x 500 image
% where cells can be simulated in every part of the image.
population.template = ones(256);

% Amount of cells simulated in the image
population.N = 100;

% Amount of clusters
population.clust = 3;

% Probability for assiging simulated cell into a cluster. Otherwise the
% cell is uniformly distributed on the image.
population.clustprob = 0.0;

% Variance for clustered cells
population.spatvar = 0.05;

% Amount of allowed overlap for cells [0,1]. For example, 0 = no overlap
% allowed and 1 = overlap allowed.
population.overlap = 1;

% Is the overlap measured on nuclei (=1) level, or cytoplasm (=2) level
population.overlap_obj = 1; %Overlap: nuclei = 1, cytoplasm = 2


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for the measurement system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Energy of illumination compared to the energy of cells
measurement.illumscale = 1;

% Misalignment of illumination source in x and y direction
measurement.misalign_x = 0;
measurement.misalign_y = 0;

% Energy of autofluorescence compared to the energy of cells
measurement.autofluorscale = 0.05;

% Variance of noise for ccd detector
measurement.ccd = 0.0001;

% Amount of compression artefacts
measurement.comp = 0.0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cell level parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Cytoplasm

% Is cytoplasm included in the simulation ( 0 = no, 1 = yes)
cell_obj.cytoplasm.include = 0; %1

% Cytoplasm radius
cell_obj.cytoplasm.radius = 25;

% Parameters for random shape
cell_obj.cytoplasm.shape = [0.3 0.05];

% Parameters for texture: persistence, 1st octave, last octave, and
% intensity bias
cell_obj.cytoplasm.texture = [0.9 2 8 0.2];

%%% Nuclei (see cytoplasm parameters for details)
cell_obj.nucleus.include = 1;
cell_obj.nucleus.radius = 5; %%cell size default as 10
cell_obj.nucleus.shape = [0.1 0.1];
cell_obj.nucleus.texture = [0.5 2 5 0.2];

%%% Subcellular parts (modeled as objects inside the cytoplasm; note cytoplasm
%%% simulation needed for simulation of subcellular parts).

cell_obj.subcell.include = 0;

% Number of subcellular objects
cell_obj.subcell.ns = 4;

% Radius of single object
cell_obj.subcell.radius = 3; %3
cell_obj.subcell.shape = [0.1 0.1];
cell_obj.subcell.texture = [0.5 2 5 0.2];
