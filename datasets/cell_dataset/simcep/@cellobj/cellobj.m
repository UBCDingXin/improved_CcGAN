%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CELLOBJ Class constructor for cytoplasm objects
% The main class from which NUCLEI, CYTOPLASM, and SUBCELL classes are
% inherited.
%
% Input:  (1) coordinates for the objects
%         (2) index 
%         (3) radius of cell
%         (4) shape parameters
%         (5) texture parameters
% Output: (1) new cell object
%
% (C) 22.2.2007 Antti Lehmussola
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function c = cellobj(coords,ind,shape,object,feat)

c.coords = coords;
c.index = ind;
c.shape = shape;
c.object = object;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All information inserted in the features struct are available through
% output of SIMCEP. Good place to store information for validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c.features = feat;

c = class(c,'cellobj');
