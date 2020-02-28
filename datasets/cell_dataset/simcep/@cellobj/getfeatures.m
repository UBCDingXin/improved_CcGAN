%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GETFEATURES Output all information stored in the features struct of
% objects from cellobj class.
% Input:  (1) object from cellobj class
% Output: (1) struct containing arrays with information from all objects
%
% (C) 22.2.2007 Antti Lehmussola
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[out] = getfeatures(cellObj);

fn = fieldnames(cellObj(1).features);

for fn_ind = 1:length(fn)

	tmpdata = cell(size(cellObj));
	for ind = 1:length(cellObj)
		tmpdata{ind} = eval(['cellObj(' num2str(ind) ').features.' fn{fn_ind}]);
	end
		eval(['out.' fn{fn_ind} '= tmpdata;']);
		
end