%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GENERATE_OBJECTS Function for generating required amount of objects
% Input:  (1) struct for cell level parameters
%         (2) struct for population level parameters
% Output: (1) struct containing arrays for all different object types
%
% (C) 22.2.2007 Antti Lehmussola
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[out] = generate_objects(cell_obj,population,num_cells)

nuclei_vector =[];
cytoplasm_vector =[];
subcell_vector = [];

% Limit for unsuccesfully trying to generate new objects
FAIL_LIMIT = 50;

locations = cell(size(population.template));

%Generate cluster center coordinates ~ U(0,dimensions)
cluster = [randi(1,population.clust,size(population.template,1))+1;...
    randi(1,population.clust,size(population.template,2))+1];

%Generate requited amount objects
for ind = 1:num_cells %population.N
	%Status for succesful object generation
	status = 0;
	
	%Amount of unsuccesfully generated objects
	failed = 0;
	
	%Stay here until object is generated succesfully
	while status == 0
		
		%Assign coordinates for each object
		
		if rand < population.clustprob
			%Select randomly some cluster center
			C = randi(1,1,population.clust)+1;
			
			%Random coordinates for the object ~ N(0,dimensions)
			Y = round(cluster(1,C)+randn*population.spatvar*size(population.template,1));
			X = round(cluster(2,C)+randn*population.spatvar*size(population.template,2));
		else
			Y = randi([1,size(population.template,1)],1,1);
			X = randi([1,size(population.template,2)],1,1);
		end
		
		%Overlap controlled in nucleus level
		if population.overlap_obj == 1		
			if cell_obj.nucleus.include == 0
				error('Impossible to control nuclei overlap since nuclei simulation is not selected');
			end
			
			%Generate new nucleus
			n = nucleus([Y X], ind, cell_obj.nucleus.radius,cell_obj.nucleus.shape,...
				cell_obj.nucleus.texture);
			[locations,status] = add_object(n,locations,population.overlap);
	
			if status == 0
				continue;
			end
			nuclei_vector = [nuclei_vector n];
			
		%Overlap controlled in cytoplasm level	
		elseif population.overlap_obj == 2
			if cell_obj.cytoplasm.include == 0
				error('Impossible to control cytoplasm overlap since cytoplasm simulation is not selected');
			end
			
			%Generate new cytoplasm
			c = cytoplasm([Y X], ind, cell_obj.cytoplasm.radius,cell_obj.cytoplasm.shape,...
				cell_obj.cytoplasm.texture);
			[locations,status] = add_object(c,locations,population.overlap);
			if status == 0
				continue;
			end
			cytoplasm_vector = [cytoplasm_vector c];
		end	
		
		%Image is too full to add new objects
		if failed > FAIL_LIMIT
			warning(['Not enough space for new objects. Only ' num2str(eval('ind')-1) ...
				' objects were generated. Wait until simulation is finished...']);
			break;
		end
		
		failed = failed + 1;
		
	end
	
	%Limit for how many times we try to generate one cell
    if failed > FAIL_LIMIT
        break;
	end
	
	%Generate other required objects
	if cell_obj.cytoplasm.include == 1 & population.overlap_obj == 1
		c = cytoplasm([Y X], ind, cell_obj.cytoplasm.radius,cell_obj.cytoplasm.shape,...
			cell_obj.cytoplasm.texture);
		cytoplasm_vector = [cytoplasm_vector c];
	end
	if cell_obj.nucleus.include == 1 & population.overlap_obj == 2
		n = nucleus([Y X], ind, cell_obj.nucleus.radius,cell_obj.nucleus.shape,...
			cell_obj.nucleus.texture);
		nuclei_vector = [nuclei_vector n];
	end
	
	if cell_obj.subcell.include == 1 & cell_obj.cytoplasm.include == 1
		s = subcell([Y X],ind,cell_obj.cytoplasm.radius/10,cell_obj.subcell.ns,c);
		subcell_vector = [subcell_vector s];
	end
	
end

out.nuclei = nuclei_vector;
out.cytoplasm = cytoplasm_vector;
out.subcell = subcell_vector;
