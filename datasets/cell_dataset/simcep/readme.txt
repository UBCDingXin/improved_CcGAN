-------------------------------------------

GETTING STARTED WITH SIMCEP SIMULATION TOOL

-------------------------------------------


INTRODUCTION
------------

SIMCEP is a simulation tool providing the functionality presented in the manuscript "Computational framework for simulating fluorescence microscopy images with cell populations". The tool is implemented with Matlab and all source codes are freely available under terms of GNU General Public License. The motivation behind the tool is to allow a hands-on experience on the simulation methodology presented in the manuscript. Moreover, it provides a modular tool for future research.

GETTING STARTED
---------------

1) For compiling mex function, run 

	>> make_simcep

2) For example simulation, run

	>> [image,binary,features] = simcep;

3) Visualize simulated image

	>> imshow(image,[]);

4) For modifying simulation parameters, 

	>> edit simcep_options.m


CONTACT
-------

Antti Lehmussola
Institute of Signal Processing
Tampere University of Technology
PO Box 553
33101 Tampere
FINLAND
lehmusso@cs.tut.fi

----------------------------------------
http://www.cs.tut.fi/sgn/csb/simcep/
----------------------------------------