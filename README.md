# Graphene plasmons
Python scripts to calculate plasmonic band structures in graphene using Meep FDTD simulation software (https://github.com/NanoComp/meep).

**Script 1: "meep\_code.py"**
The first script is where the actual simulation is executed with the use of [Meep](https://meep.readthedocs.io) FDTD simulation software. The python script, when ran without any alterations, will calculate the data needed to create the band structure of a single sheet of graphene with hexagonal holes inside a uniform magnetic field of 4 Tesla. To make it more versatile, the possibility of switching to a geometry with h-BN and SiO2 substrates and gold islands is also provided. Additionally, one can choose to collect data to reconstruct full eigen fields, but be aware that this generally requires a lot more memory. These alterations can be made by simply commenting and uncommenting the relevant pieces of code.
With the exception of Meep and its dependencies, all required python 3 modules can be installed using the "pip install" command. To install a working Meep environment, please refer to the [Meep installation section](https://meep.readthedocs.io/en/latest/Installation/) of their website for more information. Also, make sure the directories "$data\$" (output folder for data) and "data\uc_harm\$" (output folder for eigen field data) exist in the folder containing the script file.

**Script 2: "raw\_analyse.py"**
The second script requires the raw data file (data\raw_field1.dat) created by the first script. It will perform a harmonic inversion, using the tool [harminv](https://github.com/NanoComp/harminv), to extract harmonic modes and subsequently create an energy band diagram.

**Script 3: "get\_unk.py"**
The third script can be used to reconstruct the eigen fields belonging to the eigen energies calculated with the 2nd script (raw_analyse.py). This will require the data files "data\uc_harm#.dat", where # is the number corresponding to the specified k-vector, which can be created with the first script (meep_code.py). In order to do this, you will need to add the line "mp.after_sources(UNK)" to the sim.run() function of the first script and run the simulation in single k-vector mode (for memory reasons). 

In order to run simulations on a cluster computer using Slurm workload manager three sample bash scripts have been added for convenience (Not needed for serial execution).
