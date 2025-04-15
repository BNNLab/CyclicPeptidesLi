# **<p align="center">README Document to Accompany Final Report</p>**
## <p align="center"> Explanation of the Code on GitHub</p>

## **1.	Methodology**
### **1.1**	
The following files are used to carry out the steps discussed in **3.2 Cyclic Peptide Ligands**.  

**tetracyclic_generator.py**  
- Produces xyz co-ordinates of all cyclic peptides from amino acid SMILES  

**xtb_file_generator.py**  
- Processes the co-ordinates to produce xTB input files  

### **1.2**	
The following files are used to carry out the steps discussed in **3.3 Lithium Sandwich Complexes**.

**xyz_outfile_reader.py**  
- Processes the optimised xyz out files produced by xTB to produce mol2 files

**appending_ligands.py**  
- Combines all the individual mol2 files into one long file

**lithium_sandwiches.py**  
- Processes the combined mol2 file of ligands and constructs the first orientation of complexes producing a combined mol2 file of sandwich complexes
- *Relies on the CCDC miniconda python 3.7.12 environment

**sandwich_maker_v2.py**  
- Processes the combined mol2 file of ligands and constructs the second orientation of complexes producing a combined mol2 file of sandwich complexes
- *Relies on MoleculeHandler.py

**sandwich_maker_v3.py**  
- Processes the combined mol2 file of ligands and constructs the third orientation of complexes producing a combined mol2 file of sandwich complexes
- *Relies on MoleculeHandler.py

**xtb_file_generator.py**  
- Processes the combined mol2 file of sandwich complexes to produce xTB input files for each complex

### **1.3**	
The following files are used to carry out the steps discussed in **3.4 Energy & Thermochemistry Calculations**.

**pm6_file_generator.py**  
- Processes the optimised co-ordinates of the structures, ligands and complexes, and produces PM6 input files 

**pm6_energy_finder.py**  
- Processes the aux files produced by MOPAC, locating the entropy and enthalpy terms calculated at 298 K and calculating the Gibbs free energy, saving the data to a csv

### **1.4**	
The following files are used to carry out the steps discussed in **3.6 Basis Set Comparison**.

**xtb_submit.py**  
- Submits the optimised co-ordinates of the structures, ligands and complexes, via the xTB executable

**xtb_energy_finder.py**  
- Processes the xTB out files, locating the calculated total free energy term, saving the data to a csv

## **2.	Analysis**
### **2.1**	
The following files are used to carry out the analysis in **4.3 Geometry Analysis** and **4.4 Binding Energies**.

**li_bond_filtering.py**  
- Processes optimised co-ordinates, calculates Li-O Euclidean distances and sorts the files into different directories based on the number of Li-O distances less than 2.5 Å 

**pm6_energy_analysis.ipynb**  
- Processes all the PM6 binding energy data, including number of Li-O bonds, and produces the graphs seen

### **2.2**	
The following files are used to carry out the analysis in **4.5 Method Comparison**.

**xtb_binding_energy_analysis.ipynb**  
- Processes the xTB binding energy data for orientation 1, including number of Li-O bonds, and produces the graphs seen

**pm6_binding_energy_v1.ipynb**  
- Processes the PM6 binding energy data for orientation 1, including number of Li-O bonds, and produces the graphs seen

### **2.3**	
The following files are used to carry out the analysis in **4.5.1 Imaginary Frequency Analysis**.

**negative_frequency_extractions_pm6.py**  
- Reads the file containing the 100 random file names to check, searches the directory for the matching aux files and saves a csv of the imaginary frequencies found

**negative_frequency_extractions_xtb.py**  
- Reads the file containing the 100 random file names to check, searches the directory for the matching out files and saves a csv of the imaginary frequencies found

**neg_freq_analysis.ipynb**  
- Processes the imaginary frequency data for xTB and PM6 and produces the graphs seen
	
	
