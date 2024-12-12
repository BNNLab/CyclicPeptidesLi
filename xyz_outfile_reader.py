import os
from openbabel import pybel

input_directory = "C:/Users/cm21sb/OneDrive - University of Leeds/Year 4/Sophie Blanch/code/tetracyclic_new/ligands"
output_directory = input_directory

for file in os.listdir(input_directory):
    filepath = os.path.join(input_directory, file)
    filename = os.path.splitext(file)[0]

    if file.endswith('.xyz'):

        with open(file, 'r') as infile:
            xyz_str = infile.readlines()
            xyz_str[1] = xyz_str[1][26:30] + '\n' 
            
            xyz_str = ''.join(xyz_str)

            mol = pybel.readstring("xyz", xyz_str)
            output_directory = os.path.join(
                input_directory, file.replace('.xyz', '.mol2')
            )
            mol.write("mol2", output_directory, overwrite=True)

            with open(output_directory, 'r') as mol2file:
                mol2_lines = mol2file.readlines()
                                        
            with open(output_directory, 'w') as mol2file:
                mol2file.writelines(mol2_lines)