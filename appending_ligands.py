import os
from openbabel import pybel

input_directory = "/nobackup/cm21sb/tetracyclic_new"
output_directory = input_directory

file_str = []

for file in os.listdir(input_directory):
    
    if file.endswith('.mol2'):
        file_path = os.path.join(input_directory, file)

        with open(file_path, 'r') as infile:
            xyz_str = infile.readlines()
            file_str.extend(xyz_str)
            
output_file = os.path.join(output_directory, "combined_ligands.mol2")
with open(output_file, 'w') as outfile:
    outfile.writelines(file_str)