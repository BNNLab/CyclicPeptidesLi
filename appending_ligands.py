# Neccesary imports
import os
from openbabel import pybel

# Define input and output directory
input_directory = "/nobackup/cm21sb/tetracyclic_new"
output_directory = input_directory

# Define file string
file_str = []

# Loop through all files in input directory
for file in os.listdir(input_directory):

    # Add all the mol2 files to one combined mol2 file
    if file.endswith('.mol2'):
        file_path = os.path.join(input_directory, file)

        with open(file_path, 'r') as infile:
            xyz_str = infile.readlines()
            file_str.extend(xyz_str)

# Write combined mol2 file
output_file = os.path.join(output_directory, "combined_ligands.mol2")
with open(output_file, 'w') as outfile:
    outfile.writelines(file_str)
