# Necessary imports
import os
import numpy as np
import copy

if __name__ == "__main__":

    # Define input and output directories
    input_directory = "/nobackup/cm21sb/ligands_xyz_pm6/"
    output_directory = input_directory

    # Loop through directory to extract optimised geometry coordinates and add PM6 commandline
    for filename in os.listdir(input_directory):
        input_file = os.path.join(input_directory, filename)

        if filename.endswith('.xyz'):

            if os.path.isfile(input_file):
                with open(input_file, 'r', encoding = 'utf-8', errors = 'replace') as f:
                    lines = f.readlines()

                code = str(filename)[:4]
                coords = lines[2:]
                lines[0] = f"PM6-D3H4X FORCE XYZ EPS=78.4 RSOLV=1.3 PRECISE GNORM=0.01 GEO-OK LET THERMO AUX\n"
                lines[1] = f"{code}\n\n"

            # Define output file
            output_file = os.path.join(output_directory, f"{code}.mop")

            # Write PM6 input files
            try:
                with open(output_file, 'w') as f:
                    f.writelines(lines)
                    
            except Exception as e:
                print(f"Error writing file for molecule {code}: {e}")
