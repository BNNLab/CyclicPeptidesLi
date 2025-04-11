# Neccessary imports
import os
import pandas as pd

# Define input directory
input_directory = "/nobackup/cm21sb/xtb_energy/sophie_ligands/sophie_ligands/jobs/"
# input_directory = "C://Users//cm21sb//OneDrive - University of Leeds//Year 4//Sophie Blanch//code//xtb_energy//"

# Establish empty lists for data
codes = []
free_energies = []
enthalpies = []

# Loop through subdirectories to find the total free energies of each .out file
directory_list = os.listdir(input_directory)
print(directory_list)

for dir in directory_list:
    if dir.startswith("job"):
        subdir_path = os.path.join(input_directory, dir)

        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)

        for filename in files:
                file_path = os.path.join(subdir_path, filename)
                code = str(filename)[:4]

                if filename.endswith(".out"):

                    if os.path.isfile(file_path):
                        with open(file_path, 'r', encoding = 'utf-8', errors = 'replace') as f:
                            lines = f.readlines()

                            final_30_lines = lines[-30:]

                            for line in final_30_lines:                      
                                if "TOTAL FREE ENERGY" in line:
                                    energy_line = line.strip()
                                    energy = energy_line.split()[-3]
                                    free_energies.append(energy)
                                    codes.append(code)
                                    break


# Make a data frame
df = pd.DataFrame({"Code": codes, "Gibbs Free Energy / Eh": free_energies})

df.to_csv("ligands_gibbs_free_energy.txt", index=False)
                
