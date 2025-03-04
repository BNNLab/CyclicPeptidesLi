import os
import pandas as pd

input_directory = "/nobackup/cm21sb/sandwich_xyz/mopac/three_bonds_mopac/"
# input_directory = "C://Users//cm21sb//OneDrive - University of Leeds//Year 4//Sophie Blanch//code//tetracyclic_new//sandwiches//"

codes = []
energies = []

for filename in os.listdir(input_directory):
    file = os.path.join(input_directory, filename)
    code = str(filename)[:4]


    if filename.endswith(".out"):

        if os.path.isfile(file):
            with open(file, 'r', encoding = 'utf-8', errors = 'replace') as f:
                for line in f:                      
                    if "HEAT OF FORMATION" in line:
                        energy_line = line.strip()
                        energy = energy_line.split()[-2]
                        energies.append(energy)
                        codes.append(code)
                        break

# Make a data frame

df = pd.DataFrame({"Code": codes, "Heat of Formation / kcal mol-1": energies})
df.to_csv("heat_of_formation_results_three_bonds.txt", index=False)
                
