import os
import pandas as pd

input_directory = "/nobackup/cm21sb/pm6_energy/ligand_mopac/"
# input_directory = "C://Users//cm21sb//OneDrive - University of Leeds//Year 4//Sophie Blanch//code//pm6_energy//"


# Initialise lists
codes = []
enthalpies = []
entropies = []

# List input directory
files = os.listdir(input_directory)

# Loop through files in directory to extract enthalpies and entropies at 298 K
for filename in files:
        file_path = os.path.join(input_directory, filename)
        # To retain amino acid code
        code = str(filename)[:4]

        if filename.endswith(".aux"):

            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding = 'utf-8', errors = 'replace') as f:
                    lines = f.readlines()

                    for i, line in enumerate(lines):
                        if "ENTHALPY_TOT" in line and i + 1 < len(lines):
                            enthalpy = lines[i + 1].strip().split()[0]  # Extract first value from line after "ENTHALPY_TOT"
                        
                        if "ENTROPY_TOT" in line and i + 1 < len(lines):
                            entropy = lines[i + 1].strip().split()[0]  # Extract first value from line after "ENTROPY_TOT"

                    if enthalpy is not None and entropy is not None:
                        codes.append(code)
                        enthalpies.append(enthalpy)
                        entropies.append(entropy)                     

# Make a data frame
df = pd.DataFrame({"Code": codes, "Enthalpy / cal mol-1": enthalpies, "Entropy / cal K-1 mol-1": entropies})

# Convert entries to numbers
df["Enthalpy / cal mol-1"] = pd.to_numeric(df["Enthalpy / cal mol-1"], errors="coerce")
df["Entropy / cal K-1 mol-1"] = pd.to_numeric(df["Entropy / cal K-1 mol-1"], errors="coerce")

# Calculate Gibbs Free Energy
df['Gibbs Free Energy / cal mol-1'] = df["Enthalpy / cal mol-1"] - 298*df["Entropy / cal K-1 mol-1"]

df.to_csv("ligand_v1_gibbs_free_energy.txt", index=False)
