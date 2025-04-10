import os
import pandas as pd

input_directory = "/nobackup/cm21sb/xtb_energy/sophie_ligands/sophie_ligands/jobs/"
# input_directory = "C://Users//cm21sb//OneDrive - University of Leeds//Year 4//Sophie Blanch//code//xtb_energy//"

codes = []
free_energies = []
# enthalpies = []

directory_list = os.listdir(input_directory)
print(directory_list)

for dir in directory_list:
    if dir.startswith("job"):
        subdir_path = os.path.join(input_directory, dir)

        print(subdir_path)

        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)

        print(files)

        for filename in files:
                file_path = os.path.join(subdir_path, filename)
                code = str(filename)[:4]

                print(file_path)

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

                # print(f"{code}: {energy} ")
                    # if "TOTAL ENTHALPY" in line:
                    #      enthalpy_line = line.strip()
                    #      enthalpy = enthalpy_line.split()[-3]
                    #      enthalpies.append(enthalpy)

# Make a data frame

df = pd.DataFrame({"Code": codes, "Gibbs Free Energy / Eh": free_energies})

print(df)
# df.to_csv("ligands_gibbs_free_energy.txt", index=False)
                
        #    -------------------------------------------------
        #   | TOTAL ENERGY             -128.023088620607 Eh   |
        #   | TOTAL ENTHALPY           -127.325123683811 Eh   |
        #   | TOTAL FREE ENERGY        -127.443798057052 Eh   |
        #   | GRADIENT NORM               0.000320859649 Eh/α |
        #   | HOMO-LUMO GAP               3.350250268934 eV   |
        #    -------------------------------------------------