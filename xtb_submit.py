# Written by Alister Goodfellow for running on AIRE- thank you!
# Necessary imports
import os
import subprocess
import pandas as pd

# Define input directories
inp_directory = 'inp_files'
xyz_directory = 'tetracyclic_sandwiches_xyz_water'
xyz_files = sorted([f for f in os.listdir(xyz_directory) if f.endswith(".xyz")])
if not xyz_files:
    print("No XYZ files found in the directory.")
    raise

#inp_files = [f for f in os.listdir(inp_directory) if f.endswith(".inp")]

df = pd.read_csv('charges.csv', index_col="Code")

# Define outfile names
for xyz in xyz_files:
    base_name = os.path.splitext(xyz)[0]
    out_file = os.path.join(xyz_directory, base_name + '.out')
    if xyz.endswith('_xtbopt.xyz'):
        continue
    if os.path.exists(out_file):
        print(f"Skipping {base_name} because the output file already exists.")
        continue

    # Extract total charges from list of overall charges
    charge = df.loc[base_name,"Charge"]
    print(f'{base_name} running')
    xyz_path = os.path.join(xyz_directory, xyz)
    xyz_file = base_name + '.xyz'
    # inp_file = base_name + '.inp'
    xyz_out = base_name + '_xtbopt.xyz'


    # Run xtb frequency calculation
    xtb_command = [f"/home/bngroup/BNGroup/Sophie/xtb-dist/bin/xtb {xyz_directory + '/' + base_name}.xyz -P 4 --charge {charge} --uhf 0 --ohess --alpb water thermo --temp 298 > {xyz_directory + '/' + base_name}.out" ]

    subprocess.run(" ".join(xtb_command), shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print(f"{xyz} complete")

    try:
        os.rename('xtbopt.xyz', xyz_directory + '/' + xyz_out)
    except Exception as e:
        print(f"Error moving file: {e}")

    # Remove temporary files
    tmpfiles=['charges', 'g98.out', 'hessian', 'vibspectrum', 'wbo', 'xtbopt.log', '.xtboptok', 'xtbopt.xyz', 'xtbrestart', 'xtbtopo.mol']
    for file in tmpfiles:
        try:
            if os.path.exists(file):
                os.remove(file)
                #print(f"Removed temporary file: {file}")
        except Exception as e:
            print(f"Error removing file {file}: {e}")

    raise
