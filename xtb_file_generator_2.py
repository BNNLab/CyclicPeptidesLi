# Necessary imports
import os
import numpy as np
import copy

if __name__ == "__main__":

    input_file = "/nobackup/cm21sb/tetracyclic_sandwich_xtb/combined_sandwiches.mol2"
    output_directory = "/nobackup/cm21sb/tetracyclic_sandwich_xtb"

    # Read the combined sandwiches mol2 file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Define sections to separate each molecule
    sections = []
    current_section = []
    # Separate by the first line of the individual mol2 files
    for line in lines:
        if "@<TRIPOS>MOLECULE" in line:
            if current_section:
                sections.append(current_section)
            current_section = [line]
        else:
            current_section.append(line)
    
    if current_section:
        sections.append(current_section)
    
    for section in sections:    

        # Extract co-ordinates block
        no_of_atoms = int(section[2][3:6])
        last_line_of_coords = no_of_atoms + 9
        parts = section[9:last_line_of_coords]

        total_charge = 0
        charges = []

        # Sum up charges in the final column of the mol2 file to get total charge
        for part in parts:
            columns = part.split()
            charges.append(float(columns[-1]))
        total_charge = round(sum(charges))

        # Extract amino acid code
        molecule_identifier = section[1].strip()

        coords = section[9:last_line_of_coords]
        coords_2 = []

        # Remove additional mol2 file information from co-ordinate block
        for line in coords:
            line = line[:40]
            line = line[7:]
            atom = str(line[0])
            line = atom + str(line[6:])
            coords_2.append(line)

        # Add the i back to the Li element symbol - return just xyz coords
        coords_2[-1] = coords_2[-1][:1] + 'i' + coords_2[-1][2:]
        coords_2 = '\n'.join(coords_2)

        # Add the ORCA xtb command line with multiplicity and charge info
        xtb_input = (
                "! XTB2 Opt ALPB(WATER)"
                + "\n\n"
                + f"*xyz {total_charge} 1\n"
                + f"{coords_2}\n"
                +'*'
            )

        # Write sandwich xtb input file
        output_file = os.path.join(output_directory, f"{molecule_identifier}_xtb_input.inp")

        try:
            with open(output_file, 'w') as f:
                f.writelines(xtb_input)
        except Exception as e:
            print(f"Error writing file for molecule {molecule_identifier}: {e}")
