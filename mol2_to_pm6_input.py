import os

if __name__ == "__main__":

    input_directory = "/nobackup/cm21sb/tetracyclic_new"
    output_directory = input_directory

    for file in os.listdir(input_directory):
        filepath = os.path.join(input_directory, file)
        filename = os.path.splitext(file)[0]

        if file.endswith('.mol2'):

            with open(filepath, 'r') as infile:
                xyz_str = infile.readlines()

            no_of_atoms = int(xyz_str[2][1:4])
            last_line_of_coords = no_of_atoms + 7
            parts = xyz_str[7:last_line_of_coords]

            total_charge = 0
            charges = []

            for part in parts:
                columns = part.split()
                charges.append(float(columns[-1]))
            total_charge = round(sum(charges))

            molecule_identifier = xyz_str[1].strip()

            coords = xyz_str[7:last_line_of_coords]
            coords_2 = []

            for line in coords:
                line = line[:47]
                line = line[8:]
                atom = str(line[0])
                line = atom + str(line[6:])
                coords_2.append(line)

            coords_2 = '\n'.join(coords_2)

            pm6_input = (
                    "! PM6 Opt Freq"
                    + "\n\n"
                    + "%title\n"
                    + f"{molecule_identifier}\n"
                    + "end\n\n"
                    + "cpcm\n"
                    + "\tsmd true\n"
                    + "\tsolvent \"water\"\n"
                    + "end\n\n"
                    + f"*xyz {total_charge} 1\n"
                    + f"{coords_2}\n"
                    + '*'
                )

            output_file = os.path.join(output_directory, f"{molecule_identifier}_pm6_input.inp2")

            try:
                with open(output_file, 'w') as f:
                    f.writelines(pm6_input)
            except Exception as e:
                print(f"Error writing file for molecule {molecule_identifier}: {e}")

                