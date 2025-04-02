# Necessary imports 
import os
import numpy as np
import copy

# Classes and functions written by Samuel Mace - thank you!
class Molecule:
    def __init__(self, identifier, formal_charge, atoms):
        self.identifier = identifier
        self.formal_charge = formal_charge
        self.atoms = atoms

class Atom:
    def __init__(self, atomic_symbol, coordinates):
        self.atomic_symbol = atomic_symbol
        self.coordinates = coordinates

class ProcessCompounds:
    def __init__(
        self,
        read_from_location,
        save_to_location,
        excluded_vol_mol2_file=None,
        with_metal_mol2_file=None,
        mol2_file=None,
    ):
        self.read_from_location = read_from_location
        self.save_to_location = save_to_location
        self.excluded_vol_mol2_file = excluded_vol_mol2_file
        self.with_metal_mol2_file = with_metal_mol2_file
        self.mol2_file = mol2_file
        self.molecules = self._load_molecules()

    def _load_molecules(self):
        """
        Parses the MOL2 file and loads molecules into the self.molecules attribute.
        Each molecule starts with @<TRIPOS>MOLECULE and contains atom data under @<TRIPOS>ATOM.
        """
        if not self.mol2_file:
            print("No MOL2 file provided.")
            return []

        molecules = []
        with open(self.mol2_file, "r") as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == "@<TRIPOS>MOLECULE":
                # Start of a new molecule
                identifier = lines[i + 1].strip()  # Second line after @<TRIPOS>MOLECULE is the molecule name
                molecule_atoms = []
                formal_charge = 0  # Default value; update if needed from MOL2 data
                i += 2  # Move to next section

                # Look for the @<TRIPOS>ATOM section to read atoms
                while i < len(lines) and lines[i].strip() != "@<TRIPOS>ATOM":
                    i += 1

                if i < len(lines) and lines[i].strip() == "@<TRIPOS>ATOM":
                    i += 1  # Move to the first atom line

                    # Read atoms until another section starts
                    while i < len(lines) and not lines[i].startswith("@<TRIPOS>"):
                        atom_line = lines[i].strip().split()
                        if len(atom_line) >= 6:
                            atomic_symbol = atom_line[5]  # 6th column in the MOL2 file is the atomic symbol
                            x, y, z = map(float, atom_line[2:5])  # Columns 3-5 are coordinates
                            molecule_atoms.append(Atom(atomic_symbol, [x, y, z]))
                        i += 1

                # Create the Molecule object and add it to the list
                molecules.append(Molecule(identifier, formal_charge, molecule_atoms))

            else:
                i += 1  # Move to the next line

        print(f"Loaded {len(molecules)} molecules from {self.mol2_file}")
        return molecules
        
    

    def MakeGFN2xTBWithORCA6InputFiles(
                self,
                output_dir_name,
                multiplicity,
                Freq=False
            ):
            try:
                os.mkdir(self.save_to_location + output_dir_name)
            except FileExistsError:
                pass
            job_count = 0
            starting_acr4_string = (
                "#!/bin/bash\n"
                + "#$ -cwd -V\n"
                + "#$ -l h_vmem="
                + str(1)
                + "G\n"
                + "#$ -l h_rt="
                + "00:30:00"
                + "\n"
                + "#$ -l disk="
                + str(1)
                + "G\n"
                + "#$ -pe smp "
                + str(1)
                + "\n"
                + "#$ -m be\n"
            )
            main_bash_job_number = 0
            main_bash_string = ""
            for molecule in self.molecules:
                # Write xyz string to its own directory - Calculation in water
                if Freq == True:
                    ORCA_input_string = """! XTB2 Opt Freq ALPB(WATER)

    """
                else:
                    ORCA_input_string = """! XTB2 Opt ALPB(WATER)

    """
                ORCA_input_string = (
                    ORCA_input_string
                    + "*xyz "
                    + str(molecule.formal_charge)
                    + " "
                    + str(multiplicity)
                    + "\n"
                )
                for atom in molecule.atoms:
                    atomic_symbol = atom.atomic_symbol
                    coordinates = np.array(atom.coordinates)
                    ORCA_input_string = (
                        ORCA_input_string
                        + atomic_symbol
                        + " "
                        + str(coordinates[0])
                        + " "
                        + str(coordinates[1])
                        + " "
                        + str(coordinates[2])
                        + "\n"
                    )
                ORCA_input_string = ORCA_input_string + "*"
                with open(
                    self.save_to_location
                    + output_dir_name
                    + "/"
                    + molecule.identifier
                    + "_ORCA6Input.inp",
                    "w",
                ) as f:
                    f.write(ORCA_input_string)
                    f.close()
                if job_count == 0:
                    arc4_string = copy.copy(starting_acr4_string)
                    arc4_string = (
                        arc4_string
                        + "/nobackup/cm21sb/orca/orca_6_0_0_shared_openmpi416/orca "
                        + molecule.identifier
                        + "_ORCA6Input.inp > "
                        + molecule.identifier
                        + "_ORCA6Output.out\n"
                    )
                    job_count = job_count + 1
                elif job_count < 9:
                    arc4_string = (
                        arc4_string
                        + "/nobackup/cm21sb/orca/orca_6_0_0_shared_openmpi416/orca "
                        + molecule.identifier
                        + "_ORCA6Input.inp > "
                        + molecule.identifier
                        + "_ORCA6Output.out\n"
                    )
                    job_count = job_count + 1
                elif job_count == 9:
                    arc4_string = (
                        arc4_string
                        + "/nobackup/cm21sb/orca/orca_6_0_0_shared_openmpi416/orca "
                        + molecule.identifier
                        + "_ORCA6Input.inp > "
                        + molecule.identifier
                        + "_ORCA6Output.out\n"
                    )
                    job_count = 0
                    with open(
                        self.save_to_location
                        + output_dir_name
                        + "/"
                        + "xtb_job_"
                        + str(main_bash_job_number)
                        + ".sh",
                        "w",
                    ) as f:
                        f.write(arc4_string)
                        f.close()
                    main_bash_string = (
                        main_bash_string
                        + "qsub xtb_job_"
                        + str(main_bash_job_number)
                        + ".sh\n"
                    )
                    main_bash_job_number = main_bash_job_number + 1
            with open(
                self.save_to_location
                + output_dir_name
                + "/"
                + "xtb_job_"
                + str(main_bash_job_number)
                + ".sh",
                "w",
            ) as f:
                f.write(arc4_string)
                f.close()
            main_bash_string = (
                main_bash_string + "qsub xtb_job_" + str(main_bash_job_number) + ".sh\n"
            )
            main_bash_job_number = main_bash_job_number + 1
            with open(
                self.save_to_location + output_dir_name + "/" + "main_bash.sh",
                "w",
            ) as f:
                f.write(main_bash_string)
                f.close()

if __name__ == "__main__":
        
        complexes = ProcessCompounds(
            read_from_location="C:/Users/cm21sb/OneDrive - University of Leeds/Year 4/Sophie Blanch/code/tetracyclic_new/ligands/",
            save_to_location="C:/Users/cm21sb/OneDrive - University of Leeds/Year 4/Sophie Blanch/code/tetracyclic_new/ligands/",
            mol2_file="combined_ligands.mol2"
    )
        complexes.MakeGFN2xTBWithORCA6InputFiles(
            output_dir_name = "xTBInputFiles",
            multiplicity = 1,
            Freq=False,
    )
