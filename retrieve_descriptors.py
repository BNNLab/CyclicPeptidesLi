# Written by George Hodgin - Thank you!

"""Python script to extract mopac and xTB descriptors from their output files"""
import argparse
import os
import re

import numpy as np
from rdkit import Chem
import pandas as pd

class xyzFile:
    def __init__(self, filepath):
        """
        Initialize the xTBFile object.

        Parameters:
        filepath (str): The file path to the xTB .out file.
        """
        self.filepath = filepath  # Path to the MOPAC .out file
        self.file_text = self._read_file()  # Text content of the file

    def _read_file(self):
        """
        Read the entire xyz file into memory as a list of lines.
        """
        with open(self.filepath, "r") as file:
            file_text = file.readlines()
        return file_text
    
    def get_atom_coords(self)->dict[np.array]:
        """
        Loop over xyz lines, return 3D coords dict
        """
        atom_coords = {}
        for index, line in enumerate(self.file_text[2:]):
            split_line = line.split()
            coords = np.array([float(i) for i in split_line[1:]])
            atom_coords[index] = coords
        return atom_coords

class xTBFile:
    def __init__(self, filepath):
        """
        Initialize the xTBFile object.

        Parameters:
        filepath (str): The file path to the xTB .out file.
        """
        self.filepath = filepath  # Path to the MOPAC .out file
        self.file_text = self._read_file()  # Text content of the file

    def _read_file(self):
        """Read the entire xTB output file into memory as a list of lines."""
        with open(self.filepath, "r") as file:
            file_text = file.readlines()
        return file_text
    
    def get_Fukui_descriptors(self):
        """
        Extract the Fukui descriptors from the output file
        """
        opt_start = None
        for i, line in enumerate(self.file_text):
            if "Fukui functions:" in line:
                if i + 1 < len(self.file_text) and self.file_text[i + 2].startswith(
                    "     1"
                ):
                    opt_start = i + 2  # Skip header lines
                    break

        if opt_start is None:
            return None

        fukui_lines = []
        for line in self.file_text[opt_start:]:
            if not "-------------------------------------------------" in line:
                fukui_lines.append(line)
            else:
                break
        fukui_dict = {}
        split_lines = [i.split() for i in fukui_lines]
        for line in split_lines:
            atom_num = int(re.findall(r"\d+", line[0])[0])
            fukui_dict[atom_num] = [float(i) for i in line[1:]] 

        return fukui_dict

class MopacFile:
    def __init__(self, filepath):
        """
        Initialize the MopacFile object.

        Parameters:
        filepath (str): The file path to the MOPAC .out file.
        """
        self.filepath = filepath  # Path to the MOPAC .out file
        self.file_text = self._read_file()  # Text content of the file

    def _read_file(self):
        """Read the entire MOPAC output file into memory as a list of lines."""
        with open(self.filepath, "r") as file:
            file_text = file.readlines()
        return file_text

    def get_coords(self):
        """
        Extract the optimised geometry from the MOPAC outfile
        """
        opt_start = None
        for i, line in enumerate(self.file_text):
            if "CARTESIAN COORDINATES" in line:
                if i + 2 < len(self.file_text) and self.file_text[i + 2].startswith(
                    "   1"
                ):
                    opt_start = i + 2  # Skip header lines
                    break

        if opt_start is None:
            return None

        coord_lines = []
        for line in self.file_text[opt_start:]:
            if not line.strip() == "":
                coord_lines.append(line)
            else:
                break

        return coord_lines if coord_lines else None
    
    def save_xyz(self, outfilepath: str):
        """
        Extract the MOPAC optimised geometry, save as xyz file
        """
        coords = self.get_coords()
        coords_list = []
        atom_counter = 0
        for str in coords:
            str = str[4:]
            coords_list.append(str)
            atom_counter += 1
        coords_list.insert(0, f"{atom_counter}" + "\n" + "\n")
        with open(outfilepath,"w") as f:
            for line in coords_list:
                f.write(line)

    def get_atomic_charges(self):
        """
        Extracts the atomic charges for each atom in the molecule
        """
        chg_start = None
        for i, line in enumerate(self.file_text):
            if "NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS" in line:
                if i + 2 < len(self.file_text) and self.file_text[i + 2].startswith(
                    "  ATOM NO."
                ):
                    chg_start = i + 3  # Skip header lines
                    break

        if chg_start is None:
            return None

        chg_dict = {}
        for line in self.file_text[chg_start:]:
            if not line.startswith(" DIPOLE"):
                chg_line = line.strip().split()
                try:
                    atom_num = int(chg_line[0])  # Atom number
                    chg = float(chg_line[2])  # Charge
                    chg_dict[atom_num] = chg
                except (IndexError, ValueError):
                    continue
            else:
                break

        return chg_dict if chg_dict else None

    def get_frequencies(self):
        """Extracts the vibrational frequencies and the atom pairs involved"""
        freq_start = None
        for i, line in enumerate(self.file_text):
            if "DESCRIPTION OF VIBRATIONS" in line:
                freq_start = i + 3  # Skip header lines
                break

        if freq_start is None:
            return None

        freqs = {}
        current_freq = None
        for line in self.file_text[freq_start:]:
            split_line = line.split()
            if "FREQUENCY" in split_line:
                current_freq = float(split_line[1])
                freqs[current_freq] = []
            elif "--" in split_line and current_freq is not None:
                try:
                    atom1 = int(split_line[split_line.index("--") - 1])  # (Atom Number)
                except:
                    # if the atom and atom number isnt separated by a space
                    atom1 = int(
                        "".join(
                            re.findall(r"\d+", split_line[split_line.index("--") - 1])
                        )
                    )
                try:
                    atom2 = int(split_line[split_line.index("--") + 2])  # (Atom Number)
                except:
                    # if the atom and atom number isnt separated by a space
                    atom2 = int(
                        "".join(
                            re.findall(r"\d+", split_line[split_line.index("--") + 1])
                        )
                    )
                # get index of item with first % sign for energy contribution
                contrib_index = next(
                    (index for index, item in enumerate(split_line) if "%" in item),
                    None,
                )
                contrib = float(
                    split_line[contrib_index][1:-1]
                )  # get the floating point number for % contrib
                freqs[current_freq].append([(atom1, atom2), contrib])
            elif not line.strip():
                current_freq = None

        return freqs if freqs else None

    def get_mulliken_electronegativity(self):
        """Extracts Mulliken electronegativity in eV."""
        for line in self.file_text:
            if "Mulliken electronegativity:" in line:
                try:
                    electronegativity = float(line.split()[2])
                    return electronegativity
                except (IndexError, ValueError):
                    return None
        return None

    def get_HOMO_LUMO_energies(self):
        """Extracts HOMO and LUMO energies in eV. Returns (Ehomo, Elumo)."""
        ehomo, elumo = None, None
        for line in self.file_text:
            if "Ehomo:" in line:
                try:
                    ehomo = float(line.split()[1])
                except (IndexError, ValueError):
                    return None, None
            if "Elumo:" in line:
                try:
                    elumo = float(line.split()[1])
                    break
                except (IndexError, ValueError):
                    return None, None

        return (
            (ehomo, elumo) if ehomo is not None and elumo is not None else (None, None)
        )

    def get_pKa_OH(self):
        """
        Extracts the atom number and pKa of the most acidic hydrogen bonded
        to an oxygen atom, if available. Returns None otherwise.
        """
        for i, line in enumerate(self.file_text):
            if "Atom      pKa" in line:
                try:
                    split_line = self.file_text[i + 1].split()
                    atom_num = int(split_line[0])
                    pKa = float(split_line[1])
                    return atom_num, pKa
                except (IndexError, ValueError):
                    return None
            elif "No '-O-H' groups found.  pKa cannot be calculated" in line:
                return None
        return None

    def get_ionisation_potential(self):
        """Extracts the ionization potential in eV."""
        for line in self.file_text:
            if "IONIZATION POTENTIAL" in line:
                try:
                    IP = float(line.split()[3])
                    return IP
                except (IndexError, ValueError):
                    return None
        return None

    def get_COSMO_area(self):
        """Returns the COSMO area for the solvent supplied at calculation (default = water): Square Angstroms"""
        for line in self.file_text:
            if "COSMO AREA" in line:
                try:
                    area = float(line.split()[3])
                    return area
                except (IndexError, ValueError):
                    return None

    def get_COSMO_volume(self):
        """Returns the COSMO volume for the the solvent supplied at calculation (default = water): Cubic Angstroms"""
        for line in self.file_text:
            if "COSMO VOLUME" in line:
                try:
                    area = float(line.split()[3])
                    return area
                except (IndexError, ValueError):
                    return None
                
    def get_atomic_Dn_De(self):
        """Returns the nucleophilic and electrophilic delocalisabilities for each atom in a dictionary.
        Can be interpreted as a measure of energy stabilisaton due to nucleophilic/electrophilic attack. (eV^-1)
        """
        start = None
        for i, line in enumerate(self.file_text):
            if "  a   n        Dn(r)        De(r)   q(r) - Z(r)" in line:
                if i + 2 < len(self.file_text):
                    start = i + 2  # Skip header lines
                    break

        if start is None:
            return None

        dnde_dict = {}
        for line in self.file_text[start:]:
            if not line.strip() == "":
                line = line.strip().split()
                try:
                    atom_num = int(line[1])  # Atom number
                    Dn = float(line[2])  # Nucleophilic delocalisability
                    De = float(line[3]) # Electrophilic delocalisability
                    dnde_dict[atom_num] = [Dn, De]
                except (IndexError, ValueError):
                    continue
            else:
                break

        return dnde_dict if dnde_dict else None

    def get_Dn_De_total(self):
        """Returns the sum of the nucleophilic and electrophilic delocalizabilities for all atoms.
        Can be interpreted as a measure of energy stabilisaton due to nucleophilic/electrophilic attack. (eV^-1)
        """
        Dn_tot = None
        De_tot = None
        for line in self.file_text:
            if "Total:" in line:
                try:
                    line_split = line.split()
                    if len(line_split) == 3:
                        Dn_tot = float(line_split[1])
                        De_tot = float(line_split[2])
                except (IndexError, ValueError):
                    return (None, None)
        return (
            (Dn_tot, De_tot)
            if Dn_tot is not None and De_tot is not None
            else (None, None)
        )
    
    def get_atomic_piS(self):
        """Returns the self-polarizability (piS(r)) for each atom in a dictionary.
        """
        start = None
        for i, line in enumerate(self.file_text):
            if "  a   n        piS(r)" in line:
                if i + 2 < len(self.file_text):
                    start = i + 2  # Skip header lines
                    break

        if start is None:
            return None

        piS_dict = {}
        for line in self.file_text[start:]:
            if not line.strip() =="":
                line = line.strip().split()
                try:
                    atom_num = int(line[1])  # Atom number
                    piS = float(line[2])  # self-polarizability
                    piS_dict[atom_num] = piS
                except (IndexError, ValueError):
                    continue
            else:
                break

        return piS_dict if piS_dict else None

    def get_piS_total(self):
        """Returns the sum of the self-polarizability for all atoms: charge^2/energy"""
        for line in self.file_text:
            if "Total:" in line:
                try:
                    line_split = line.split()
                    if len(line_split) == 2:
                        piS_tot = float(line_split[1])
                        return piS_tot
                except (IndexError, ValueError):
                    return None


def main():

    ### Setup of input arguments and output directories ###

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments
    parser.add_argument(
        "--initial_smiles",
        type=str,
        required=True,
        help="initial molecule SMILES for FG group calcs",
    )
    parser.add_argument(
        "--initial_fg_directory",
        type=str,
        required=True,
        help="directory where the initial mols are sorted by fgs",
    )
    parser.add_argument(
        "--mopac_dir",
        type=str,
        required=True,
        help="directory where the mopac results for the initial mols are stored",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for molecule group results. A new directory will be created in the output directory",
    )
    # Parse the arguments
    args = parser.parse_args()
    # Access the arguments
    initial_smiles_file = args.initial_smiles
    initial_fg_directory = args.initial_fg_directory
    mopac_dir = args.mopac_dir
    outdir = args.outdir
    # Import the data
    smarts = pd.read_csv("../Functional groups/reaction_bond_SMARTS_updated.csv")
    # Drop duplicates on reactive centre name column as we dont need all orientations of bonds
    smarts.drop_duplicates(subset="Reactive Centre Name")
    initial_smiles = pd.read_csv(initial_smiles_file)
    initial_smiles = initial_smiles[
        "SMILES"
    ].to_list()  # use the indexes of this list to grab the mopac results file
    initial_fg_file_list = os.listdir(initial_fg_directory)  # loop through these files

    for file in initial_fg_file_list:
        results = []
        file_path = os.path.join(initial_fg_directory, file)
        df = pd.read_csv(file_path)
        # Get FG names
        file = file[:-13]

        if "-AND-" in file:
            fg_names = file.split("-AND-")
        else:
            fg_names = [file]
        # fg_names = file_split[:-1]

        print(fg_names)
        # Get SMARTS
        #smarts_patterns = []
        bond_patterns = []
        bond_names = []

        try:
            for name in fg_names:
                # smarts_pattern = smarts[smarts["Reactive Centre Name"] == name][
                #     "Reactive Centre SMARTS"
                # ].values[0]
                bond_pattern = smarts[smarts["Reactive Centre Name"] == name][
                    "Reactive Bond SMARTS"
                ].values[0]
                bond_name = smarts[smarts["Reactive Centre Name"] == name][
                    "Reactive Bond Name"
                ].values[0]
                #smarts_patterns.append(Chem.MolFromSmarts(smarts_pattern))
                bond_patterns.append(Chem.MolFromSmarts(bond_pattern))
                bond_names.append(bond_name)
        except:
            print("Something wrong with the SMARTS file")
            continue

        for index, row in df.iterrows():
            smiles = row["SMILES"]
            try:
                mol_index = initial_smiles.index(smiles)
            except:
                print("couldn't find mol in initial mols list")
                continue
            mopac_results_file = os.path.join(
                mopac_dir, f"{mol_index}/mop_mol{mol_index}.out"
            )
            mopac = MopacFile(mopac_results_file)
            IP = mopac.get_ionisation_potential()
            E_neg = mopac.get_mulliken_electronegativity()
            EHOMO, ELUMO = mopac.get_HOMO_LUMO_energies()
            pKa = mopac.get_pKa_OH()
            Dn_De_tots = mopac.get_Dn_De_total()
            piS_tot = mopac.get_piS_total()
            COSMO_area = mopac.get_COSMO_area()
            COSMO_volume = mopac.get_COSMO_volume()

            try:
                freq_dict = mopac.get_frequencies()
                chg_dict = mopac.get_atomic_charges()
            except:
                mol_results = {
                    "SMILES": smiles,
                    "Ionisation Potential": IP,
                    "Electronegativity": E_neg,
                    "HOMO Energy": EHOMO,
                    "LUMO Energy": ELUMO,
                    "pKa": pKa,
                    "D(n) Total": Dn_De_tots[0],
                    "D(e) Total": Dn_De_tots[1],
                    "piS Total": piS_tot,
                    "COSMO area": COSMO_area,
                    "COSMO volume": COSMO_volume,
                    "Reactive Centres": None,
                }
                print("Couldn't get freqs/charges")
                continue
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            mol_results = {
                "SMILES": smiles,
                "Ionisation Potential": IP,
                "Electronegativity": E_neg,
                "HOMO Energy": EHOMO,
                "LUMO Energy": ELUMO,
                "pKa": pKa,
                "D(n) Total": Dn_De_tots[0],
                "D(e) Total": Dn_De_tots[1],
                "piS Total": piS_tot,
                "COSMO area": COSMO_area,
                "COSMO volume": COSMO_volume,
                "Reactive Centres": {},
            }

            for bond, name in zip(
                #smarts_patterns, 
                bond_patterns, 
                bond_names
            ):
                chg_list = []
                freqs_list = []
                # centre_matches = mol.GetSubstructMatches(centre)
                bond_matches = mol.GetSubstructMatches(bond)
                if chg_dict:
                    for match in bond_matches:
                        atom_indices = (match[0] + 1, match[1] + 1)
                        chg1 = chg_dict[atom_indices[0]]
                        chg2 = chg_dict[atom_indices[1]]
                        chg_list.append((chg1, chg2))
                else:
                    chg_list.append((None, None))

                if freq_dict:
                    for match in bond_matches:
                        # RDKit output is tuple which is non-hashable
                        match = set(
                            (match[0] + 1, match[1] + 1)
                        )  # mopac is 1 indexed, rdkit is 0
                        all_freqs = {}

                        for freq, pairs in freq_dict.items():
                            for pair in pairs:
                                atoms = set(pair[0])
                                if match.issubset(atoms):
                                    all_freqs[freq] = pair[
                                        1
                                    ]  # assign contributions to frequency

                        # if all_freqs:
                        #     assigned_freq = max(all_freqs, key=all_freqs.get)
                        #     #assigned_freq = max(all_freqs.keys())
                        #     freqs_list.append(assigned_freq)
                        # else:
                        #     pass

                try:
                    avg_chg1 = np.mean([i[0] for i in chg_list])
                    avg_chg2 = np.mean([i[1] for i in chg_list])
                except:
                    avg_chg1 = None
                    avg_chg2 = None
                # try:
                #     avg_freq = np.mean(freqs_list)
                # except:
                #     avg_freq = None

                # Store results by reactive centre name
                mol_results["Reactive Centres"][name] = {
                    "Average Charge": (avg_chg1, avg_chg2),
                    # "Frequencies for atom pair": avg_freq
                    "Frequencies for atom pair": all_freqs.keys(),
                }

            results.append(mol_results)

        results_df = pd.DataFrame(results)

        save_path = os.path.join(outdir, "-AND-".join(fg_names))
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        results_file_path = os.path.join(save_path, "mopac_results.csv")
        results_df.to_csv(results_file_path, index=False)


if __name__ == "__main__":
    main()
