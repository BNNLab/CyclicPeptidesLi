# Must use the ccdc miniconda python 3.7.12 environment
# Functions written by Sam Mace, thank you!

# Necessary imports
from ccdc import io
from ccdc.molecule import Molecule, Bond, Atom
from ccdc.descriptors import MolecularDescriptors
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
import os
import copy
import zipfile
from scipy.optimize import minimize
import glob

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

        # It is important to remember that this code is built around how CSD cross miner
        # searchers and saves the ligands
        if self.excluded_vol_mol2_file != None and self.with_metal_mol2_file != None:
            # Read metal centre containing molecules with the ccdc reader
            metal_molecules = io.MoleculeReader(
                self.read_from_location + self.with_metal_mol2_file
            )
            # Load molecules from ccdc list type object to list
            metal_molecules = [molecule for molecule in metal_molecules]
            print(
                "Size of ligand set with metal centre is: " + str(len(metal_molecules))
            )
            # As it is only the ligand and not the metal centre that is needed
            # the metal centre must be removed
            # So the metal atom closest to the cartisien centre of (0, 0, 0) will be removed
            for molecule in metal_molecules:
                normal_list = []
                for atom in molecule.atoms:
                    normal_list.append(
                        [np.linalg.norm(np.array(atom.coordinates)), atom]
                    )
                # sort list in ascending order based on normal of atomic vector
                normal_list = sorted(normal_list, key=lambda x: x[0])
                # Identifie the molecule coming from a metal complex
                molecule.identifier = molecule.identifier + "_m"
                for atom in normal_list:
                    atomic_symbol = atom[1].atomic_symbol
                    # Note how these are alkaline metals and 1st row TM metals
                    # CSD only has these metals saved in its CrossMiner database
                    if (
                        atomic_symbol == "Li"
                        or atomic_symbol == "Na"
                        or atomic_symbol == "K"
                        or atomic_symbol == "Rb"
                        or atomic_symbol == "Cs"
                        or atomic_symbol == "Be"
                        or atomic_symbol == "Mg"
                        or atomic_symbol == "Ca"
                        or atomic_symbol == "Sr"
                        or atomic_symbol == "Ba"
                        or atomic_symbol == "Sc"
                        or atomic_symbol == "Ti"
                        or atomic_symbol == "V"
                        or atomic_symbol == "Cr"
                        or atomic_symbol == "Mn"
                        or atomic_symbol == "Fe"
                        or atomic_symbol == "Co"
                        or atomic_symbol == "Ni"
                        or atomic_symbol == "Cu"
                        or atomic_symbol == "Zn"
                    ):
                        molecule.remove_atom(atom[1])
                        # Break away from the for loop as we only need to remove
                        # the centre atom
                        break
            # Load the excluded volume mol2 file
            self.molecules = io.MoleculeReader(
                self.read_from_location + self.excluded_vol_mol2_file
            )
            # Load molecules from ccdc list type object to list
            self.molecules = [molecule for molecule in self.molecules]
            print(
                "Size of ligand set with excluded volume is: "
                + str(len(self.molecules))
            )
            # Final molecule list object
            self.molecules = self.molecules + metal_molecules

        elif self.excluded_vol_mol2_file != None:
            self.molecules = io.MoleculeReader(
                self.read_from_location + self.excluded_vol_mol2_file
            )
            # Load molecules from ccdc list type object to list
            self.molecules = [molecule for molecule in self.molecules]
            print(
                "Size of ligand set with excluded volume is: "
                + str(len(self.molecules))
            )

        elif self.with_metal_mol2_file != None:
            # Read metal centre containing molecules with the ccdc reader
            metal_molecules = io.MoleculeReader(
                self.read_from_location + self.with_metal_mol2_file
            )
            # Load molecules from ccdc list type object to list
            metal_molecules = [molecule for molecule in metal_molecules]
            print(
                "Size of ligand set with metal centre is: " + str(len(metal_molecules))
            )
            # As it is only the ligand and not the metal centre that is needed
            # the metal centre must be removed
            # So the metal atom closest to the cartisien centre of (0, 0, 0) will be removed
            for molecule in metal_molecules:
                normal_list = []
                for atom in molecule.atoms:
                    normal_list.append(
                        [np.linalg.norm(np.array(atom.coordinates)), atom]
                    )
                # sort list in ascending order based on normal of atomic vector
                normal_list = sorted(normal_list, key=lambda x: x[0])
                # Identifie the molecule coming from a metal complex
                molecule.identifier = molecule.identifier + "_m"
                for atom in normal_list:
                    atomic_symbol = atom[1].atomic_symbol
                    # Note how these are alkaline metals and 1st row TM metals
                    # CSD only has these metals saved in its CrossMiner database
                    if (
                        atomic_symbol == "Li"
                        or atomic_symbol == "Na"
                        or atomic_symbol == "K"
                        or atomic_symbol == "Rb"
                        or atomic_symbol == "Cs"
                        or atomic_symbol == "Be"
                        or atomic_symbol == "Mg"
                        or atomic_symbol == "Ca"
                        or atomic_symbol == "Sr"
                        or atomic_symbol == "Ba"
                        or atomic_symbol == "Sc"
                        or atomic_symbol == "Ti"
                        or atomic_symbol == "V"
                        or atomic_symbol == "Cr"
                        or atomic_symbol == "Mn"
                        or atomic_symbol == "Fe"
                        or atomic_symbol == "Co"
                        or atomic_symbol == "Ni"
                        or atomic_symbol == "Cu"
                        or atomic_symbol == "Zn"
                    ):
                        molecule.remove_atom(atom[1])
                        # Break away from the for loop as we only need to remove the centre atom
                        break
            self.molecules = metal_molecules

        elif self.mol2_file != None:
            self.molecules = io.MoleculeReader(self.read_from_location + self.mol2_file)
            # Load molecules from ccdc list type object to list
            self.molecules = [molecule for molecule in self.molecules]
            print("Size of ligand set is: " + str(len(self.molecules)))

        self.molecules_dict = {}
        for molecule in self.molecules:
            self.molecules_dict[molecule.identifier] = molecule

    def FilterCrossMinerHits(self):
        # Remove molecule if contains 3d, group 1 or group 2 metal
        remove_molecule_index = []
        for idx, molecule in enumerate(self.molecules):
            for atom in molecule.atoms:
                if atom.atomic_symbol == "Li":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Be":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Na":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Mg":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "K":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Ca":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Sc":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Ti":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "V":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Cr":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Mn":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Fe":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Co":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Ni":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Cu":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Zn":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Rb":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Sr":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Cs":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Ba":
                    remove_molecule_index.append(idx)
                    break
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        print(
            "Size of ligand set after removal of metal containing compounds is: "
            + str(len(self.molecules))
        )
        # Remove molecule if carbon is not present
        remove_molecule_index = []
        for idx, molecule in enumerate(self.molecules):
            contains_carbon = False
            for atom in molecule.atoms:
                if atom.atomic_symbol == "C":
                    contains_carbon = True
                    break
            if contains_carbon == False:
                remove_molecule_index.append(idx)
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        print(
            "Size of ligand set after removal of compounds not containing carbon is: "
            + str(len(self.molecules))
        )
        # Add protons to molecules containing absolutly no protons at all.
        remove_molecule_index = []
        for idx, molecule in enumerate(self.molecules):
            contains_hydrogen = False
            for atom in molecule.atoms:
                if atom.atomic_symbol == "H":
                    contains_hydrogen = True
                # Sometimes the Carbon has incorrect valence so this must be corrected
                elif atom.atomic_symbol == "C":
                    valence = 0
                    for bond in atom.bonds:
                        if bond.bond_type == 1:
                            valence = valence + 1  # Single bond valence is 1
                        elif bond.bond_type == 5:  # ccdc aromatic bond value is 5
                            valence = valence + 1.5  # aromatic bond valence is 1.5
                        elif bond.bond_type == 2:
                            valence = valence + 2  # double bond valence is 2
                        elif bond.bond_type == 3:
                            valence = valence + 3  # Triple bond valence is 3
                    if valence < 4:
                        try:
                            molecule.add_hydrogens(mode="all", add_sites=True)
                            contains_hydrogen = True
                            break
                        except RuntimeError:
                            remove_molecule_index.append(idx)
                            print(
                                "Could not add H to "
                                + molecule.to_string("mol2").split("\n")[1]
                            )
                            break
            if contains_hydrogen == False:
                try:
                    molecule.add_hydrogens(mode="all", add_sites=True)
                except RuntimeError:
                    remove_molecule_index.append(idx)
                    print(
                        "Could not add H to "
                        + molecule.to_string("mol2").split("\n")[1]
                    )
        remove_molecule_index = list(set(remove_molecule_index))
        remove_molecule_index.sort()
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        # Set formal charges on all molecules
        remove_molecule_index = []
        for idx, molecule in enumerate(self.molecules):
            try:
                molecule.set_formal_charges()
            except RuntimeError:
                print(
                    "Could not add formal charges "
                    + molecule.to_string("mol2").split("\n")[1]
                )
                remove_molecule_index.append(idx)
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        # Remove molecules with the same smiles string
        # A random metal atom is added. Sometimes the CSD CrossMiner will find a ligand
        # with 2 or more possible configurations of ligand binding
        # So we do not want to loose that in the same SMILES purge
        smiles_list = []
        remove_molecule_index = []
        metal_atom = Atom("Mn", coordinates=(0, 0, 0))
        max_bond_distance = 3
        for idx, molecule in enumerate(self.molecules):
            copy_molecule = molecule.copy()
            a_id = copy_molecule.add_atom(metal_atom)
            for atom in copy_molecule.atoms[:-1]:
                normal = np.linalg.norm(np.array(atom.coordinates))
                atomic_symbol = atom.atomic_symbol
                if (
                    normal <= max_bond_distance
                    and atomic_symbol != "C"
                    and atomic_symbol != "H"
                ):
                    b_id = copy_molecule.add_bond(Bond.BondType(1), a_id, atom)
            smiles = copy_molecule.smiles
            if smiles not in smiles_list:
                smiles_list.append(smiles)
            else:
                remove_molecule_index.append(idx)
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        print(
            "Size of ligand set after removal of compounds with the same SMILES string is: "
            + str(len(self.molecules))
        )
        # Want to remove ligands that are actually two or more components
        remove_molecule_index = []
        for idx, molecule in enumerate(self.molecules):
            if len(molecule.components) >= 2:
                remove_molecule_index.append(idx)
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        with open(
            self.save_to_location + "filtered_ligand_set.mol2", "w"
        ) as filtered_ligand_set:
            for molecule in self.molecules:
                string = molecule.to_string("mol2")
                filtered_ligand_set.write(string + "\n")
            filtered_ligand_set.close()

    def RemoveProtonfromONS(self, atom, molecule):
        # remove protons from oxygens, nitrogens and sulphers as appropiate
        atomic_symbol = atom.atomic_symbol
        atom_neighbours = atom.neighbours
        # Testing for alcohols and thiols
        if (
            (atomic_symbol == "O" or atomic_symbol == "S")
            and len(atom_neighbours) == 2
            and (
                atom_neighbours[0].atomic_symbol == "H"
                or atom_neighbours[1].atomic_symbol == "H"
            )
        ):
            for neighbour_atom in atom_neighbours:
                if neighbour_atom.atomic_symbol == "H":
                    molecule.remove_atom(neighbour_atom)
                    atom.formal_charge = -1
                    break
        # Testing for protonated nitrogens
        elif atomic_symbol == "N":
            """
            ccdc's bond integer
            Type	Integer
            Unknown	0
            Single	1
            Double	2
            Triple	3
            Quadruple	4
            Aromatic	5
            Delocalised	7
            Pi	9
            """
            valence = 0
            for bond in atom.bonds:
                if bond.bond_type == 1:
                    valence = valence + 1  # Single bond valence is 1
                elif bond.bond_type == 5:  # ccdc aromatic bond value is 5
                    valence = valence + 1.5  # aromatic bond valence is 1.5
                elif bond.bond_type == 2:
                    valence = valence + 2  # double bond valence is 2
                elif bond.bond_type == 3:
                    valence = valence + 3  # Triple bond valence is 3
            # nitrogen is always has a positive formal charge if it has a valence of 4.
            # If there is a proton, the proton will be removed
            if valence == 4:
                for neighbour_atom in atom_neighbours:
                    if neighbour_atom.atomic_symbol == "H":
                        molecule.remove_atom(neighbour_atom)
                        atom.formal_charge = 0
                        break

    def AddHydrogensToAtom(
        self, atom, molecule, num_of_H_to_add, bond_length, new_hydrogen_idx
    ):
        c_atom_coor = np.array(atom.coordinates)
        n_atom_coors = [np.array(i.coordinates) for i in atom.neighbours]
        if num_of_H_to_add == 1:
            resultant_vector = np.array([0, 0, 0])
            for n_atom_coor in n_atom_coors:
                resultant_vector = resultant_vector + (n_atom_coor - c_atom_coor)
            norm_of_resultant_vector = np.linalg.norm(resultant_vector)
            unit_resultant_vector = resultant_vector / norm_of_resultant_vector
            new_H_coor_1 = (unit_resultant_vector * -1 * bond_length) + c_atom_coor
            new_atom_id = molecule.add_atom(
                Atom(
                    "H",
                    coordinates=(new_H_coor_1[0], new_H_coor_1[1], new_H_coor_1[2]),
                    label="H" + str(new_hydrogen_idx),
                )
            )
            new_bond_id = molecule.add_bond(Bond.BondType(1), new_atom_id, atom)
        elif num_of_H_to_add == 2:
            resultant_vector = np.array([0, 0, 0])
            for n_atom_coor in n_atom_coors:
                resultant_vector = resultant_vector + (n_atom_coor - c_atom_coor)
            resultant_vector = (
                resultant_vector / np.linalg.norm(resultant_vector)
            ) * bond_length
            try:
                rotation_axis = np.cross(
                    np.cross(
                        n_atom_coors[0] - c_atom_coor, n_atom_coors[1] - c_atom_coor
                    ),
                    resultant_vector,
                )
            except IndexError:
                rotation_axis = np.array([1, 0, 0])
            rotation_axis = rotation_axis / np.norm(rotation_axis)
            new_H_coors = [
                self.RotateVector(
                    vector_to_rotate=resultant_vector,
                    rotation_axis=rotation_axis,
                    theta=np.deg2rad(125),
                ),
                self.RotateVector(
                    vector_to_rotate=resultant_vector,
                    rotation_axis=rotation_axis,
                    theta=np.deg2rad(-125),
                ),
            ]
            new_H_coors = [
                ((new_H_coor / np.norm(new_H_coor)) * bond_length) + c_atom_coor
                for new_H_coor in new_H_coors
            ]
            for new_H_coor_1 in new_H_coors:
                new_atom_id = molecule.add_atom(
                    Atom(
                        "H",
                        coordinates=(new_H_coor_1[0], new_H_coor_1[1], new_H_coor_1[2]),
                        label="H" + str(new_hydrogen_idx),
                    )
                )
                new_bond_id = molecule.add_bond(Bond.BondType(1), new_atom_id, atom)
                new_hydrogen_idx = new_hydrogen_idx + 1
        elif num_of_H_to_add == 3:
            n_atom = atom.neighbours[0]
            n_n_atoms = n_atom.neighbours[0:2]
            n_n_atoms_vectors = [
                np.array(i.coordinates) - np.array(n_atom.coordinates)
                for i in n_n_atoms
            ]
            n_n_atoms_cross = np.cross(n_n_atoms_vectors[0], n_n_atoms_vectors[1])
            n_n_atoms_cross_unit = n_n_atoms_cross / np.linalg.norm(n_n_atoms_cross)
            rotation_axis = np.cross(
                n_n_atoms_cross, c_atom_coor - np.array(n_atom.coordinates)
            )
            rotation_axis = rotation_axis / np.norm(rotation_axis)
            new_H_coor_1 = (
                self.RotateVector(
                    vector_to_rotate=n_n_atoms_cross_unit * bond_length,
                    rotation_axis=rotation_axis,
                    theta=np.deg2rad(-(109 - 90)),
                )
                + c_atom_coor
            )
            new_atom_id = molecule.add_atom(
                Atom(
                    "H",
                    coordinates=(new_H_coor_1[0], new_H_coor_1[1], new_H_coor_1[2]),
                    label="H" + str(new_hydrogen_idx),
                )
            )
            new_bond_id = molecule.add_bond(Bond.BondType(1), new_atom_id, atom)
            new_hydrogen_idx = new_hydrogen_idx + 1
            rotation_axis = np.array(n_atom.coordinates) - c_atom_coor
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            new_H_coor_2 = (
                self.RotateVector(
                    vector_to_rotate=new_H_coor_1 - c_atom_coor,
                    rotation_axis=rotation_axis,
                    theta=np.deg2rad(120),
                )
                + c_atom_coor
            )
            new_atom_id = molecule.add_atom(
                Atom(
                    "H",
                    coordinates=(new_H_coor_2[0], new_H_coor_2[1], new_H_coor_2[2]),
                    label="H" + str(new_hydrogen_idx),
                )
            )
            new_bond_id = molecule.add_bond(Bond.BondType(1), new_atom_id, atom)
            new_hydrogen_idx = new_hydrogen_idx + 1
            new_H_coor_3 = (
                self.RotateVector(
                    vector_to_rotate=new_H_coor_1 - c_atom_coor,
                    rotation_axis=rotation_axis,
                    theta=np.deg2rad(-120),
                )
                + c_atom_coor
            )
            new_atom_id = molecule.add_atom(
                Atom(
                    "H",
                    coordinates=(new_H_coor_3[0], new_H_coor_3[1], new_H_coor_3[2]),
                    label="H" + str(new_hydrogen_idx),
                )
            )
            new_bond_id = molecule.add_bond(Bond.BondType(1), new_atom_id, atom)
            new_hydrogen_idx = new_hydrogen_idx + 1
        return new_hydrogen_idx

    def AddMetalCentre(
        self,
        metal,
        oxidation_state,
        max_bond_dist,
        output_file_name,
        number_of_bonds_formed,
        symmetric_coordination_environment=True,
        vary_protons=True,
    ):
        # Add metal centre to ligand at 0, 0, 0
        # Care is required when adding bonds between the ligand and the metal centre.
        # Atom is not a carbon, hydrogen or halogen
        # Can not be 4 coordinate Sulpher or 5 coordinate phosphours
        # Some times there will be more than 5 potential metal to ligand bonds
        # Different combinations will have to be made and then tested to see which combination gives the lowest energy in the xTB output file
        metal_atom = Atom(metal, coordinates=(0, 0, 0), formal_charge=oxidation_state)
        for molecule in self.molecules:
            m_id = molecule.add_atom(metal_atom)
            molecule.atoms[-1].label = metal + "1"
            bonding_atoms = []  # list of bonding atoms
            for atom in molecule.atoms[:-1]:
                atomic_symbol = atom.atomic_symbol
                num_neighbours = len(atom.neighbours)
                contains_H = False
                for is_H in atom.neighbours:
                    if is_H.atomic_symbol == "H":
                        contains_H = True
                        break
                normal = np.linalg.norm(np.array(atom.coordinates))
                # Add bond between metal and coordinating atoms.
                # Must filter for appropiate potential coordianting atom based on type and coordination number
                if (
                    max_bond_dist >= normal
                    and atomic_symbol != "H"
                    and atomic_symbol != "C"
                    and atomic_symbol != "F"
                    and atomic_symbol != "Cl"
                    and atomic_symbol != "Br"
                    and atomic_symbol != "I"
                    and atomic_symbol != "B"
                ):
                    if atomic_symbol == "P" and num_neighbours >= 4:
                        pass
                    elif atomic_symbol == "S" and num_neighbours >= 4:
                        pass
                    elif (
                        atomic_symbol == "N"
                        and num_neighbours >= 4
                        and contains_H == False
                    ):
                        pass
                    else:
                        bonding_atoms.append(atom)
            # Add bonds if there is the right amount of coordinating atoms to metal centre
            if len(bonding_atoms) == number_of_bonds_formed:
                for atom in bonding_atoms:
                    ProcessCompounds.RemoveProtonfromONS(
                        self, atom=atom, molecule=molecule
                    )
                    b_id = molecule.add_bond(Bond.BondType(1), m_id, atom)

            # Most of the time there are more potential coordinating atoms
            # then the specified amount of coordinating atoms we need
            # Find the magnitude of all possible combinations of vectors
            # The group of vectors with the smallest possible magnitude will be
            # bonded to the metal
            # If symmetric_coordination_environment=True
            elif len(bonding_atoms) < number_of_bonds_formed:
                print("WARNING FIX THIS: " + molecule.identifier)
            else:
                if symmetric_coordination_environment == True:
                    combs = combinations(bonding_atoms, number_of_bonds_formed)
                    magnitude = []
                    for comb in list(combs):
                        magx = 0
                        magy = 0
                        magz = 0
                        for atom in comb:
                            magx = magx + np.array(atom.coordinates)[0]
                            magy = magy + np.array(atom.coordinates)[1]
                            magz = magz + np.array(atom.coordinates)[2]
                        normal = np.linalg.norm([magx, magy, magz])
                        magnitude.append([normal, comb])
                    magnitude = sorted(magnitude, key=lambda x: x[0])
                    for atom in magnitude[0][1]:
                        ProcessCompounds.RemoveProtonfromONS(
                            self, atom=atom, molecule=molecule
                        )
                        b_id = molecule.add_bond(Bond.BondType(1), m_id, atom)

        # Add protons to molecules
        if vary_protons == True:
            molecules_to_be_protonated = []
            # find and make copys of molecules that need to be protonated
            for molecule in self.molecules:
                for atom in molecule.atoms:
                    if atom.atomic_symbol == metal:
                        n_atoms = atom.neighbours
                        for n_atom in n_atoms:
                            n_atom_type = n_atom.atomic_symbol
                            n_atom_charge = n_atom.formal_charge
                            n_atom_total_bond_order = 0
                            for bond in n_atom.bonds:
                                if bond.bond_type == 1:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 1
                                    )
                                elif bond.bond_type == 2:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 2
                                    )
                                elif bond.bond_type == 3:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 3
                                    )
                                elif bond.bond_type == 4:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 4
                                    )
                                elif bond.bond_type == 5:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 1.5
                                    )
                            if (
                                n_atom_type == "O"
                                and len(n_atom.neighbours) == 2
                                and n_atom_total_bond_order == 2
                            ):
                                molecule_copy = molecule.copy()
                                molecule_copy.identifier = (
                                    molecule_copy.identifier + "_protonated"
                                )
                                molecules_to_be_protonated.append(molecule_copy)
                                break
                            elif (
                                n_atom_type == "s"
                                and len(n_atom.neighbours) == 2
                                and n_atom_total_bond_order == 2
                            ):
                                molecule_copy = molecule.copy()
                                molecule_copy.identifier = (
                                    molecule_copy.identifier + "_protonated"
                                )
                                molecules_to_be_protonated.append(molecule_copy)
                                break
                            elif (
                                n_atom_type == "N"
                                and len(n_atom.neighbours) == 3
                                and n_atom_total_bond_order == 3
                            ):
                                molecule_copy = molecule.copy()
                                molecule_copy.identifier = (
                                    molecule_copy.identifier + "_protonated"
                                )
                                molecules_to_be_protonated.append(molecule_copy)
                                break
                            elif (
                                n_atom_type == "P"
                                and len(n_atom.neighbours) == 3
                                and n_atom_total_bond_order == 3
                            ):
                                molecule_copy = molecule.copy()
                                molecule_copy.identifier = (
                                    molecule_copy.identifier + "_protonated"
                                )
                                molecules_to_be_protonated.append(molecule_copy)
                                break
                        break
            # protonate the molecules
            for molecule in molecules_to_be_protonated:
                for atom in molecule.atoms:
                    if atom.atomic_symbol == metal:
                        n_atoms = atom.neighbours
                        for n_atom in n_atoms:
                            n_atom_type = n_atom.atomic_symbol
                            n_atom_charge = n_atom.formal_charge
                            n_atom_total_bond_order = 0
                            for bond in n_atom.bonds:
                                if bond.bond_type == 1:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 1
                                    )
                                elif bond.bond_type == 2:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 2
                                    )
                                elif bond.bond_type == 3:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 3
                                    )
                                elif bond.bond_type == 4:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 4
                                    )
                                elif bond.bond_type == 5:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 1.5
                                    )
                            if (
                                n_atom_type == "O"
                                and len(n_atom.neighbours) == 2
                                and n_atom_total_bond_order == 2
                            ):
                                n_atom.formal_charge = n_atom_charge + 1
                                self.AddHydrogensToAtom(
                                    atom=n_atom,
                                    molecule=molecule,
                                    num_of_H_to_add=1,
                                    bond_length=0.5,
                                    new_hydrogen_idx="H" + str(len(molecule.atoms)),
                                )
                            elif (
                                n_atom_type == "s"
                                and len(n_atom.neighbours) == 2
                                and n_atom_total_bond_order == 2
                            ):
                                n_atom.formal_charge = n_atom_charge + 1
                                self.AddHydrogensToAtom(
                                    atom=n_atom,
                                    molecule=molecule,
                                    num_of_H_to_add=1,
                                    bond_length=0.5,
                                    new_hydrogen_idx="H" + str(len(molecule.atoms)),
                                )
                            elif (
                                n_atom_type == "N"
                                and len(n_atom.neighbours) == 3
                                and n_atom_total_bond_order == 3
                            ):
                                n_atom.formal_charge = n_atom_charge + 1
                                self.AddHydrogensToAtom(
                                    atom=n_atom,
                                    molecule=molecule,
                                    num_of_H_to_add=1,
                                    bond_length=0.5,
                                    new_hydrogen_idx="H" + str(len(molecule.atoms)),
                                )
                            elif (
                                n_atom_type == "P"
                                and len(n_atom.neighbours) == 3
                                and n_atom_total_bond_order == 3
                            ):
                                n_atom.formal_charge = n_atom_charge + 1
                                self.AddHydrogensToAtom(
                                    atom=n_atom,
                                    molecule=molecule,
                                    num_of_H_to_add=1,
                                    bond_length=0.5,
                                    new_hydrogen_idx="H" + str(len(molecule.atoms)),
                                )
                        break
            # molecules have been protonated and added to the set of molecules where they will be saved
            self.molecules = self.molecules + molecules_to_be_protonated
        with open(self.save_to_location + output_file_name + ".mol2", "w") as f:
            for molecule in self.molecules:
                string = molecule.to_string("mol2")
                f.write(string + "\n")
            f.close()

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

    def MakePM6WithORCA6InputFiles(
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
                ORCA_input_string = """! PM6 Opt Freq SCRF(PCM,Solvent=Water)

"""
            else:
                ORCA_input_string = """! PM6 Opt SCRF(PCM,Solvent=Water)

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

    def mol2_to_CRESTInput(
        self,
        spin,
        method,
        output_dir_name,
        CPU_num,
        RAM_num,
        disk_space,
        arc_dir,
        time="06:00:00",
    ):
        # GFN2-xTB => gfn2
        # GFN-FF => gfnff
        # GFN-FF opt with GFN2-xTB single point => gfn2//gfnff
        try:
            os.mkdir(self.save_to_location + output_dir_name)
        except FileExistsError:
            pass
        crest_move_script = """
import os
import shutil
if not os.path.exists('{0}Output'):
    os.makedirs('{0}Output')
""".format(
            arc_dir
        )
        with open(
            self.save_to_location + output_dir_name + "/crest_input_script.sh", "w"
        ) as g:
            script = ""
            for idx, molecule in enumerate(self.molecules):
                # Make .py file to move output crest files into appropiate directories to make output files smaller memory
                crest_move_script = (
                    crest_move_script
                    + """
try:
    if not os.path.exists('{1}Output/{0}'):
        os.makedirs('{1}Output/{0}')
    shutil.copy('{1}Input/{0}/output.txt', '{1}Output/{0}/output.txt')
    shutil.copy('{1}Input/{0}/crest_best.xyz', '{1}Output/{0}/crest_best.xyz')
except FileNotFoundError:
    pass
""".format(
                        molecule.identifier, arc_dir
                    )
                )
                # Make .sh for all .sh files
                script = script + "cd " + molecule.identifier + "\n"
                script = (
                    script + "dos2unix *.sh\n" + "qsub " + molecule.identifier + ".sh\n"
                )
                script = script + "cd ..\n"
                try:
                    os.mkdir(
                        self.save_to_location
                        + output_dir_name
                        + "/"
                        + molecule.identifier
                    )
                except FileExistsError:
                    pass
                crest_dir = arc_dir.split("/")[:-1]
                crest_dir_new = ""
                for i in crest_dir:
                    crest_dir_new = crest_dir_new + i + "/"
                # Make .sh for .xyz inputfile
                RAM_num_per_CPU = RAM_num
                arc_input_string = (
                    "#!/bin/bash\n#$ -cwd -V\n#$ -l h_vmem="
                    + str(RAM_num_per_CPU)
                    + "G\n"
                    + "#$ -l h_rt="
                    + time
                    + "\n"
                    + "#$ -l disk="
                    + str(disk_space)
                    + "G\n"
                    + "#$ -pe smp "
                    + str(CPU_num)
                    + "\n#$ -m be\nchmod +x "
                    + arc_dir
                    + "\nexport PATH=$PATH:"
                    + crest_dir_new
                    + ""
                    + '\ncrest "'
                    + molecule.identifier
                    + '.xyz" --'
                    + method
                    + " --chrg "
                    + str(molecule.formal_charge)
                    + " --uhf "
                    + str(spin)
                    + " -T "
                    + str(CPU_num)
                    + " > output.txt\n"
                )
                with open(
                    self.save_to_location
                    + output_dir_name
                    + "/"
                    + molecule.identifier
                    + "/"
                    + molecule.identifier
                    + ".sh",
                    "w",
                ) as f:
                    f.write(arc_input_string)
                    f.close()

                # Make .xyz inputfile
                xyz_string = str(len(molecule.atoms)) + "\n"
                xyz_string = (
                    xyz_string
                    + "chrg = "
                    + str(molecule.formal_charge)
                    + " uhf = "
                    + str(spin)
                    + "\n"
                )
                for atom in molecule.atoms:
                    atomic_symbol = atom.atomic_symbol
                    coordinates = list(atom.coordinates)
                    xyz_string = (
                        xyz_string
                        + atomic_symbol
                        + " "
                        + str(coordinates[0])
                        + " "
                        + str(coordinates[1])
                        + " "
                        + str(coordinates[2])
                        + "\n"
                    )
                with open(
                    self.save_to_location
                    + output_dir_name
                    + "/"
                    + molecule.identifier
                    + "/"
                    + molecule.identifier
                    + ".xyz",
                    "w",
                ) as f:
                    f.write(xyz_string)
                    f.close()
            g.write(script)
            g.close()
        with open(
            self.save_to_location + output_dir_name + "/move_crest_files.py", "w"
        ) as f:
            f.write(crest_move_script)
            f.close()

    def IsolateLigand(
        self, metal_centre_to_remove, output_file_name, remove_water=True
    ):
        mol_string = ""
        for molecule in self.molecules:
            for atom in molecule.atoms:
                if atom.atomic_symbol == metal_centre_to_remove:
                    molecule.remove_atom(atom)
                    break
            for component in molecule.components:
                if len(component.atoms) > 3:
                    component.identifier = molecule.identifier
                    mol_string = mol_string + component.to_string("mol2")
                    break
        with open(self.save_to_location + output_file_name + ".mol2", "w") as f:
            f.write(mol_string)
            f.close()

    def theta(self, vector_1, vector_2):
        theta = np.arccos(
            np.vdot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        )  # careful, this only works for unit vectors
        if theta == np.nan:
            theta = 0
            return theta
        else:
            return theta

    def TranslateVector(self, trans_vector, input_vector):
        trans_mat = np.array(
            [
                [1, 0, 0, -trans_vector[0]],
                [0, 1, 0, -trans_vector[1]],
                [0, 0, 1, -trans_vector[2]],
                [0, 0, 0, 1],
            ]
        )
        input_vector = np.append(input_vector, [1])
        output_vector = trans_mat @ input_vector
        output_vector = np.delete(output_vector, -1)
        return output_vector

    def TranslateMolecule(self, molecule, trans_vector):
        for atom in molecule.atoms:
            coordinates = np.array(atom.coordinates)
            atom.coordinates = self.TranslateVector(
                trans_vector=trans_vector, input_vector=coordinates
            )
        return molecule

    def RotateVector(self, vector_to_rotate, rotation_axis, theta):
        x, y, z = rotation_axis[0], rotation_axis[1], rotation_axis[2]
        theta = -theta
        rotation_matrix = np.array(
            [
                [
                    np.cos(theta) + (x**2) * (1 - np.cos(theta)),
                    x * y * (1 - np.cos(theta)) - z * np.sin(theta),
                    x * z * (1 - np.cos(theta)) + y * np.sin(theta),
                ],
                [
                    y * x * (1 - np.cos(theta)) + z * np.sin(theta),
                    np.cos(theta) + (y**2) * (1 - np.cos(theta)),
                    y * z * (1 - np.cos(theta)) - x * np.sin(theta),
                ],
                [
                    z * x * (1 - np.cos(theta)) - y * np.sin(theta),
                    z * y * (1 - np.cos(theta)) + x * np.sin(theta),
                    np.cos(theta) + (z**2) * (1 - np.cos(theta)),
                ],
            ]
        ).reshape((3, 3))
        return rotation_matrix @ vector_to_rotate

    def RotateMolecule(self, molecule, rotation_axis, theta):
        for atom in molecule.atoms:
            coordinates = np.array(atom.coordinates)
            atom.coordinates = self.RotateVector(
                vector_to_rotate=coordinates, rotation_axis=rotation_axis, theta=theta
            )
        return molecule

    def NamZCoorSum(self, Nam_atoms):
        z_coor = []
        for Nam_atom in Nam_atoms:
            z_coor.append(abs(np.array(Nam_atom.coordinates)[2]))
        return sum(z_coor)
    
    def RotateNam(self, theta_phi, molecule):
        theta = theta_phi[0]
        phi = theta_phi[1]
        Nam_molecule = Molecule(identifier="Nam_mol")
        # Find all N.am sybol type atoms
        Nam_list = []
        for atom in molecule.atoms:
            if atom.sybyl_type == "N.am":
                Nam_molecule.add_atom(atom)

        molecule = self.RotateMolecule(
            molecule=molecule,
            rotation_axis=np.array([1, 0, 0]),
            theta=theta,
        )
        molecule = self.RotateMolecule(
            molecule=molecule,
            rotation_axis=np.array([0, 1, 0]),
            theta=phi,
        )
        # Find all N.am sybol type atoms
        Nam_list = []
        for atom in Nam_molecule.atoms:
            if atom.sybyl_type == "N.am":
                Nam_list.append(atom)
        return self.NamZCoorSum(
            Nam_atoms=Nam_list
        )

    def MakeLithiumSandwich(self, output_file_name, metal, oxidation_state):
        metal_atom = Atom(metal, coordinates=(0, 0, 0), formal_charge=oxidation_state, label=metal+"1")
        metal_atom_p3 = Atom(metal, coordinates=(0, 0, 3), formal_charge=oxidation_state, label=metal+"2")
        metal_atom_m3 = Atom(metal, coordinates=(0, 0, -3), formal_charge=oxidation_state, label=metal+"3")
        mol2_string = ""
        for molecule in self.molecules:

            # Find all N.am sybol type atoms
            Nam_list = []
            for atom in molecule.atoms:
                if atom.sybyl_type == "N.am":
                    Nam_list.append(atom)
            # Move molecule to Nam centre of mass
            centre_of_mass = np.zeros(3)
            for Nam_atom in Nam_list:
                centre_of_mass = centre_of_mass + np.array(Nam_atom.coordinates)
            centre_of_mass = centre_of_mass/len(Nam_list)
            self.TranslateMolecule(molecule=molecule, trans_vector=centre_of_mass)
            # Rotate so Nam atoms are roughly in xy plane
            # Go through each Nam atom and rotate so each Nam atom has its z coordinate equal to 0
            for Nam_atom in Nam_list:
                Nam_atom_coor = np.array(Nam_atom.coordinates)
                Nam_atom_coor_z = Nam_atom_coor.copy()
                Nam_atom_coor_z[2] = 0
                Nam_atom_coor = Nam_atom_coor/np.linalg.norm(Nam_atom_coor)
                Nam_atom_coor_z = Nam_atom_coor_z/np.linalg.norm(Nam_atom_coor_z)
                rotation_axis = np.cross(
                    a=Nam_atom_coor,
                    b=np.array([0, 0, 1])
                )
                rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)
                theta = self.theta(
                    vector_1=Nam_atom_coor,
                    vector_2=Nam_atom_coor_z,
                )
                self.RotateMolecule(
                    molecule=molecule,
                    rotation_axis=rotation_axis,
                    theta=theta,
                )
            # Minimise Nam z values as much as possible
            res = minimize(
                fun=self.RotateNam,
                args=molecule,
                method="nelder-mead",
                x0=[0, 0],
            )
            print(res.success)
            # Copy molecule and make a sandwich
            molecule2 = molecule.copy()
            molecule = self.TranslateMolecule(molecule=molecule, trans_vector=np.array([0, 0, -3]))
            molecule2 = self.TranslateMolecule(molecule=molecule2, trans_vector=np.array([0, 0, 3]))
            m_id0 = molecule.add_molecule(molecule2)
            m_id1 = molecule.add_atom(metal_atom)
            mol2_string = mol2_string + molecule.to_string("mol2")

        with open(self.save_to_location + output_file_name + ".mol2", "w") as f:
            f.write(mol2_string)
            f.close()

if __name__ == "__main__": 

        # Read combined ligands mol2 file, make the lithium sandwiches and write them to another mol2 file,
        # then produce individual xtb input files
        complexes = ProcessCompounds(
            read_from_location="C:/Users/cm21sb/OneDrive - University of Leeds/Year 4/Sophie Blanch/code/tetracyclic_new/sandwiches/",
            save_to_location="C:/Users/cm21sb/OneDrive - University of Leeds/Year 4/Sophie Blanch/code/tetracyclic_new/sandwiches/",
            mol2_file = 'combined_ligands.mol2'
    )
        complexes.MakeLithiumSandwich(output_file_name="combined_sandwiches.mol2", metal="Li", oxidation_state=1)

