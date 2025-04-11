# Code from Sam Mace - thank you!
import subprocess
import os
import platform
import json

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
import rdkit.Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import SanitizeMol, SanitizeFlags
import rdkit

from openbabel import pybel
from openbabel import openbabel as ob

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.optimize import basinhopping

from copy import copy
from copy import deepcopy
from itertools import product
import itertools

from ase import Atoms
from ase.constraints import FixAtoms
from ase.constraints import FixBondLengths
from ase.optimize import BFGS
import logging

# from ase.io import write, read
# from ase.build import molecule
# if platform.system() == "Windows":
#     from xtb.ase.calculator import XTB


class Atom:
    def __init__(
        self,
        Label=str,
        Coordinates=np.ndarray,
        SybylType=str,
        AtomicSymbol=str,
        FormalCharge=int,
        SubstructureIndex=1,
        SubstructureName="SUB1",
    ):
        self.Label = Label
        self.Coordinates = Coordinates
        self.SybylType = SybylType
        self.AtomicSymbol = AtomicSymbol
        self.FormalCharge = FormalCharge
        self.SubstructureIndex = SubstructureIndex
        self.SubstructureName = SubstructureName


class Molecule:
    def __init__(
        self,
        Identifier=str,
        NumberOfAtoms=int,
        NumberOfBonds=int,
        Atoms=list,
        AtomsDict=dict,
        ConnectivityMatrix=np.array or None,
        BondOrderMatrix=np.array or None,
        BondTypeMatrix=np.array or None,
        NumberOfSubstructures=int,
    ):
        self.Identifier = Identifier
        self.NumberOfAtoms = NumberOfAtoms
        self.NumberOfBonds = NumberOfBonds
        self.Atoms = Atoms or []
        self.AtomsDict = AtomsDict or {}
        self.ConnectivityMatrix = ConnectivityMatrix
        self.BondOrderMatrix = BondOrderMatrix
        self.BondTypeMatrix = BondTypeMatrix
        self.NumberOfSubstructures = NumberOfSubstructures

    def NormaliseAtomLabels(
        self,
    ):
        count_dict = {}
        for atom in self.Atoms:
            try:
                count_dict[atom.AtomicSymbol] += 1
            except KeyError:
                count_dict[atom.AtomicSymbol] = 1
            atom.Label = atom.AtomicSymbol + str(count_dict[atom.AtomicSymbol])
        self.AtomsDict = {
            atom.Label: [atom_idx, atom] for atom_idx, atom in enumerate(self.Atoms)
        }

    def MoleculeToSMILES(self, allHsExplicit=True):
        # Create an empty RDKit molecule
        rdkit_mol = Chem.RWMol()

        # Add atoms to the RDKit molecule
        atom_indices = []
        for atom in self.Atoms:
            rdkit_atom = Chem.Atom(atom.AtomicSymbol)
            rdkit_atom.SetFormalCharge(atom.FormalCharge)
            atom_idx = rdkit_mol.AddAtom(rdkit_atom)
            atom_indices.append(atom_idx)

        # Add bonds based on the connectivity matrix
        if self.ConnectivityMatrix is not None:
            for i in range(self.NumberOfAtoms):
                for j in range(i + 1, self.NumberOfAtoms):
                    if self.ConnectivityMatrix[i][j] > 0:  # Bond exists
                        bond_type = self.GetRDKitBondType(self.BondOrderMatrix[i][j])
                        rdkit_mol.AddBond(i, j, bond_type)

        # Finalize the molecule
        rdmolops.Kekulize(rdkit_mol, clearAromaticFlags=True)
        return Chem.MolToSmiles(rdkit_mol, allHsExplicit=allHsExplicit)

    def SMILESToMolecule(
        self,
        SMILES_string=str,
        identifier=str,
    ):
        rdkit_hybridisation_to_sybyl_type_dict = {
            "C": {
                "SP3": 3,
                "SP2": 2,
                "SP1": 1,
                "SP": 1,
            },
            "P": {
                "SP3": 3,
                "SP2": 2,
                "SP1": 1,
            },
            "N": {
                "SP3": 3,
                "SP2": 2,
                "SP1": 1,
                "SP": 1,
            },
            "O": {
                "SP3": 3,
                "SP2": 2,
                "SP1": 1,
                "SP": 1,
            },
            "S": {
                "SP3": 3,
                "SP2": 2,
                "SP1": 1,
                "SP": 1,
            },
            "F": {
                "SP3": 0,
            },
            "Cl": {
                "SP3": 0,
            },
            "H": {
                "S": 0,
            },
        }
        rdkit_bondtype_to_bondtype_dict = {
            "SINGLE": [1, "1"],
            "DOUBLE": [2, "2"],
            "TRIPLE": [3, "3"],
            "AROMATIC": [1.5, "ar"],
            "DATIVE": [1, "1"],
        }
        # Parse the SMILES string using RDKit
        mol = Chem.MolFromSmiles(SMILES_string)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {SMILES_string}")

        # Add hydrogens to the molecule to ensure proper geometry
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(
            mol, AllChem.ETKDG()
        )  # Use the ETKDG method for good initial geometry
        AllChem.UFFOptimizeMolecule(
            mol
        )  # Perform a force field optimization for better geometry

        number_of_atoms = mol.GetNumAtoms()
        number_of_bonds = mol.GetNumBonds()
        number_of_substructures = len(
            Chem.GetMolFrags(mol)
        )  # Approximation for substructures

        # Build Atom objects
        atoms = []
        atoms_dict = {}
        conformer = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            label = atom.GetIdx() + 1  # Use 1-based indexing
            coordinates = np.array(conformer.GetAtomPosition(i))  # Get 3D coordinates
            is_aromatic = atom.GetIsAromatic()
            Hybridization = (
                atom.GetHybridization().name
            )  # Hybridization as a proxy for Sybyl type
            atomic_symbol = atom.GetSymbol()
            formal_charge = atom.GetFormalCharge()

            try:
                sybyl_type_part2 = rdkit_hybridisation_to_sybyl_type_dict[atomic_symbol]
                sybyl_type_part2 = sybyl_type_part2[Hybridization]
                if sybyl_type_part2 == 0:
                    sybyl_type = atomic_symbol
                elif is_aromatic == True:
                    sybyl_type = f"{atomic_symbol}.ar"
                else:
                    sybyl_type = f"{atomic_symbol}.{sybyl_type_part2}"
            except KeyError:
                if Hybridization == "UNSPECIFIED":
                    sybyl_type = atomic_symbol
                elif Hybridization == "SP3D":
                    sybyl_type = atomic_symbol
                if atomic_symbol != "H":
                    print((Hybridization, atomic_symbol))

            substructure_index = 1  # Assuming one substructure for simple molecules
            substructure_name = "SUB1"  # Generic name
            atom_obj = Atom(
                Label=atomic_symbol + str(label),
                Coordinates=coordinates,
                SybylType=sybyl_type,
                AtomicSymbol=atomic_symbol,
                FormalCharge=formal_charge,
                SubstructureIndex=substructure_index,
                SubstructureName=substructure_name,
            )
            atoms.append(atom_obj)
            atoms_dict[atom_obj.Label] = [i, atom_obj]

        # Create adjacency matrices
        connectivity_matrix = np.zeros((number_of_atoms, number_of_atoms), dtype=int)
        bond_order_matrix = np.zeros((number_of_atoms, number_of_atoms), dtype=int)
        bond_type_matrix = np.zeros((number_of_atoms, number_of_atoms), dtype=int)

        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_order_type = rdkit_bondtype_to_bondtype_dict[str(bond_type)]

            connectivity_matrix[start_idx, end_idx] = 1
            connectivity_matrix[end_idx, start_idx] = 1
            bond_order_matrix[start_idx, end_idx] = bond_order_type[0]
            bond_order_matrix[end_idx, start_idx] = bond_order_type[0]
            try:
                bond_type_matrix[start_idx, end_idx] = bond_order_type[1]
                bond_type_matrix[end_idx, start_idx] = bond_order_type[1]
            except ValueError:
                bond_type_matrix = bond_type_matrix.astype(str)
                bond_type_matrix[start_idx, end_idx] = bond_order_type[1]
                bond_type_matrix[end_idx, start_idx] = bond_order_type[1]

        # Create Molecule object
        molecule = Molecule(
            Identifier=identifier,
            NumberOfAtoms=number_of_atoms,
            NumberOfBonds=number_of_bonds,
            Atoms=atoms,
            AtomsDict=atoms_dict,
            ConnectivityMatrix=connectivity_matrix,
            BondOrderMatrix=bond_order_matrix,
            BondTypeMatrix=bond_type_matrix.astype(str),
            NumberOfSubstructures=number_of_substructures,
        )

        return molecule

    def MatchSMARTSIdxToAtomIdx(self, SMARTS=str):
        rdkit_mol = self.MoleculeToRDKitMol()
        rdkit_mol = self.RemoveDativeBondsFromRDKitMol(molecule=rdkit_mol)
        matches = rdkit_mol.GetSubstructMatches(Chem.MolFromSmarts(SMARTS))
        if len(matches) > 0:
            SMARTS_to_AtomIdx_dict = {}
            SMARTS_idx_list = [i.split("]")[0] for i in SMARTS.split(":")[1:]]
            for SMARTS_idx, atom_idx in zip(SMARTS_idx_list, matches[0]):
                SMARTS_to_AtomIdx_dict[str(SMARTS_idx)] = atom_idx
            return SMARTS_to_AtomIdx_dict
        else:
            return None

    def GetRDKitBondType(self, bond_order):
        # Map bond order to RDKit bond type
        if bond_order == 1:
            return Chem.BondType.SINGLE
        elif bond_order == 1.5:
            return Chem.BondType.AROMATIC
        elif bond_order == 2:
            return Chem.BondType.DOUBLE
        elif bond_order == 3:
            return Chem.BondType.TRIPLE
        else:
            return Chem.BondType.UNSPECIFIED

    def GetRDKitHybridisation(self, sybyl_type):
        # Map sybyl type to RDKit hybridisation
        if sybyl_type.split(".")[-1] == "1":
            return Chem.rdchem.HybridizationType.SP
        elif sybyl_type.split(".")[-1] == "2":
            return Chem.rdchem.HybridizationType.SP2
        elif sybyl_type.split(".")[-1] == "3":
            return Chem.rdchem.HybridizationType.SP3
        elif sybyl_type.split(".")[-1] == "ar":
            return Chem.rdchem.HybridizationType.SP2
        elif sybyl_type.split(".")[-1] == "pl3":
            return Chem.rdchem.HybridizationType.SP2
        elif sybyl_type.split(".")[0] == "Fe":
            return Chem.rdchem.HybridizationType.SP3D2
        else:
            return Chem.rdchem.HybridizationType.S

    def MoleculeToRDKitMol(self):
        # Writtern by ChatGPT
        """
        Converts a custom Molecule object into an RDKit molecule.

        Returns:
            rdkit.Chem.Mol: The corresponding RDKit molecule.
        """

        # Initialize an editable RDKit molecule
        rdkit_mol = Chem.EditableMol(Chem.Mol())

        # Add atoms
        atom_mapping = {}  # Maps custom atom indices to RDKit indices
        SP2_atom_idx_list = []
        Fe_idx = None
        for i, atom in enumerate(self.Atoms):
            if atom.SybylType.split(".")[-1] == "ar":
                SP2_atom_idx_list.append(i)
            elif atom.SybylType.split(".")[-1] == "2":
                SP2_atom_idx_list.append(i)
            elif atom.SybylType.split(".")[-1] == "pl3":
                SP2_atom_idx_list.append(i)
            elif atom.SybylType == "Fe":
                Fe_idx = i
            rdkit_atom = Chem.Atom(atom.AtomicSymbol)
            rdkit_atom.SetFormalCharge(atom.FormalCharge)
            rdkit_atom.SetHybridization(
                self.GetRDKitHybridisation(sybyl_type=atom.SybylType)
            )
            atom_index = rdkit_mol.AddAtom(rdkit_atom)
            atom_mapping[i] = atom_index

        # Add bonds using the connectivity matrix
        if self.ConnectivityMatrix is not None:
            for i in range(self.NumberOfAtoms):
                for j in range(i + 1, self.NumberOfAtoms):
                    if self.ConnectivityMatrix[i][j] != 0:
                        bond_type = self.GetRDKitBondType(self.BondOrderMatrix[i][j])
                        rdkit_mol.AddBond(atom_mapping[i], atom_mapping[j], bond_type)

        # Finalize the molecule to make it immutable
        final_mol = rdkit_mol.GetMol()

        # Sanitize the molecule (after finalization)
        try:
            SanitizeMol(final_mol, SanitizeFlags.SANITIZE_ALL)
        except rdkit.Chem.rdchem.AtomValenceException:
            try:
                SanitizeMol(
                    final_mol,
                    sanitizeOps=SanitizeFlags.SANITIZE_ADJUSTHS
                    | SanitizeFlags.SANITIZE_KEKULIZE
                    | SanitizeFlags.SANITIZE_SETAROMATICITY,
                )
            except rdkit.Chem.rdchem.KekulizeException:
                SanitizeMol(
                    final_mol,
                    sanitizeOps=SanitizeFlags.SANITIZE_ADJUSTHS
                    | SanitizeFlags.SANITIZE_SETAROMATICITY,
                )
        except rdkit.Chem.rdchem.KekulizeException:
            SanitizeMol(
                final_mol,
                sanitizeOps=SanitizeFlags.SANITIZE_ADJUSTHS
                | SanitizeFlags.SANITIZE_SETAROMATICITY,
            )
        # Sanitize molecule can change the SP2 atoms to SP3, change back
        for idx in SP2_atom_idx_list:
            SP2_atom = final_mol.GetAtomWithIdx(idx)
            SP2_atom.SetHybridization(Chem.rdchem.HybridizationType.SP2)
        # Change TS metal centre to be 6-coordinate
        if Fe_idx != None:
            Fe_atom = final_mol.GetAtomWithIdx(Fe_idx)
            Fe_atom.SetHybridization(Chem.rdchem.HybridizationType.SP3D2)

        # Add 3D coordinates
        conformer = Chem.Conformer(self.NumberOfAtoms)
        for i, atom in enumerate(self.Atoms):
            x, y, z = atom.Coordinates
            conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        final_mol.AddConformer(conformer, assignId=True)

        for atom in final_mol.GetAtoms():
            # Try to retrieve the Sybyl atom type
            if atom.GetSymbol() == "Fe":
                print(atom.GetHybridization())

        return final_mol

    def RemoveDativeBondsFromRDKitMol(self, molecule: rdkit):
        Test_SMILES = Chem.MolToSmiles(molecule)
        dative_bond_count = 0
        if "->" in Test_SMILES:
            dative_bond_count += Test_SMILES.count("->")
        if "<-" in Test_SMILES:
            dative_bond_count += Test_SMILES.count("<-")
        if dative_bond_count > 0:
            actual_dative_bond_count = 0
            for bond in molecule.GetBonds():
                if str(bond.GetBondType()) == "DATIVE":
                    editable_mol = Chem.EditableMol(molecule)
                    editable_mol.RemoveBond(
                        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    )  # Remove the existing bond
                    editable_mol.AddBond(
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx(),
                        Chem.BondType.SINGLE,
                    )
                    molecule = editable_mol.GetMol()
                    actual_dative_bond_count += 1
                if actual_dative_bond_count == dative_bond_count:
                    break
            return molecule
        else:
            return molecule

    def SplitMoleculeIntoComponents(self):
        """
        Splits the molecule into connected components based on the connectivity matrix.
        Returns a list of Molecule objects for each connected component.
        """
        visited = [False] * self.NumberOfAtoms
        components = []

        def dfs(atom_index, component_atoms, component_indices):
            """Depth-First Search to traverse connected atoms."""
            visited[atom_index] = True
            component_atoms.append(self.Atoms[atom_index])
            component_indices.append(atom_index)
            for neighbor, connected in enumerate(self.ConnectivityMatrix[atom_index]):
                if connected and not visited[neighbor]:
                    dfs(neighbor, component_atoms, component_indices)

        # Traverse each atom and identify connected components
        for i in range(self.NumberOfAtoms):
            if not visited[i]:
                component_atoms = []
                component_indices = []
                dfs(i, component_atoms, component_indices)

                # Create sub-molecule
                sub_connectivity_matrix = self.ConnectivityMatrix[
                    np.ix_(component_indices, component_indices)
                ]
                sub_bond_order_matrix = (
                    self.BondOrderMatrix[np.ix_(component_indices, component_indices)]
                    if self.BondOrderMatrix is not None
                    else None
                )
                sub_bond_type_matrix = (
                    self.BondTypeMatrix[np.ix_(component_indices, component_indices)]
                    if self.BondTypeMatrix is not None
                    else None
                )
                sub_molecule = Molecule(
                    Identifier=f"{self.Identifier}_component_{len(components) + 1}",
                    NumberOfAtoms=len(component_atoms),
                    NumberOfBonds=np.sum(sub_connectivity_matrix) // 2,
                    Atoms=component_atoms,
                    AtomsDict={atom.Label: atom for atom in component_atoms},
                    ConnectivityMatrix=sub_connectivity_matrix,
                    BondOrderMatrix=sub_bond_order_matrix,
                    BondTypeMatrix=sub_bond_type_matrix,
                )
                components.append(sub_molecule)

        return components

    def NormaliseSubstructuresLabels(self):
        components = self.SplitMoleculeIntoComponents()
        SMILES_AtomLabels_list = [
            [c.MoleculeToSMILES(), [atom.Label for atom in c.Atoms]] for c in components
        ]
        substructure_index_count = 0
        for smile_AtomLabels in SMILES_AtomLabels_list:
            substructure_index_count += 1
            smile_name = smile_AtomLabels[0]
            AtomLabels = smile_AtomLabels[1]
            for atom_label in AtomLabels:
                self.AtomsDict[atom_label][1].SubstructureName = smile_name
                self.AtomsDict[atom_label][
                    1
                ].SubstructureIndex = substructure_index_count
        self.NumberOfSubstructures = substructure_index_count

    def AddAtom(
        self,
        AtomicSymbol=str,
        SybylType=str,
        FormalCharge=int,
        Coordinates=np.array,
        Label=str,
        Adjust_Bond_Matrices=True,
    ):
        self.Atoms.append(
            Atom(
                Label=Label,
                Coordinates=Coordinates,
                FormalCharge=FormalCharge,
                SybylType=SybylType,
                AtomicSymbol=AtomicSymbol,
                SubstructureIndex=1,
                SubstructureName="SUB1",
            )
        )
        self.AtomsDict = {
            atom.Label: [atom_idx, atom] for atom_idx, atom in enumerate(self.Atoms)
        }
        self.NumberOfAtoms += 1
        if Adjust_Bond_Matrices == True:
            self.ConnectivityMatrix = np.pad(
                self.ConnectivityMatrix,
                pad_width=((0, 1), (0, 1)),
                mode="constant",
                constant_values=0,
            )
            self.BondOrderMatrix = np.pad(
                self.BondOrderMatrix,
                pad_width=((0, 1), (0, 1)),
                mode="constant",
                constant_values=0,
            )
            self.BondTypeMatrix = np.pad(
                self.BondTypeMatrix,
                pad_width=((0, 1), (0, 1)),
                mode="constant",
                constant_values="0.0",
            )
        else:
            pass

    def RemoveAtom(self, Label=str, normalise_substructure_labels=False):
        atom_idx = self.AtomsDict[Label][0]
        del self.Atoms[atom_idx]
        self.AtomsDict = {
            atom.Label: [atom_idx, atom] for atom_idx, atom in enumerate(self.Atoms)
        }
        for axis in [0, 1]:
            self.ConnectivityMatrix = np.delete(
                self.ConnectivityMatrix,
                atom_idx,
                axis=axis,
            )
            self.BondOrderMatrix = np.delete(
                self.BondOrderMatrix,
                atom_idx,
                axis=axis,
            )
            self.BondTypeMatrix = np.delete(
                self.BondTypeMatrix,
                atom_idx,
                axis=axis,
            )
        self.NumberOfBonds = int(sum(np.sum(self.ConnectivityMatrix, axis=0)) / 2)
        self.NumberOfAtoms -= 1
        if normalise_substructure_labels == True:
            self.NormaliseSubstructuresLabels()

    def AddBond(
        self,
        Atom1Label=str,
        Atom2Label=str,
        BondType=str,
        normalise_substructur_labels=False,
    ):
        self.NumberOfBonds += 1
        atom1_idx = self.AtomsDict[Atom1Label][0]
        atom1_atomic_symbol = self.AtomsDict[Atom1Label][1].AtomicSymbol
        atom2_idx = self.AtomsDict[Atom2Label][0]
        atom2_atomic_symbol = self.AtomsDict[Atom2Label][1].AtomicSymbol
        bond_types_to_bond_order_dict = {
            "1": 1,
            "2": 2,
            "3": 3,
            "am": None,
            "ar": 1.5,
            "du": 1,
            "un": 1,
            "nc": 0,
        }
        BondOrder = bond_types_to_bond_order_dict[BondType]
        if BondType == "am":
            if atom1_atomic_symbol == "C" and atom2_atomic_symbol == "N":
                self.BondOrderMatrix[atom1_idx][atom2_idx] = 1
            elif atom2_atomic_symbol == "C" and atom1_atomic_symbol == "N":
                self.BondOrderMatrix[atom1_idx][atom2_idx] = 1
            if atom1_atomic_symbol == "C" and atom2_atomic_symbol == "O":
                self.BondOrderMatrix[atom1_idx][atom2_idx] = 2
            elif atom2_atomic_symbol == "C" and atom1_atomic_symbol == "O":
                self.BondOrderMatrix[atom1_idx][atom2_idx] = 2
        else:
            self.BondOrderMatrix[atom1_idx][atom2_idx] = BondOrder
            self.BondOrderMatrix[atom2_idx][atom1_idx] = BondOrder
        self.BondTypeMatrix[atom1_idx][atom2_idx] = BondType
        self.BondTypeMatrix[atom2_idx][atom1_idx] = BondType
        self.ConnectivityMatrix[atom1_idx][atom2_idx] = 1
        self.ConnectivityMatrix[atom2_idx][atom1_idx] = 1
        if normalise_substructur_labels == True:
            self.NormaliseSubstructuresLabels()

    def RemoveBond(
        self,
        Atom1Label=str,
        Atom2Label=str,
        normalise_substructure_labels=False,
    ):
        self.NumberOfBonds -= 1
        atom1_idx = self.AtomsDict[Atom1Label][0]
        atom2_idx = self.AtomsDict[Atom2Label][0]
        self.BondOrderMatrix[atom1_idx][atom2_idx] = 0
        self.BondOrderMatrix[atom2_idx][atom1_idx] = 0
        self.BondTypeMatrix[atom1_idx][atom2_idx] = "0.0"
        self.BondTypeMatrix[atom2_idx][atom1_idx] = "0.0"
        self.ConnectivityMatrix[atom1_idx][atom2_idx] = 0
        self.ConnectivityMatrix[atom2_idx][atom1_idx] = 0
        if normalise_substructure_labels == True:
            self.NormaliseSubstructuresLabels()

    def AdjustBond(
        self,
        Atom1Label=str,
        Atom2Label=str,
        BondType=str,
    ):
        self.RemoveBond(
            Atom1Label=Atom1Label,
            Atom2Label=Atom2Label,
        )
        self.AddBond(
            Atom1Label=Atom1Label,
            Atom2Label=Atom2Label,
            BondType=BondType,
        )

    def AdjustBondLength(
        self,
        Atom1Label=str,
        Atom2Label=str,
        NewBondLength=float,
    ):
        Atom1Coordinates = self.AtomsDict[Atom1Label][1].Coordinates
        Atom2Coordinates = self.AtomsDict[Atom2Label][1].Coordinates
        MidPoint = (Atom1Coordinates + Atom2Coordinates) / 2
        Atom1Direction = (Atom1Coordinates - MidPoint) / np.linalg.norm(
            Atom1Coordinates - MidPoint
        )
        Atom2Direction = (Atom2Coordinates - MidPoint) / np.linalg.norm(
            Atom1Coordinates - MidPoint
        )
        NewBondLength = NewBondLength / 2
        NewAtom1Coordinates = MidPoint + Atom1Direction * NewBondLength
        NewAtom2Coordinates = MidPoint + Atom2Direction * NewBondLength
        self.AtomsDict[Atom1Label][1].Coordinates = NewAtom1Coordinates
        self.AtomsDict[Atom2Label][1].Coordinates = NewAtom2Coordinates

    def AddMolecule(
        self,
        Molecule,
        Custom_Atom_Label="_templabel",
        normalise_substructur_labels=False,
    ):
        if len(Molecule.Atoms) > 1:
            for atom in Molecule.Atoms:
                self.AddAtom(
                    AtomicSymbol=atom.AtomicSymbol,
                    SybylType=atom.SybylType,
                    FormalCharge=atom.FormalCharge,
                    Coordinates=atom.Coordinates,
                    Label=str(atom.Label) + Custom_Atom_Label,
                    Adjust_Bond_Matrices=False,
                )

            OG_ConnectivityMatrix = copy(self.ConnectivityMatrix)
            Appending_Matrix = copy(Molecule.ConnectivityMatrix)
            size1 = OG_ConnectivityMatrix.shape[0]
            size2 = Appending_Matrix.shape[0]
            new_size = size1 + size2
            combined_matrix = np.zeros((new_size, new_size), dtype=int)

            self.ConnectivityMatrix = copy(combined_matrix)
            self.ConnectivityMatrix[:size1, :size1] = OG_ConnectivityMatrix
            self.ConnectivityMatrix[size1:, size1:] = Appending_Matrix

            OG_BondOrderMatrix = copy(self.BondOrderMatrix)
            Appending_Matrix = copy(Molecule.BondOrderMatrix)
            self.BondOrderMatrix = copy(combined_matrix)
            self.BondOrderMatrix[:size1, :size1] = OG_BondOrderMatrix
            self.BondOrderMatrix[size1:, size1:] = Appending_Matrix

            OG_BondTypeMatrix = copy(self.BondTypeMatrix)
            Appending_Matrix = copy(Molecule.BondTypeMatrix)
            self.BondTypeMatrix = combined_matrix.astype(str)
            self.BondTypeMatrix[:size1, :size1] = OG_BondTypeMatrix
            self.BondTypeMatrix[size1:, size1:] = Appending_Matrix

            self.NumberOfBonds = int(sum(np.sum(self.ConnectivityMatrix, axis=0)) / 2)
        elif len(Molecule.Atoms) == 1:
            self.AddAtom(
                AtomicSymbol=Molecule.Atoms[0].AtomicSymbol,
                SybylType=Molecule.Atoms[0].SybylType,
                FormalCharge=Molecule.Atoms[0].FormalCharge,
                Coordinates=Molecule.Atoms[0].Coordinates,
                Label=str(Molecule.Atoms[0].Label) + Custom_Atom_Label,
                Adjust_Bond_Matrices=True,
            )

        if normalise_substructur_labels == True:
            self.NormaliseAtomLabels()
            self.NormaliseSubstructuresLabels()

    def RemoveMolecule(
        self,
        SMARTS=str,
        normalise_substructur_labels=True,
    ):
        # Match SMARTS to molecule
        rdkitObj = self.MoleculeToRDKitMol()
        rdkitObj = self.RemoveDativeBondsFromRDKitMol(molecule=rdkitObj)
        matches = rdkitObj.GetSubstructMatches(Chem.MolFromSmarts(SMARTS))
        atom_idxs = ()
        for match in matches:
            atom_idxs = atom_idxs + match
        atomLabels_to_remove = [self.Atoms[i].Label for i in atom_idxs]
        for atomLabel in atomLabels_to_remove:
            self.RemoveAtom(Label=atomLabel)

    def GetAtomValence(self, atom):
        atom_idx = self.AtomsDict[atom.Label][0]
        return sum(self.BondOrderMatrix[atom_idx, :])

    def GetMoleculeFormalCharge(self):
        formal_charge = 0
        for atom in self.Atoms:
            formal_charge += atom.FormalCharge
        return formal_charge

    def GetMoleculeWeight(self):
        weight_dict = {
            "H": 1.008,
            "He": 4.003,
            "Li": 6.94,
            "Be": 9.012,
            "B": 10.81,
            "C": 12.01,
            "N": 14.01,
            "O": 16.00,
            "F": 19.00,
            "Ne": 20.18,
            "Na": 22.99,
            "Mg": 24.31,
            "Al": 26.87,
            "Si": 28.09,
            "P": 30.97,
            "S": 32.07,
            "Cl": 35.45,
            "Ar": 39.95,
            "K": 39.10,
            "Ca": 40.08,
            "Sc": 44.96,
            "Ti": 47.87,
            "V": 50.94,
            "Cr": 52.00,
            "Mn": 54.94,
            "Fe": 55.85,
            "Co": 58.93,
            "Ni": 58.69,
            "Cu": 63.55,
            "Zn": 65.38,
            "Ga": 69.72,
            "Ge": 72.63,
            "As": 74.92,
            "Se": 78.96,
            "Br": 79.90,
            "Kr": 83.80,
            "Rb": 85.47,
            "Sr": 87.62,
            "Y": 88.91,
            "Zr": 91.22,
            "Nb": 92.91,
            "Mo": 95.95,
            "Tc": 98.00,
            "Ru": 101.07,
            "Rh": 102.91,
            "Pd": 106.42,
            "Ag": 107.87,
            "Cd": 112.41,
            "In": 114.82,
            "Sn": 118.71,
            "Sb": 121.76,
            "Te": 127.60,
            "I": 126.90,
            "Xe": 131.29,
            "Cs": 132.91,
            "Ba": 137.33,
            "La": 138.91,
            "Ce": 140.12,
            "Pr": 140.91,
            "Nd": 144.24,
            "Pm": 145.00,
            "Sm": 150.36,
            "Eu": 151.96,
            "Gd": 157.25,
            "Tb": 158.93,
            "Dy": 162.50,
            "Ho": 164.93,
            "Er": 167.26,
            "Tm": 168.93,
            "Yb": 173.04,
            "Lu": 174.97,
            "Hf": 178.49,
            "Ta": 180.95,
            "W": 183.84,
            "Re": 186.21,
            "Os": 190.23,
            "Ir": 192.22,
            "Pt": 195.08,
            "Au": 196.97,
            "Hg": 200.59,
            "Tl": 204.38,
            "Pb": 207.2,
            "Bi": 208.98,
            "Po": 209.00,
            "At": 210.00,
            "Rn": 222.00,
            "Fr": 223.00,
            "Ra": 226.00,
            "Ac": 227.00,
            "Th": 232.04,
            "Pa": 231.04,
            "U": 238.03,
            "Np": 237.00,
            "Pu": 244.00,
            "Am": 243.00,
            "Cm": 247.00,
            "Bk": 247.00,
            "Cf": 251.00,
            "Es": 252.00,
            "Fm": 257.00,
            "Md": 258.00,
            "No": 259.00,
            "Lr": 262.00,
            "Rf": 267.00,
            "Db": 270.00,
            "Sg": 271.00,
            "Bh": 274.00,
            "Hs": 277.00,
            "Mt": 278.00,
            "Ds": 281.00,
            "Rg": 282.00,
            "Cn": 285.00,
            "Nh": 286.00,
            "Fl": 289.00,
            "Mc": 290.00,
            "Lv": 293.00,
            "Ts": 294.00,
            "Og": 294.00,
        }
        weight = 0
        for atom in self.Atoms:
            weight += weight_dict[atom.AtomicSymbol]
        return weight

    def GetBasisSetSize(
        self,
        basis_set=str,
        atomic_symbol=str,
    ):
        basis_set_dict = {
            "cc-pVTZ": {
                "H": 5,
                "He": 5,
                "Li": 14,
                "Be": 14,
                "B": 14,
                "C": 14,
                "N": 14,
                "O": 14,
                "F": 14,
                "Ne": 14,
                "Na": 21,
                "Mg": 21,
                "Al": 21,
                "Si": 21,
                "P": 21,
                "S": 21,
                "Cl": 21,
                "Ar": 21,
                "Ca": 41,
                "Sc": 47,
                "Ti": 47,
                "V": 47,
                "Cr": 47,
                "Mn": 47,
                "Fe": 47,
                "Co": 47,
                "Ni": 47,
                "Cu": 47,
                "Zn": 47,
                "Ga": 43,
                "Ge": 43,
                "As": 43,
                "Se": 43,
                "Br": 43,
                "Kr": 43,
            }
        }
        return basis_set_dict[basis_set][atomic_symbol]

    def GetTotalMoleculeBasisSetSize(self, basis_set=str):
        basis_set_count = 0
        for atom in self.Atoms:
            basis_set_count += self.GetBasisSetSize(
                basis_set=basis_set,
                atomic_symbol=atom.AtomicSymbol,
            )
        return basis_set_count

    def FindLTypeBindingAtoms(self):
        LType_atom_labels = []
        for atom in self.Atoms:
            if atom.AtomicSymbol == "C":
                pass
            elif atom.AtomicSymbol == "N":
                if self.GetAtomValence(atom=atom) == 3 and atom.FormalCharge == 0:
                    LType_atom_labels.append(atom.Label)
            elif atom.AtomicSymbol == "P":
                if self.GetAtomValence(atom=atom) == 3 and atom.FormalCharge == 0:
                    LType_atom_labels.append(atom.Label)
        return LType_atom_labels

    def TranslateAtom(self, Atom, TranslationVector=np.array):
        Atom.Coordinates = Atom.Coordinates - TranslationVector

    def TranslateMolecule(self, TranslationVector=np.array):
        [
            self.TranslateAtom(Atom=atom, TranslationVector=TranslationVector)
            for atom in self.Atoms
        ]

    def RotateAtom(self, Atom, rotation_axis=np.array, theta=float):
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
        Atom.Coordinates = rotation_matrix @ Atom.Coordinates

    def RotateMolecule(self, rotation_axis=np.array, theta=float):
        [
            self.RotateAtom(Atom=atom, rotation_axis=rotation_axis, theta=theta)
            for atom in self.Atoms
        ]

    def FindBondAngle(self, a=np.array, b=np.array, c=None or np.array):
        # Writtern by ChatGPT
        """
        Calculate the angle (in radians) between two vectors.

        Parameters:
            v1 (array-like): First vector.
            v2 (array-like): Second vector.

        Returns:
            float: Angle between the vectors in radians.
        """
        if c is not None:
            # Calculate bond vectors
            v1 = a - b
            v2 = c - b
        else:
            v1 = a
            v2 = b
        # Calculate dot product and magnitudes
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        # Calculate cosine of the angle
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)

        # Ensure the value is within valid range for arccos (handle numerical precision issues)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Calculate the angle in radians and convert to degrees
        angle_radians = np.arccos(cos_theta)

        return angle_radians

    def FindTorsionAngle(self, a=np.array, b=np.array, c=np.array, d=np.array):
        # Vectors defining the dihedral
        b1 = b - a
        b2 = c - b
        b3 = d - c
        # Normalize vectors
        b1 /= np.linalg.norm(b1)
        b2 /= np.linalg.norm(b2)
        b3 /= np.linalg.norm(b3)

        # Compute normals to planes formed by the bonds
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        # Normalize the normals
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)

        # Compute the angle between the normals
        cos_theta = np.dot(n1, n2)
        sin_theta = np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2))

        # Compute torsional angle in radians and convert to degrees
        angle = np.arctan2(sin_theta, cos_theta)
        return angle

    def CalcNegThetaCalcCrossRotateMolecule(
        self, StartingPosition=np.array, EndPosition=np.array
    ):
        theta = self.FindBondAngle(a=StartingPosition, b=EndPosition, c=None)
        if np.degrees(theta) == 180:
            rotation_axis = np.array([0, 0, 1])
        elif np.degrees(theta) == 0:
            rotation_axis = np.array([0, 0, 1])
        else:
            rotation_axis = np.cross(a=StartingPosition, b=EndPosition)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        self.RotateMolecule(rotation_axis=rotation_axis, theta=-theta)

    def CalcNegThetaCalcCrossRotateMoleculeAroundCentre(
        self,
        StartingPosition=np.array,
        EndPosition=np.array,
        RotationCentre=np.array,
    ):
        self.TranslateMolecule(TranslationVector=RotationCentre)
        StartingPosition = StartingPosition - RotationCentre
        EndPosition = EndPosition - RotationCentre
        rotation_axis = np.cross(a=StartingPosition, b=EndPosition)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        theta = self.FindBondAngle(a=StartingPosition, b=EndPosition, c=None)
        self.RotateMolecule(rotation_axis=rotation_axis, theta=-theta)
        self.TranslateMolecule(TranslationVector=RotationCentre * -1)

    def CalcPosThetaCalcCrossRotateMolecule(
        self, StartingPosition=np.array, EndPosition=np.array
    ):
        rotation_axis = np.cross(a=StartingPosition, b=EndPosition)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        theta = self.FindBondAngle(a=StartingPosition, b=EndPosition, c=None)
        self.RotateMolecule(rotation_axis=rotation_axis, theta=theta)

    def TranslateAtomByCertainDistance(start, target, distance):
        # Writtern by ChatGPT
        """
        Translates a vector by a certain distance towards a target point.

        Parameters:
            start (array-like): Starting position of the vector (e.g., [x, y, z]).
            target (array-like): Target point to move towards (e.g., [x, y, z]).
            distance (float): Distance to translate towards the target.

        Returns:
            numpy.ndarray: Translated vector position.
        """
        # Calculate the direction vector from start to target
        direction = target - start

        # Normalize the direction vector
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError(
                "Start and target points are the same; translation direction is undefined."
            )
        direction = direction / norm

        # Translate the vector by the specified distance
        translated_vector = start + direction * distance

        return translated_vector

    def TranslateAtomByCertainDistanceBasedOnDirection(
        self, start, direction, distance
    ):
        # Writtern by ChatGPT
        """
        Translates a vector by a certain distance towards a target point.

        Parameters:
            distance (float): Distance to translate towards the target.

        Returns:
            numpy.ndarray: Translated vector position.
        """
        # Normalize the direction vector
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError(
                "Start and target points are the same; translation direction is undefined."
            )
        direction = direction / norm

        # Translate the vector by the specified distance
        translated_vector = start + direction * distance

        return translated_vector

    def GetAtomNeighbours(self, AtomLabel=str):
        # Writtern with ChatGPT
        """
        Returns the indices of atoms directly bonded to a given atom.

        Args:
            atom_index (int): The index of the atom.

        Returns:
            list of int: Indices of neighboring atoms.
        """
        if not self.ConnectivityMatrix.any():
            raise ValueError("ConnectivityMatrix is not defined for this molecule.")

        atom_index = self.AtomsDict[AtomLabel][0]
        if atom_index < 0 or atom_index >= self.NumberOfAtoms:
            raise IndexError("Atom index is out of bounds.")

        # Get neighbors by checking the connectivity matrix row for the atom
        neighbors = [
            idx
            for idx, is_connected in enumerate(self.ConnectivityMatrix[atom_index])
            if is_connected
        ]
        return [self.Atoms[i].Label for i in neighbors]

    def LigandToMetalBondVector(self, binding_atom_label=str):
        NeighbourAtomLabels = self.GetAtomNeighbours(binding_atom_label)
        self.TranslateMolecule(
            TranslationVector=self.AtomsDict[binding_atom_label][1].Coordinates
        )
        resultant_vector = 0
        for natom_label in NeighbourAtomLabels:
            resultant_vector = resultant_vector + np.array(
                self.AtomsDict[natom_label][1].Coordinates
                / np.linalg.norm(np.array(self.AtomsDict[natom_label][1].Coordinates))
            )
        return -resultant_vector / np.linalg.norm(resultant_vector)

    def CleanRDKitMolForUFFOpt(self, rdkit_mol):

        distance_constraint_dict = {}
        avoid_repeating_list = []
        not_clean = True
        while not_clean:
            try:
                ff = AllChem.UFFGetMoleculeForceField(rdkit_mol)
                not_clean = False
                if not_clean == False:
                    break

            except rdkit.Chem.rdchem.AtomValenceException as e:

                e = str(e)
                e = e.replace(",", "")
                e_code = e.split("#")[-1]
                e_code = [i for i in e_code.split(" ") if i != ""]
                atom_idx = e_code[0]
                atomic_symbol = e_code[1]
                atomic_valence = e_code[2]

                if e in avoid_repeating_list:
                    for bond in atom.GetBonds():
                        editable_mol = Chem.RWMol(rdkit_mol)
                        start_atom_idx = bond.GetBeginAtomIdx()
                        end_atom_idx = bond.GetEndAtomIdx()
                        editable_mol.RemoveBond(start_atom_idx, end_atom_idx)
                        rdkit_mol = editable_mol.GetMol()
                    rdkit_mol.UpdatePropertyCache(
                        strict=False
                    )  # Updates properties like aromaticity, valence
                    Chem.GetSymmSSSR(rdkit_mol)  # Computes ring information
                    continue
                else:
                    avoid_repeating_list.append(e)

                if atomic_symbol == "H":
                    atom = rdkit_mol.GetAtomWithIdx(int(atom_idx))
                    atom.SetFormalCharge(0)
                    print(atomic_valence)

                elif atomic_symbol == "Cl" and atomic_valence == "1":
                    atom = rdkit_mol.GetAtomWithIdx(int(atom_idx))
                    atom.SetFormalCharge(0)
                    try:
                        temp_ff = AllChem.UFFGetMoleculeForceField(rdkit_mol)
                    except rdkit.Chem.rdchem.AtomValenceException:
                        atom.SetFormalCharge(-1)
                        for bond in atom.GetBonds():
                            editable_mol = Chem.RWMol(rdkit_mol)
                            start_atom_idx = bond.GetBeginAtomIdx()
                            end_atom_idx = bond.GetEndAtomIdx()
                            bond = editable_mol.GetBondBetweenAtoms(
                                start_atom_idx, end_atom_idx
                            )
                            bond.SetBondType(Chem.BondType.DATIVE)
                            rdkit_mol = editable_mol.GetMol()

                elif atomic_symbol == "N" and atomic_valence == "4":
                    atom = rdkit_mol.GetAtomWithIdx(int(atom_idx))
                    atom.SetFormalCharge(1)

                else:
                    print("#############################")
                    print(f"Opt setup Error Code: {e}")
                    print("#############################")
                    return None

            except RuntimeError as e:
                print("#############################")
                print(f"Opt setup Error Code: {e}")
                print("#############################")
                return None

        return ff

    def CleanRDKitMolForUFFOpt_v2(self, rdkit_mol):
        distance_constraint_dict = {}
        avoid_repeating_atom_idx_list = []
        not_clean = True
        while not_clean:
            try:
                ff = AllChem.UFFGetMoleculeForceField(rdkit_mol)
                not_clean = False
                if not_clean == False:
                    break
            except rdkit.Chem.rdchem.AtomValenceException as e:
                editable_mol = Chem.RWMol(rdkit_mol)
                e = str(e)
                e = e.replace(",", "")
                e_code = e.split("#")[-1]
                e_code = [i for i in e_code.split(" ") if i != ""]
                atom_idx = int(e_code[0])
                atomic_symbol = e_code[1]
                atomic_valence = e_code[2]
                atomRDKitObj = editable_mol.GetAtomWithIdx(atom_idx)
                if atom_idx in avoid_repeating_atom_idx_list:
                    print(atom_idx)
                    pass
                else:
                    OG_FormalCharge = atomRDKitObj.GetFormalCharge()
                    atomRDKitObj.SetFormalCharge(OG_FormalCharge + 1)
                rdkit_mol = editable_mol.GetMol()
                Chem.SanitizeMol(rdkit_mol)
        return ff

    def OptimizeGeometry_UFF(
        self,
        fixed_atoms=None,
        max_steps=10000,
        energy_tol=1e-6,
        fixed_bonds=None,
    ):
        print(
            """
######################################################################
OptimizeGeometry_UFF() Function is Deprecated and no longer maintained
######################################################################
"""
        )

        rdkit_mol = self.MoleculeToRDKitMol()
        ff = self.CleanRDKitMolForUFFOpt_v2(rdkit_mol=rdkit_mol)

        # Add constraints (e.g., freeze atoms)
        if fixed_atoms:
            for atom_idx in fixed_atoms:
                ff.AddFixedPoint(atom_idx)  # Freeze the atom at the given index

        ff.Initialize()
        ff.Minimize(maxIts=max_steps, energyTol=energy_tol)
        print("Minimized")
        energy = ff.CalcEnergy()

        optimized_coords = []
        conf = rdkit_mol.GetConformer()
        for i in range(rdkit_mol.GetNumAtoms()):
            # Access x, y, z coordinates directly from the Point3D object
            pos = conf.GetAtomPosition(i)
            self.Atoms[i].Coordinates = np.array([pos.x, pos.y, pos.z])

        return energy

    def OptimizeGeometry_pybel_UFF(
        self,
        fixed_atoms=None,
        max_steps=1000,
        energy_tol=1e-6,
        fixed_bonds=None,
        file_path=str,
        suppress_output=False,
    ):
        temp_mol = ReadWriteFiles()
        temp_mol.MoleculeDict[self.Identifier] = self
        temp_mol.MoleculeList.append(self)
        copy_temp_mol = deepcopy(temp_mol)
        # Change all formal charge to 0, so OB minimise can run
        # Remove inappropiate sybyl types for openbabel
        # and have just atomic symbols
        for molecule in copy_temp_mol.MoleculeList:
            for atom in molecule.Atoms:
                if atom.SybylType == "O.1":
                    atom.SybylType = "O.2"
        new_file_path = ""
        for string in file_path.split("/")[:-1]:
            new_file_path = new_file_path + f"{string}/"
        copy_temp_mol.WriteMol2File(
            output_mol2_file_name=f"{self.Identifier}_temp.mol2"
        )
        print(f"{self.Identifier}_temp.mol2")
        mol = pybel.readfile("mol2", f"{self.Identifier}_temp.mol2")
        mol = next(mol)
        os.remove(f"{self.Identifier}_temp.mol2")

        # Set up constraints
        if fixed_atoms:
            constrs = ob.OBFFConstraints()
            for atom_idx in fixed_atoms:
                constrs.AddAtomConstraint(atom_idx + 1)

        # Set up force field
        ff = ob.OBForceField.FindForceField("UFF")
        if not ff:
            raise ValueError("Could not find UFF forcefield")

        # Setup minimization
        if fixed_atoms:
            ff.Setup(mol.OBMol, constrs)
            ff.SetConstraints(constrs)
        else:
            ff.Setup(mol.OBMol)

        # Run minimization
        ff.ConjugateGradients(max_steps, energy_tol)
        ff.SteepestDescent(max_steps, energy_tol)
        ff.GetCoordinates(mol.OBMol)

        # Update coordinates
        for ob_atom, atom in zip(mol, self.Atoms):
            atom.Coordinates = np.array(ob_atom.coords)
        UFF_energy = ff.Energy()
        if suppress_output == False:
            print(
                f"""
    ##########################################
    Successful Optimisation with openbabel UFF
    Energy of Optimised Structure:
    {UFF_energy:.1f} kcal/mol
    ##########################################
    """
            )
        return UFF_energy

    def to_ase_atoms(self):
        # Writtern by ChatGPT
        """
        Converts the Molecule object to an ASE Atoms object.

        Returns:
            ase.Atoms: The ASE Atoms representation of the molecule.
        """
        symbols = [atom.AtomicSymbol for atom in self.Atoms]
        positions = [atom.Coordinates for atom in self.Atoms]
        return Atoms(symbols=symbols, positions=positions)

    def OptimizeGeometry_ASE_GFN(
        self,
        fixed_atoms=None,
        max_steps=100,
        energy_tol=1e-4,
        method=str,
        fixed_bonds=None,
        multiplicity=1,
    ):
        # Writtern by ChatGPT
        """
        Optimizes the molecular geometry using an xTB method.
        GFN2-xTB, GFN1-xTB, GFN0-xTB or GFN-FF
        Returns:
            float: Final energy of the optimized structure.
        """
        # Convert to ASE Atoms
        ase_atoms = self.to_ase_atoms()

        # Apply constraints if provided
        if fixed_atoms:
            constraint = FixAtoms(indices=fixed_atoms)
            ase_atoms.set_constraint(constraint)
        if fixed_bonds:
            constraint = FixBondLengths(pairs=fixed_bonds)
            ase_atoms.set_constraint(constraint)

        # Set up xTB calculator
        formal_charge = sum([atom.FormalCharge for atom in self.Atoms])
        calc = XTB(
            method=method,
            formal_charge=formal_charge,
            multiplicity=multiplicity,
        )

        # Attach the calculator to the ASE Atoms object
        ase_atoms.calc = calc

        try:
            # Suppress the output by setting the logger level to CRITICAL
            logging.getLogger("ase.optimize").setLevel(logging.CRITICAL)

            # Initialize the optimizer (BFGS method)
            optimizer = BFGS(
                ase_atoms,
                logfile=None,
            )

            # Perform optimization
            optimizer.run(fmax=energy_tol)

            # Trigger optimization by accessing the potential energy
            energy = ase_atoms.get_potential_energy()

            # Update atom coordinates after optimization
            optimized_coords = ase_atoms.get_positions()
            for i, coord in enumerate(optimized_coords):
                self.Atoms[i].Coordinates = coord
            return energy

        except Exception as e:
            print(f"xTB optimization failed: {e}")
            return None

    def OptimizeGeometry_ORCA_XTB2(
        self,
        fixed_atoms=None,
        max_steps=100,
        energy_tol=1e-4,
        method=str,
        fixed_bonds=None,
        multiplicity=1,
    ):
        readwrite = ReadWriteFiles()
        xyz_input_string = readwrite.WriteXYZBlock(molecule=self)
        # Generate Constraints string
        if fixed_atoms:
            ORCA6_input_string = ""
            fixed_atoms_string = ""
            for atom_idx in fixed_atoms:
                fixed_atoms_string = (
                    fixed_atoms_string + "{C " + str(atom_idx + 1) + " C}\n"
                )
            ORCA6_input_string = f"""!XTB Opt

%geom
constraints
{fixed_atoms_string}end
end

*xyz {sum([atom.FormalCharge for atom in self.Atoms])} {multiplicity}
{xyz_input_string}
*"""
        else:
            ORCA6_input_string = f"""!XTB Opt

*xyz {sum([atom.FormalCharge for atom in self.Atoms])} {multiplicity}
{xyz_input_string}
*"""
        with open(f"{self.Identifier}_Temp.inp", "w") as f:
            f.write(ORCA6_input_string)
            f.close()
        try:
            # Run xTB and capture output
            result = subprocess.run(
                f"/users/cmsma/orca/orca_6_0_0_shared_openmpi416/orca {self.Identifier}_Temp.inp > {self.Identifier}_Temp.out",
                shell=True,
                text=True,
                capture_output=True,
                check=True,
            )
            print("xTB Output:")
            print(result.stdout)  # Standard output from xTB
        except subprocess.CalledProcessError as e:
            print("xTB encountered an error:")
            print(e.stderr)  # Error message from xTB

        # read in xyz files to update coordinates
        with open(f"{self.Identifier}_Temp.xyz", "r") as f:
            xyz_file = f.read()
            f.close()
        xyz_file = xyz_file.split("\n")[2:]
        for atom, line in zip(self.Atoms, xyz_file):
            line = [i for i in line.split(" ") if i != ""]
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])
            atom.Coordinates = np.array([x, y, z])


class MoleculeBuilder:
    def __init__(self):
        self.AtomicSymbol_to_AtomicNumber_Dict = {
            "Fe": 26,
        }
        self.Diatomic_Bond_Lengths = {
            "H-H": 0.741,
            "H-Li": 1.595,
            "Li-H": 1.595,
            "H-Be": 1.343,
            "Be-H": 1.343,
            "H-B": 1.232,
            "B-H": 1.232,
            "H-C": 1.120,
            "C-H": 1.120,
            "H-N": 1.036,
            "N-H": 1.036,
            "H-O": 0.970,
            "O-H": 0.970,
            "H-F": 0.917,
            "F-H": 0.917,
            "H-Si": 1.520,
            "Si-H": 1.520,
            "H-P": 1.422,
            "P-H": 1.422,
            "H-S": 1.341,
            "S-H": 1.341,
            "H-Cl": 1.275,
            "Cl-H": 1.275,
            "H-Fe": 1.655,  # Based on CCCDBD H-Cr BL
            "Fe-H": 1.655,
            "Fe-O": 1.724,  # Based on CCCDBD O-Cu BL
            "O-Fe": 1.724,
        }

    def SMILESToFragments(
        self,
        FragmentDict=dict,
    ):
        LigandIdentifierDict = {}
        for idx, Fragment in enumerate(FragmentDict):
            try:
                if type(FragmentDict[Fragment]["MOL2FILE"]) == list:
                    fragment_molecules_list = []
                    for mol2_file in FragmentDict[Fragment]["MOL2FILE"]:
                        temp_fragment_molecules = ReadWriteFiles()
                        temp_fragment_molecules.ReadMol2File(
                            mol2_file=mol2_file,
                        )
                        fragment_molecules_list = (
                            fragment_molecules_list
                            + temp_fragment_molecules.MoleculeList
                        )
                    fragment_molecules = ReadWriteFiles()
                    fragment_molecules.MoleculeList = fragment_molecules_list
                    fragment_molecules.MoleculeDict = {
                        fragment.Identifier: fragment
                        for fragment in fragment_molecules_list
                    }
            except KeyError as e:
                if type(FragmentDict[Fragment]["SMILES"]) == str:
                    fragment_molecules = ReadWriteFiles()
                    fragment_molecules.ReadSMILEScsv(
                        SMILES_csv_file=FragmentDict[Fragment]["SMILES"],
                    )
                elif type(FragmentDict[Fragment]["SMILES"]) == dict:
                    fragment_molecules = ReadWriteFiles()
                    molecule_class = Molecule()
                    for identifier in FragmentDict[Fragment]["SMILES"]:
                        molecule = molecule_class.SMILESToMolecule(
                            SMILES_string=FragmentDict[Fragment]["SMILES"][identifier],
                            identifier=identifier,
                        )
                        fragment_molecules.MoleculeList.append(molecule)
                        fragment_molecules.MoleculeDict[identifier] = molecule
            LigandIdentifierDict[Fragment] = fragment_molecules.MoleculeList
        return LigandIdentifierDict

    def CreateFragmentCombinations(
        self,
        Fragment_MoleculeDict=dict,
    ):
        ligand_combinations = list(product(*Fragment_MoleculeDict.values()))
        ligand_combinations = [
            dict(zip(Fragment_MoleculeDict.keys(), combination))
            for combination in ligand_combinations
        ]
        return ligand_combinations

    def BindFragmentToMetalCenter(
        self,
        Fragment,
        MainMolecule,
        BindingPositions=list,
        BindingSybylTypes=list,
        MetalCenterLabel=str,
        AtomLabelTempIdx=str,
    ):
        BindingAtomLabels = []
        visited_idx = []
        for sybyltype in BindingSybylTypes:
            for idx, atom in enumerate(Fragment.Atoms):
                if idx in visited_idx:
                    continue
                elif sybyltype == atom.SybylType:
                    BindingAtomLabels.append(atom.Label)
                    visited_idx.append(idx)
                    break
        if len(BindingAtomLabels) == 1:
            if len(Fragment.Atoms) == 1:
                adding_atom = Fragment.Atoms[0]
                MainMolecule.AddAtom(
                    AtomicSymbol=adding_atom.AtomicSymbol,
                    SybylType=adding_atom.SybylType,
                    FormalCharge=adding_atom.FormalCharge,
                    Coordinates=BindingPositions[0],
                    Label=adding_atom.Label + f"_TMP{AtomLabelTempIdx}",
                )
                MainMolecule.AddBond(
                    Atom1Label=adding_atom.Label + f"_TMP{AtomLabelTempIdx}",
                    Atom2Label=MetalCenterLabel,
                    BondType="1",
                )
                return MainMolecule, [adding_atom.Label + f"_TMP{AtomLabelTempIdx}"]
            elif len(Fragment.Atoms) > 1:
                print(BindingAtomLabels)
                bonding_vector = Fragment.LigandToMetalBondVector(
                    binding_atom_label=BindingAtomLabels[0]
                )
                Fragment.CalcNegThetaCalcCrossRotateMolecule(
                    StartingPosition=bonding_vector * -1,
                    EndPosition=BindingPositions[0]
                    / np.linalg.norm(BindingPositions[0]),
                )
                Fragment.TranslateMolecule(TranslationVector=BindingPositions[0] * -1)
                MainMolecule.AddMolecule(
                    Molecule=Fragment,
                    Custom_Atom_Label=f"_TMP{AtomLabelTempIdx}",
                )
                for BindingAtomLabel in BindingAtomLabels:
                    MainMolecule.AddBond(
                        Atom1Label=BindingAtomLabel + f"_TMP{AtomLabelTempIdx}",
                        Atom2Label=MetalCenterLabel,
                        BondType="1",
                    )
                return MainMolecule, [BindingAtomLabel + f"_TMP{AtomLabelTempIdx}"]
        elif len(BindingAtomLabels) > 1:
            BindingPositions_idx = [idx for idx, _ in enumerate(BindingPositions)]
            # Generate permutations of list2 to pair with list1
            permutations = list(itertools.permutations(BindingAtomLabels))
            # Create combinations as pairings
            BindingCombinations = [
                [list(pair) for pair in zip(BindingPositions_idx, perm)]
                for perm in permutations
            ]
            number_of_points_to_travel = 50
            CopyMolecule_UFF_Energy_List = []
            for comb in BindingCombinations:
                # Make deep copy of molecule
                # Add fragment to molecule
                CopyMolecule = deepcopy(MainMolecule)
                CopyMolecule.AddMolecule(
                    Molecule=Fragment,
                    Custom_Atom_Label=f"_TMP{AtomLabelTempIdx}",
                )
                # Add bonds between Fragment and Metal Center
                for BindingAtomLabel in BindingAtomLabels:
                    CopyMolecule.AddBond(
                        Atom1Label=BindingAtomLabel + f"_TMP{AtomLabelTempIdx}",
                        Atom2Label=MetalCenterLabel,
                        BondType="1",
                    )
                # Calculate trajectories for fragment binding to metal centre
                trajectory_dict = {}
                for bonding_pair in comb:
                    BindingPositionCoordinates = BindingPositions[bonding_pair[0]]
                    BindingAtomCoordinates = CopyMolecule.AtomsDict[
                        bonding_pair[1] + f"_TMP{AtomLabelTempIdx}"
                    ][1].Coordinates
                    distance_to_travel = np.linalg.norm(
                        BindingPositionCoordinates - BindingAtomCoordinates
                    )
                    distance_to_travel_per_step = (
                        distance_to_travel / number_of_points_to_travel
                    )
                    trajectory = []
                    for i in range(1, number_of_points_to_travel + 1):
                        trajectory.append(
                            Molecule.TranslateAtomByCertainDistance(
                                start=BindingAtomCoordinates,
                                target=BindingPositionCoordinates,
                                distance=distance_to_travel_per_step * i,
                            )
                        )
                    trajectory_dict[bonding_pair[1] + f"_TMP{AtomLabelTempIdx}"] = (
                        trajectory
                    )
                # optimise fragment to metal centre
                FixedAtomsIdx = [
                    label + f"_TMP{AtomLabelTempIdx}" for label in BindingAtomLabels
                ] + [MetalCenterLabel]
                FixedAtomsIdx = sorted(
                    [CopyMolecule.AtomsDict[label][0] for label in FixedAtomsIdx]
                )
                # Optimise along trajectory to build TS structure
                for step in range(0, number_of_points_to_travel):
                    for atom_label in trajectory_dict:
                        position = trajectory_dict[atom_label][step]
                        CopyMolecule.AtomsDict[atom_label][1].Coordinates = position
                    energy = CopyMolecule.OptimizeGeometry_pybel_UFF(
                        file_path="",
                        fixed_atoms=FixedAtomsIdx,
                        energy_tol=1e-6,
                        suppress_output=True,
                    )
                CopyMolecule_UFF_Energy_List.append([energy, CopyMolecule])
            CopyMolecule_UFF_Energy_List = sorted(
                CopyMolecule_UFF_Energy_List, key=lambda x: x[0]
            )
            Lowest_en_CopyMolecule = CopyMolecule_UFF_Energy_List[0][1]
            return Lowest_en_CopyMolecule, [
                BindingAtomLabel + f"_TMP{AtomLabelTempIdx}"
                for BindingAtomLabel in BindingAtomLabels
            ]

    def BuildMetalComplex(
        self,
        MetalCenter=str,
        MetalOxidation=int,
        FragmentDict={},
        GFN2xTB_Opt_FinalStructure=False,
        CREST_Conformer_Search=False,
        Output_mol2_file=str,
    ):
        # Build Complex from ground up
        # It is assumed metal center is at centre of origin

        # Convert SMILES to Molecule Objects
        Fragment_SMILEStoMoleculeDict = self.SMILESToFragments(
            FragmentDict=FragmentDict
        )

        # Create Fragment combinations for the complexes
        Fragment_Combinations = self.CreateFragmentCombinations(
            Fragment_MoleculeDict=Fragment_SMILEStoMoleculeDict
        )
        MoleculeList = []
        MoleculeDict = {}
        # Add fragments to metal centre that have no stereochemical information
        for idx, Frag_Comb in enumerate(Fragment_Combinations):
            # Intilise Complex with Metal Center
            atoms_list = [
                Atom(
                    Label=f"{MetalCenter}1",
                    Coordinates=np.array([0, 0, 0]),
                    SybylType=MetalCenter,
                    AtomicSymbol=MetalCenter,
                    FormalCharge=MetalOxidation,
                    SubstructureIndex=1,
                    SubstructureName="SUB1",
                )
            ]
            atoms_dict = {}
            atoms_dict[f"{MetalCenter}1"] = [0, atoms_list[0]]
            MainMolecule = Molecule(
                Identifier=f"mol{idx+1}",
                NumberOfAtoms=1,
                NumberOfBonds=0,
                Atoms=atoms_list,
                AtomsDict=atoms_dict,
                ConnectivityMatrix=np.array([0]),
                BondOrderMatrix=np.array([0]),
                BondTypeMatrix=np.array(["0"]),
                NumberOfSubstructures=1,
            )
            # First attach fragments not dependant on stereochemistry
            TempIdx = 1
            BindingAtomLabelsList = [f"{MetalCenter}1"]
            for FragmentIdentifier in FragmentDict:
                if (
                    FragmentDict[FragmentIdentifier]["StereoChemicalInformation"]
                    == None
                ):
                    binding_info = FragmentDict[FragmentIdentifier][
                        "BindingInformation"
                    ]
                    BindingPositions = [info[0] for info in binding_info]
                    BindingSybylTypes = [info[1] for info in binding_info]
                    Fragment = Frag_Comb[FragmentIdentifier]
                    MainMolecule, BindingAtomLabels = self.BindFragmentToMetalCenter(
                        Fragment=Fragment,
                        MainMolecule=MainMolecule,
                        BindingPositions=BindingPositions,
                        BindingSybylTypes=BindingSybylTypes,
                        MetalCenterLabel=f"{MetalCenter}1",
                        AtomLabelTempIdx=str(TempIdx),
                    )
                    TempIdx += 1
                    BindingAtomLabelsList = BindingAtomLabelsList + BindingAtomLabels
            # Second attach fragments dependant on stereochemistry
            for FragmentIdentifier in FragmentDict:
                if (
                    type(FragmentDict[FragmentIdentifier]["StereoChemicalInformation"])
                    == dict
                ):
                    StereoChemicalInfo = FragmentDict[FragmentIdentifier][
                        "StereoChemicalInformation"
                    ]
                    binding_info = FragmentDict[FragmentIdentifier][
                        "BindingInformation"
                    ]
                    BindingPositions = [info[0] for info in binding_info]
                    BindingSybylTypes = [info[1] for info in binding_info]
                    # Load in MainMolecule as rdkit molecule
                    # Convert dative metal bonds to single bonds
                    main_rdkit_mol = MainMolecule.MoleculeToRDKitMol()
                    metal_substructure = (
                        f"[#{self.AtomicSymbol_to_AtomicNumber_Dict[MetalCenter]}]"
                    )
                    metal_rdkit_idx = main_rdkit_mol.GetSubstructMatch(
                        Chem.MolFromSmarts(metal_substructure)
                    )[0]
                    atom = main_rdkit_mol.GetAtomWithIdx(metal_rdkit_idx)
                    bonds = atom.GetBonds()
                    for bond in bonds:
                        if str(bond.GetBondType()) == "DATIVE":
                            editable_mol = Chem.EditableMol(main_rdkit_mol)
                            editable_mol.RemoveBond(
                                bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                            )  # Remove the existing bond
                            editable_mol.AddBond(
                                bond.GetBeginAtomIdx(),
                                bond.GetEndAtomIdx(),
                                Chem.BondType.SINGLE,
                            )
                            main_rdkit_mol = editable_mol.GetMol()
                    # Search for substructure
                    Substructure_SMILES = StereoChemicalInfo["CorrespondingSMILES"]
                    matches = main_rdkit_mol.GetSubstructMatches(
                        Chem.MolFromSmarts(Substructure_SMILES)
                    )
                    # Ideally there should be only one substructure
                    # Find out which is the stereo atom and which is the metal atom
                    if len(matches) == 1:
                        Atom1 = MainMolecule.Atoms[matches[0][0]]
                        Atom2 = MainMolecule.Atoms[matches[0][-1]]
                        if Atom1.AtomicSymbol == MetalCenter:
                            metal_atom = Atom1
                            stereo_atom = Atom2
                        elif Atom2.AtomicSymbol == MetalCenter:
                            metal_atom = Atom2
                            stereo_atom = Atom1
                    # Make copy of MainMolecule
                    # Establish bonding vector of Fragment
                    CopyMolecule = deepcopy(MainMolecule)
                    stereo_atom = CopyMolecule.AtomsDict[stereo_atom.Label][1]
                    stereo_atom_bond_vector = CopyMolecule.LigandToMetalBondVector(
                        binding_atom_label=stereo_atom.Label
                    )
                    if len(BindingPositions) == 1:
                        if StereoChemicalInfo["Stereochemistry"] == "Cis":
                            binding_coordinates = (
                                (BindingPositions[0] * stereo_atom_bond_vector)
                                / np.linalg.norm(
                                    BindingPositions[0] * stereo_atom_bond_vector
                                )
                            ) * StereoChemicalInfo["BondLength"]
                        elif StereoChemicalInfo["Stereochemistry"] == "Trans":
                            binding_coordinates = (
                                (
                                    (BindingPositions[0] * stereo_atom_bond_vector)
                                    / np.linalg.norm(
                                        BindingPositions[0] * stereo_atom_bond_vector
                                    )
                                )
                                * StereoChemicalInfo["BondLength"]
                                * -1
                            )
                    Fragment = Frag_Comb[FragmentIdentifier]
                    MainMolecule, BindingAtomLabels = self.BindFragmentToMetalCenter(
                        Fragment=Fragment,
                        MainMolecule=MainMolecule,
                        BindingPositions=[binding_coordinates],
                        BindingSybylTypes=BindingSybylTypes,
                        MetalCenterLabel=f"{MetalCenter}1",
                        AtomLabelTempIdx=str(TempIdx),
                    )
                    TempIdx += 1
                    BindingAtomLabelsList = BindingAtomLabelsList + BindingAtomLabels
            BindingAtomIdxList = sorted(
                [MainMolecule.AtomsDict[Label][0] for Label in BindingAtomLabelsList]
            )
            MainMolecule.OptimizeGeometry_pybel_UFF(
                file_path="",
                fixed_atoms=BindingAtomIdxList,
                energy_tol=1e-6,
                suppress_output=True,
            )
            if GFN2xTB_Opt_FinalStructure == True:
                MainMolecule.OptimizeGeometry_ASE_GFN(method="GFN2-xTB")
            MainMolecule.NormaliseAtomLabels()
            MoleculeList.append(MainMolecule)
        MoleculeDict = {molecule.Identifier: molecule for molecule in MoleculeList}
        writer = ReadWriteFiles()
        writer.MoleculeDict = MoleculeDict
        writer.MoleculeList = MoleculeList
        writer.WriteMol2File(output_mol2_file_name=Output_mol2_file)

    def SMARTS_BondsFormed(self, reactants_SMARTS=list, TS_SMARTS=str):
        # Map bonds in reactants
        if len(reactants_SMARTS) > 1:
            main_reactant_SMART = reactants_SMARTS[0] + "."
            for reactant_SMART in reactants_SMARTS[1:]:
                main_reactant_SMART = main_reactant_SMART + reactant_SMART + "."
            main_reactant_SMART = main_reactant_SMART[:-1]
        else:
            main_reactant_SMART = reactants_SMARTS[0]
        atom_idx_list = [i.split("]")[0] for i in main_reactant_SMART.split(":")[1:]]
        reactants_idx_dict = {str(idx): i for idx, i in enumerate(atom_idx_list)}
        mol = Chem.MolFromSmarts(main_reactant_SMART)
        reactant_bonds = []
        for bond in mol.GetBonds():
            reactant_bonds.append(
                (
                    reactants_idx_dict[str(bond.GetBeginAtomIdx())],
                    reactants_idx_dict[str(bond.GetEndAtomIdx())],
                )
            )
        # Map bonds in TS
        atom_idx_list = [i.split("]")[0] for i in TS_SMARTS.split(":")[1:]]
        TS_idx_dict = {str(idx): i for idx, i in enumerate(atom_idx_list)}
        TS_bonds = []
        mol = Chem.MolFromSmarts(TS_SMARTS)
        for bond in mol.GetBonds():
            TS_bonds.append(
                (
                    TS_idx_dict[str(bond.GetBeginAtomIdx())],
                    TS_idx_dict[str(bond.GetEndAtomIdx())],
                )
            )
        # Find which bonds have being formed for TS
        TS_bonds = [
            tuple(sorted((int(TupObj[0]), int(TupObj[1])))) for TupObj in TS_bonds
        ]
        reactant_bonds = [
            tuple(sorted((int(TupObj[0]), int(TupObj[1])))) for TupObj in reactant_bonds
        ]
        bonds_formed = []
        for bond in TS_bonds:
            if bond in reactant_bonds:
                pass
            else:
                bonds_formed.append(bond)
        return bonds_formed

    def SMARTS_BondsBroken(self, TS_SMARTS=str, products_SMARTS=list):
        # Map bonds in reactants
        if len(products_SMARTS) > 1:
            main_product_SMART = products_SMARTS[0] + "."
            for product_SMART in products_SMARTS[1:]:
                main_product_SMART = main_product_SMART + product_SMART + "."
            main_product_SMART = main_product_SMART[:-1]
        else:
            main_product_SMART = products_SMARTS[0]
        atom_idx_list = [i.split("]")[0] for i in main_product_SMART.split(":")[1:]]
        products_idx_dict = {str(idx): i for idx, i in enumerate(atom_idx_list)}
        mol = Chem.MolFromSmarts(main_product_SMART)
        product_bonds = []
        for bond in mol.GetBonds():
            product_bonds.append(
                (
                    products_idx_dict[str(bond.GetBeginAtomIdx())],
                    products_idx_dict[str(bond.GetEndAtomIdx())],
                )
            )
        # Map bonds in TS
        atom_idx_list = [i.split("]")[0] for i in TS_SMARTS.split(":")[1:]]
        TS_idx_dict = {str(idx): i for idx, i in enumerate(atom_idx_list)}
        TS_bonds = []
        mol = Chem.MolFromSmarts(TS_SMARTS)
        for bond in mol.GetBonds():
            TS_bonds.append(
                (
                    TS_idx_dict[str(bond.GetBeginAtomIdx())],
                    TS_idx_dict[str(bond.GetEndAtomIdx())],
                )
            )
        # Find which bonds have being formed for TS
        TS_bonds = [
            tuple(sorted((int(TupObj[0]), int(TupObj[1])))) for TupObj in TS_bonds
        ]
        product_bonds = [
            tuple(sorted((int(TupObj[0]), int(TupObj[1])))) for TupObj in product_bonds
        ]
        bonds_broken = []
        for bond in TS_bonds:
            if bond in product_bonds:
                pass
            else:
                bonds_broken.append(bond)
        return bonds_broken

    def SMARTS_ChangeInCharge(self, reactants_SMARTS=list, products_SMARTS=list):
        # Map charges in Reactants
        if len(reactants_SMARTS) > 1:
            main_reactant_SMART = reactants_SMARTS[0] + "."
            for reactant_SMART in reactants_SMARTS[1:]:
                main_reactant_SMART = main_reactant_SMART + reactant_SMART + "."
            main_reactant_SMART = main_reactant_SMART[:-1]
        else:
            main_reactant_SMART = reactants_SMARTS[0]
        atom_idx_list = [i.split("]")[0] for i in main_reactant_SMART.split(":")[1:]]
        reactants_idx_dict = {str(idx): i for idx, i in enumerate(atom_idx_list)}
        Reac_mol = Chem.MolFromSmarts(main_reactant_SMART)
        SMARTS_reac_idx_charge_dict = {}
        for reac_atom in Reac_mol.GetAtoms():
            SMARTS_reac_idx_charge_dict[
                str(reactants_idx_dict[str(reac_atom.GetIdx())])
            ] = reac_atom.GetFormalCharge()

        if len(products_SMARTS) > 1:
            main_product_SMART = products_SMARTS[0] + "."
            for product_SMART in products_SMARTS[1:]:
                main_product_SMART = main_product_SMART + product_SMART + "."
            main_product_SMART = main_product_SMART[:-1]
        else:
            main_product_SMART = products_SMARTS[0]
        atom_idx_list = [i.split("]")[0] for i in main_product_SMART.split(":")[1:]]
        products_idx_dict = {str(idx): i for idx, i in enumerate(atom_idx_list)}
        Prod_mol = Chem.MolFromSmarts(main_product_SMART)
        SMARTS_prod_idx_charge_dict = {}
        for prod_atom in Prod_mol.GetAtoms():
            SMARTS_prod_idx_charge_dict[
                str(products_idx_dict[str(prod_atom.GetIdx())])
            ] = prod_atom.GetFormalCharge()
        change_in_charge_dict = {}
        for reac_idx in SMARTS_reac_idx_charge_dict:
            reac_charge = SMARTS_reac_idx_charge_dict[reac_idx]
            prod_charge = SMARTS_prod_idx_charge_dict[reac_idx]
            change_in_charge_dict[reac_idx] = prod_charge - reac_charge
        return change_in_charge_dict

    def SMARTS_ChangeInBondType(self, reactants_SMARTS=list, products_SMARTS=list):
        # Map bond type changes
        bondtype_to_bondorder_dict = {
            "SINGLE": 1,
            "DOUBLE": 2,
            "TRIPLE": 3,
            "AROMATIC": 1.5,
        }
        if len(reactants_SMARTS) > 1:
            main_reactant_SMART = reactants_SMARTS[0] + "."
            for reactant_SMART in reactants_SMARTS[1:]:
                main_reactant_SMART = main_reactant_SMART + reactant_SMART + "."
            main_reactant_SMART = main_reactant_SMART[:-1]
        else:
            main_reactant_SMART = reactants_SMARTS[0]
        atom_idx_list = [i.split("]")[0] for i in main_reactant_SMART.split(":")[1:]]
        reactants_idx_dict = {str(idx): i for idx, i in enumerate(atom_idx_list)}
        Reac_mol = Chem.MolFromSmarts(main_reactant_SMART)
        reactant_bond_order_matrix = np.zeros((len(atom_idx_list), len(atom_idx_list)))
        for bond in Reac_mol.GetBonds():
            atom1_idx = bond.GetBeginAtomIdx()
            atom1_SMARTS_idx = int(reactants_idx_dict[str(atom1_idx)])
            atom2_idx = bond.GetEndAtomIdx()
            atom2_SMARTS_idx = int(reactants_idx_dict[str(atom2_idx)])
            bond_type = bond.GetBondType()
            bond_order = bondtype_to_bondorder_dict[str(bond_type)]
            reactant_bond_order_matrix[atom1_SMARTS_idx - 1, atom2_SMARTS_idx - 1] = (
                bond_order
            )
            reactant_bond_order_matrix[atom2_SMARTS_idx - 1, atom1_SMARTS_idx - 1] = (
                bond_order
            )

        if len(products_SMARTS) > 1:
            main_product_SMART = products_SMARTS[0] + "."
            for product_SMART in products_SMARTS[1:]:
                main_product_SMART = main_product_SMART + product_SMART + "."
            main_product_SMART = main_product_SMART[:-1]
        else:
            main_product_SMART = products_SMARTS[0]
        atom_idx_list = [i.split("]")[0] for i in main_product_SMART.split(":")[1:]]
        products_idx_dict = {str(idx): i for idx, i in enumerate(atom_idx_list)}
        Prod_mol = Chem.MolFromSmarts(main_product_SMART)
        product_bond_order_matrix = np.zeros((len(atom_idx_list), len(atom_idx_list)))
        for bond in Prod_mol.GetBonds():
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            atom1_idx = bond.GetBeginAtomIdx()
            atom1_SMARTS_idx = int(products_idx_dict[str(atom1_idx)])
            atom2_idx = bond.GetEndAtomIdx()
            atom2_SMARTS_idx = int(products_idx_dict[str(atom2_idx)])
            bond_type = bond.GetBondType()
            bond_order = bondtype_to_bondorder_dict[str(bond_type)]
            product_bond_order_matrix[atom1_SMARTS_idx - 1, atom2_SMARTS_idx - 1] = (
                bond_order
            )
            product_bond_order_matrix[atom2_SMARTS_idx - 1, atom1_SMARTS_idx - 1] = (
                bond_order
            )
        # Use bond order matricies to find put which bonds have changed
        # No need to look for bonds fully formed or fully broken
        # [[(atom1_SMARTS_idx, atom2_SMARTS_idx), new_bond_order_str]]
        bondorder_to_change_list = []
        for idx, i in enumerate(range(0, len(atom_idx_list))):
            for j in range(idx + 1, len(atom_idx_list)):
                reactant_bond_order = reactant_bond_order_matrix[i, j]
                product_bond_order = product_bond_order_matrix[i, j]
                if reactant_bond_order >= 1 and product_bond_order >= 1:
                    if reactant_bond_order != product_bond_order:
                        bondorder_to_change_list.append(
                            [(str(i + 1), str(j + 1)), product_bond_order]
                        )
        return bondorder_to_change_list

    def MatchReactantsToSMARTS(
        self, reaction_combination=dict, reactants_SMARTS_list=list
    ):
        SMARTS_matches_dict = {}
        for reactant_id in reaction_combination:
            # Step 1) Retrieve and Clean
            # Retrieve Reactant
            # Convert to rdkit mol object
            # Remove Dative Bonds
            reactant = reaction_combination[reactant_id]
            remove_dative_bonds = Molecule()
            rdkit_reactant = reactant.MoleculeToRDKitMol()
            rdkit_reactant = remove_dative_bonds.RemoveDativeBondsFromRDKitMol(
                molecule=rdkit_reactant
            )
            # Step 2) Match SMARTS with reactant
            for reactant_SMARTS in reactants_SMARTS_list:
                matches = rdkit_reactant.GetSubstructMatches(
                    Chem.MolFromSmarts(reactant_SMARTS)
                )
                if len(matches) >= 1:
                    SMARTS_matches_dict[reactant_id] = {
                        "Matches": matches,
                        "Reactant_SMARTS": reactant_SMARTS,
                        "Reactant": reactant,
                    }
        return SMARTS_matches_dict

    def CheckForMultipleSMARTSMatches(self, SMARTS_matches_dict=dict):
        multiple_matches = False
        for reactant_id in SMARTS_matches_dict:
            if len(SMARTS_matches_dict[reactant_id]["Matches"]) > 1:
                multiple_matches = True
                break
        return multiple_matches

    def GetSMARTSIdxToReactantAtomIdx(
        self, SMARTS_string=str, Reactant_Match_Tuple=tuple
    ):
        SMARTS_string = SMARTS_string.replace("]:[", "][")
        SMARTS_string = SMARTS_string.split(":")[1:]
        SMARTS_string = [i.split("]")[0] for i in SMARTS_string]
        SMARTSIdx_to_reactantAtomIdx = {}
        for SMARTSIdx, reactantAtomIdx in zip(SMARTS_string, Reactant_Match_Tuple):
            SMARTSIdx_to_reactantAtomIdx[SMARTSIdx] = reactantAtomIdx
        return SMARTSIdx_to_reactantAtomIdx

    def FindCorrectAtomsBasedOnStereoInfo(
        self,
        Stereochemical_Information=dict,
        reactant_SMARTS_matches_dict=dict,
    ):
        for fragment_id in Stereochemical_Information:
            isomer_type = Stereochemical_Information[fragment_id]["Isomer_Type"]
            SMARTS_string = Stereochemical_Information[fragment_id]["SMARTS"]
            SMARTS_idx_Dihedral = Stereochemical_Information[fragment_id][
                "SMARTS_idx_Dihedral"
            ]
            for reactant_id in reactant_SMARTS_matches_dict:
                reactant = reactant_SMARTS_matches_dict[reactant_id]["Reactant"]
                remove_dative_bonds = Molecule()
                rdkit_reactant = reactant.MoleculeToRDKitMol()
                rdkit_reactant = remove_dative_bonds.RemoveDativeBondsFromRDKitMol(
                    molecule=rdkit_reactant
                )
                matches = rdkit_reactant.GetSubstructMatches(
                    Chem.MolFromSmarts(SMARTS_string)
                )
                for match in matches:
                    # Get SMARTSIdx_To_ReactantAtomIdx_dict
                    SMARTSIdx_To_ReactantAtomIdx_dict = (
                        self.GetSMARTSIdxToReactantAtomIdx(
                            SMARTS_string=SMARTS_string,
                            Reactant_Match_Tuple=match,
                        )
                    )
                    # print(match)
                    # print(SMARTSIdx_To_ReactantAtomIdx_dict)
                    torsion_angle = abs(
                        np.degrees(
                            reactant.FindTorsionAngle(
                                a=reactant.Atoms[
                                    SMARTSIdx_To_ReactantAtomIdx_dict[
                                        SMARTS_idx_Dihedral[0]
                                    ]
                                ].Coordinates,
                                b=reactant.Atoms[
                                    SMARTSIdx_To_ReactantAtomIdx_dict[
                                        SMARTS_idx_Dihedral[1]
                                    ]
                                ].Coordinates,
                                c=reactant.Atoms[
                                    SMARTSIdx_To_ReactantAtomIdx_dict[
                                        SMARTS_idx_Dihedral[2]
                                    ]
                                ].Coordinates,
                                d=reactant.Atoms[
                                    SMARTSIdx_To_ReactantAtomIdx_dict[
                                        SMARTS_idx_Dihedral[3]
                                    ]
                                ].Coordinates,
                            )
                        )
                    )
                    if abs(torsion_angle) > 90:
                        actural_isomer = "Trans"
                    elif abs(torsion_angle) <= 90:
                        actural_isomer = "Cis"
                    correct_isomer_fragment = isomer_type == actural_isomer
                    # print(correct_isomer_fragment)
                    if correct_isomer_fragment == False:
                        # Find matching idx values with fragment and main molecule
                        # then remove tuple from list
                        for reactant_id in reactant_SMARTS_matches_dict:
                            main_reactant_matches = list(
                                reactant_SMARTS_matches_dict[reactant_id]["Matches"]
                            )
                            for idx, main_reactant_match in enumerate(
                                main_reactant_matches
                            ):
                                # print(main_reactant_match)
                                # print(match)
                                # check to see all fragment matches match with main reactant matches
                                fully_matches = True
                                for atom1idx in match:
                                    atom_matches_another_atom = True
                                    for atom2idx in main_reactant_match:
                                        if atom1idx == atom2idx:
                                            atom_matches_another_atom = True
                                            break
                                        else:
                                            atom_matches_another_atom = False
                                    if atom_matches_another_atom == False:
                                        fully_matches = False
                                        break
                                if fully_matches == True:
                                    del main_reactant_matches[idx]
                                    reactant_SMARTS_matches_dict[reactant_id][
                                        "Matches"
                                    ] = tuple(main_reactant_matches)
                                    break
                    # print(reactant_SMARTS_matches_dict)
                    # print("")
        return reactant_SMARTS_matches_dict

    def AlignMoleculeWithSMARTSCoordinates(
        self,
        reactant_SMART_matches_dict=dict,
        SMARTSIdx_Coordinates=dict,
    ):
        # Align molecule into transition state positions
        # Before Aligning molecules into position
        # Need to create dictionary of SMARTS idx to match idx and molecule object
        # Need to create tracker dictionary
        # To track whereever the molecule needs to be translated or rotated
        SMARTS_idx_to_AtomIdx_molObj_dict = {}
        tracker_dict = {}
        for reactant_id in reactant_SMART_matches_dict:
            reactant_info_dict = reactant_SMART_matches_dict[reactant_id]
            molObj = reactant_info_dict["Reactant"]
            SMARTS_idx_to_molIdx_dict = self.GetSMARTSIdxToReactantAtomIdx(
                SMARTS_string=reactant_info_dict["Reactant_SMARTS"],
                Reactant_Match_Tuple=reactant_info_dict["Matches"][0],
            )
            for SMARTS_idx in SMARTS_idx_to_molIdx_dict:
                AtomIdx = SMARTS_idx_to_molIdx_dict[SMARTS_idx]
                SMARTS_idx_to_AtomIdx_molObj_dict[SMARTS_idx] = {
                    "Reactant_ID": reactant_id,
                    "Reactant": molObj,
                    "Atom_Idx": AtomIdx,
                }
            tracker_dict[reactant_id] = {
                "Translated?": "Not_Translated",
                "Yesterdays_Atom_Idx": None,
            }
        # With tracker dictionary and smarts idx to atom idx and mol object dict
        # align molecule to TS atomic positions
        no_SMART_matches = False
        for TS_SMART_idx in SMARTSIdx_Coordinates:
            try:
                atomIdx_molObj_dict = SMARTS_idx_to_AtomIdx_molObj_dict[TS_SMART_idx]
                template_coordinate = SMARTSIdx_Coordinates[TS_SMART_idx]
                molObj = atomIdx_molObj_dict["Reactant"]
                atomIdx = atomIdx_molObj_dict["Atom_Idx"]
                reactant_id = atomIdx_molObj_dict["Reactant_ID"]
                if tracker_dict[reactant_id]["Translated?"] == "Not_Translated":
                    molObj.TranslateMolecule(
                        TranslationVector=molObj.Atoms[atomIdx].Coordinates
                    )
                    molObj.TranslateMolecule(TranslationVector=template_coordinate * -1)
                    tracker_dict[reactant_id]["Translated?"] = "Translated"
                    tracker_dict[reactant_id]["Yesterdays_Atom_Idx"] = atomIdx
                elif tracker_dict[reactant_id]["Translated?"] == "Translated":
                    yesterdays_atom_idx = tracker_dict[reactant_id][
                        "Yesterdays_Atom_Idx"
                    ]
                    molObj.CalcNegThetaCalcCrossRotateMoleculeAroundCentre(
                        StartingPosition=molObj.Atoms[atomIdx].Coordinates,
                        EndPosition=template_coordinate,
                        RotationCentre=molObj.Atoms[yesterdays_atom_idx].Coordinates,
                    )
                    tracker_dict[reactant_id]["Yesterdays_Atom_Idx"] = atomIdx
            except KeyError:
                no_SMART_matches = True
                return (
                    reactant_SMART_matches_dict,
                    no_SMART_matches,
                    SMARTS_idx_to_AtomIdx_molObj_dict,
                )
        for TS_SMART_idx in SMARTSIdx_Coordinates:
            atomIdx_molObj_dict = SMARTS_idx_to_AtomIdx_molObj_dict[TS_SMART_idx]
            template_coordinate = SMARTSIdx_Coordinates[TS_SMART_idx]
            molObj = atomIdx_molObj_dict["Reactant"]
            atomIdx = atomIdx_molObj_dict["Atom_Idx"]
            molObj.Atoms[atomIdx].Coordinates = template_coordinate
        return (
            reactant_SMART_matches_dict,
            no_SMART_matches,
            SMARTS_idx_to_AtomIdx_molObj_dict,
        )

    def FindTSReactantDirectionVectors(
        self,
        TS_SMARTS_coordinates=dict,
        SMARTS_idx_to_AtomIdx_molObj_dict=dict,
    ):
        # Find TS geometric mid point
        TS_midpoint = np.array([0, 0, 0])
        for SMARTS_idx in TS_SMARTS_coordinates:
            coordinate = TS_SMARTS_coordinates[SMARTS_idx]
            TS_midpoint = TS_midpoint + coordinate
        TS_midpoint = TS_midpoint / len(TS_SMARTS_coordinates)
        # Find Resultant vectors from TS_midpoint to the atoms involved in reaction
        # (TS bonding coordinates - TS midpoint coordinate)
        TS_molecule_resultant_vector_dict = {}
        for SMARTS_idx in TS_SMARTS_coordinates:
            molObj = SMARTS_idx_to_AtomIdx_molObj_dict[SMARTS_idx]["Reactant"]
            atomIdx = SMARTS_idx_to_AtomIdx_molObj_dict[SMARTS_idx]["Atom_Idx"]
            try:
                reactant_id = SMARTS_idx_to_AtomIdx_molObj_dict[SMARTS_idx][
                    "Reactant_ID"
                ]
                TS_molecule_resultant_vector_dict[reactant_id] = (
                    TS_molecule_resultant_vector_dict[reactant_id]
                    + (molObj.Atoms[atomIdx].Coordinates - TS_midpoint)
                )
            except KeyError:
                reactant_id = SMARTS_idx_to_AtomIdx_molObj_dict[SMARTS_idx][
                    "Reactant_ID"
                ]
                TS_molecule_resultant_vector_dict[reactant_id] = (
                    molObj.Atoms[atomIdx].Coordinates - TS_midpoint
                )
        SMARTSIdx_direction_dict = {}
        for SMARTS_idx in TS_SMARTS_coordinates:
            reactant_id = SMARTS_idx_to_AtomIdx_molObj_dict[SMARTS_idx]["Reactant_ID"]
            SMARTSIdx_direction_dict[SMARTS_idx] = TS_molecule_resultant_vector_dict[
                reactant_id
            ]
        return TS_midpoint, TS_molecule_resultant_vector_dict, SMARTSIdx_direction_dict

    def PrepareTranslationToBuildTS(
        self,
        TS_SMARTS_coordinates=dict,
        SMARTS_idx_to_AtomIdx_molObj_dict=dict,
        reactant_SMART_matches_dict=dict,
        reaction_idx=int,
        translation_distance=5.0,
        number_of_steps=40,
    ):
        # Find Resultant vectors from TS_midpoint to the atoms involved in reaction
        # (TS bonding coordinates - TS midpoint coordinate)
        TS_midpoint, TS_molecule_resultant_vector_dict, SMARTSIdx_direction_dict = (
            self.FindTSReactantDirectionVectors(
                TS_SMARTS_coordinates=TS_SMARTS_coordinates,
                SMARTS_idx_to_AtomIdx_molObj_dict=SMARTS_idx_to_AtomIdx_molObj_dict,
            )
        )
        # Using TS_midpoint and resultant_vector_dict create and opt along trajectory
        stepsize = translation_distance / number_of_steps
        # Translate molecules via their respective TS resultant vector
        for reactant_id in TS_molecule_resultant_vector_dict:
            reactant_info = reactant_SMART_matches_dict[reactant_id]
            molObj = reactant_info["Reactant"]
            direction = TS_molecule_resultant_vector_dict[reactant_id]
            normal_direction = direction / np.linalg.norm(direction)
            translation_vector = Molecule.TranslateAtomByCertainDistance(
                start=np.array([0, 0, 0]),
                target=normal_direction,
                distance=translation_distance,
            )
            molObj.TranslateMolecule(TranslationVector=translation_vector * -1)
        # Add molecules together to make one molecule
        new_SMARTS = None
        main_TS_molecule = None
        SMART_matches = None
        reaction_name = f"Reaction_{reaction_idx}_TSgu"
        for reactant_id in reactant_SMART_matches_dict:
            molObj = reactant_SMART_matches_dict[reactant_id]["Reactant"]
            matches = list(reactant_SMART_matches_dict[reactant_id]["Matches"][0])
            SMARTS = reactant_SMART_matches_dict[reactant_id]["Reactant_SMARTS"]
            if new_SMARTS == None:
                new_SMARTS = SMARTS
                main_TS_molecule = molObj
                SMART_matches = matches
            else:
                new_SMARTS = new_SMARTS + f".{SMARTS}"
                SMART_matches = SMART_matches + [
                    i + main_TS_molecule.NumberOfAtoms for i in matches
                ]
                main_TS_molecule.AddMolecule(Molecule=molObj)
        main_TS_molecule.Identifier = reaction_name
        main_molecule_SMARTSIdx_to_atomIdx = self.GetSMARTSIdxToReactantAtomIdx(
            SMARTS_string=new_SMARTS, Reactant_Match_Tuple=tuple(SMART_matches)
        )
        # Retrieve idx of fixed atoms
        fixed_atom_idx_list = []
        for SMART_idx in TS_SMARTS_coordinates:
            fixed_atom_idx_list.append(main_molecule_SMARTSIdx_to_atomIdx[SMART_idx])
        # Calculate trajectories for bringing reactants together into transition state
        atom_label_traj_dict = {}
        for SMARTS_idx in SMARTSIdx_direction_dict:
            direction_vector = SMARTSIdx_direction_dict[SMARTS_idx]
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            atomObj = main_TS_molecule.Atoms[
                main_molecule_SMARTSIdx_to_atomIdx[SMARTS_idx]
            ]
            atomLabel = atomObj.Label
            starting_coordinates = atomObj.Coordinates
            traj_list = []
            for i in range(1, number_of_steps + 1):
                traj_list.append(
                    main_TS_molecule.TranslateAtomByCertainDistanceBasedOnDirection(
                        start=starting_coordinates,
                        direction=direction_vector * -1,
                        distance=stepsize * i,
                    )
                )
            atom_label_traj_dict[atomLabel] = traj_list
        return (
            main_TS_molecule,
            atom_label_traj_dict,
            fixed_atom_idx_list,
            main_molecule_SMARTSIdx_to_atomIdx,
        )

    def FromTSguProduceReacAndProd(
        self,
        main_TS_molecule: Molecule,
        reaction_idx=int,
        TS_state_SMARTS=str,
        products_SMARTS_list=list,
        SMARTSIdx_to_atomIdx=dict,
        reactants_SMARTS_list=list,
        Reac_SMARTS_coordinates=None,
        Prod_SMARTS_coordinates=None,
        output_mol2_file=str,
        GFN_FF_en_tol=1e-6,
        FixAtomsReacAndProd=False,
        OptimiseASE=True,
    ):
        main_TS_molecule.NormaliseAtomLabels()
        main_Reac_molecule = deepcopy(main_TS_molecule)
        main_Reac_molecule.Identifier = f"Reaction_{reaction_idx}_Reac"
        main_Prod_molecule = deepcopy(main_TS_molecule)
        main_Prod_molecule.Identifier = f"Reaction_{reaction_idx}_Prod"
        bonds_broken = self.SMARTS_BondsBroken(
            TS_SMARTS=TS_state_SMARTS,
            products_SMARTS=products_SMARTS_list,
        )
        for bond in bonds_broken:
            atom1Label = main_Prod_molecule.Atoms[
                SMARTSIdx_to_atomIdx[str(bond[0])]
            ].Label
            atomicsymbol1 = main_Prod_molecule.Atoms[
                SMARTSIdx_to_atomIdx[str(bond[0])]
            ].AtomicSymbol
            atom2Label = main_Prod_molecule.Atoms[
                SMARTSIdx_to_atomIdx[str(bond[1])]
            ].Label
            atomicsymbol2 = main_Prod_molecule.Atoms[
                SMARTSIdx_to_atomIdx[str(bond[1])]
            ].AtomicSymbol
            main_Prod_molecule.AdjustBondLength(
                Atom1Label=atom1Label,
                Atom2Label=atom2Label,
                NewBondLength=self.Diatomic_Bond_Lengths[
                    f"{atomicsymbol1}-{atomicsymbol2}"
                ]
                * 2,
            )
            main_Reac_molecule.AdjustBondLength(
                Atom1Label=atom1Label,
                Atom2Label=atom2Label,
                NewBondLength=self.Diatomic_Bond_Lengths[
                    f"{atomicsymbol1}-{atomicsymbol2}"
                ]
                / 2,
            )
            main_Prod_molecule.RemoveBond(
                Atom1Label=atom1Label,
                Atom2Label=atom2Label,
            )

        bonds_formed = self.SMARTS_BondsFormed(
            TS_SMARTS=TS_state_SMARTS, reactants_SMARTS=reactants_SMARTS_list
        )
        for bond in bonds_formed:
            atom1Label = main_Prod_molecule.Atoms[
                SMARTSIdx_to_atomIdx[str(bond[0])]
            ].Label
            atomicsymbol1 = main_Prod_molecule.Atoms[
                SMARTSIdx_to_atomIdx[str(bond[0])]
            ].AtomicSymbol
            atom2Label = main_Prod_molecule.Atoms[
                SMARTSIdx_to_atomIdx[str(bond[1])]
            ].Label
            atomicsymbol2 = main_Prod_molecule.Atoms[
                SMARTSIdx_to_atomIdx[str(bond[1])]
            ].AtomicSymbol
            main_Prod_molecule.AddBond(
                Atom1Label=atom1Label,
                Atom2Label=atom2Label,
                BondType="1",
            )
            main_Prod_molecule.AdjustBondLength(
                Atom1Label=atom1Label,
                Atom2Label=atom2Label,
                NewBondLength=self.Diatomic_Bond_Lengths[
                    f"{atomicsymbol1}-{atomicsymbol2}"
                ]
                / 2,
            )
            main_Reac_molecule.AdjustBondLength(
                Atom1Label=atom1Label,
                Atom2Label=atom2Label,
                NewBondLength=self.Diatomic_Bond_Lengths[
                    f"{atomicsymbol1}-{atomicsymbol2}"
                ]
                * 2,
            )

        change_in_charge = self.SMARTS_ChangeInCharge(
            reactants_SMARTS=reactants_SMARTS_list,
            products_SMARTS=products_SMARTS_list,
        )
        change_in_BondType = self.SMARTS_ChangeInBondType(
            reactants_SMARTS=reactants_SMARTS_list,
            products_SMARTS=products_SMARTS_list,
        )
        for bond in change_in_BondType:
            atom1Label = main_Prod_molecule.Atoms[
                SMARTSIdx_to_atomIdx[str(bond[0][0])]
            ].Label
            atom2Label = main_Prod_molecule.Atoms[
                SMARTSIdx_to_atomIdx[str(bond[0][1])]
            ].Label
            BondType = str(int(bond[1]))
            main_Prod_molecule.AdjustBond(
                Atom1Label=atom1Label,
                Atom2Label=atom2Label,
                BondType=BondType,
            )
        # Before Final optimisation, if SMARTS prod and reac are not == None
        # Fix atoms into position and optimise with UFF openbabel
        if Reac_SMARTS_coordinates != None:
            fixed_atom_idx_list = []
            for SMARTSIdx in Reac_SMARTS_coordinates:
                coordinates = Reac_SMARTS_coordinates[SMARTSIdx]
                atomIdx = SMARTSIdx_to_atomIdx[SMARTSIdx]
                fixed_atom_idx_list.append(atomIdx)
                main_Reac_molecule.Atoms[atomIdx].Coordinates = coordinates
            main_Reac_molecule.OptimizeGeometry_pybel_UFF(
                file_path=output_mol2_file,
                fixed_atoms=fixed_atom_idx_list,
                energy_tol=1e-6,
            )
        # If windows system, the XTB plugin ASE will be used
        if platform.system() == "Windows":
            if OptimiseASE == True:
                if FixAtomsReacAndProd == False:
                    main_Reac_molecule.OptimizeGeometry_ASE_GFN(
                        max_steps=1000,
                        energy_tol=GFN_FF_en_tol,
                        method="GFN-FF",
                    )
                elif FixAtomsReacAndProd == True:
                    main_Reac_molecule.OptimizeGeometry_ASE_GFN(
                        max_steps=1000,
                        energy_tol=GFN_FF_en_tol,
                        method="GFN-FF",
                        fixed_atoms=fixed_atom_idx_list,
                    )
        # If linux system, use the xtb module in command line
        elif platform.system() == "Linux":
            main_Reac_molecule.OptimizeGeometry_ORCA_XTB2(
                max_steps=1000,
                multiplicity=1,
            )
        print(
            """
#########################
Opt of Reactants Complete
#########################
"""
        )
        # Before Final optimisation, if SMARTS prod and reac are not == None
        # Fix atoms into position and optimise with UFF openbabel
        if Prod_SMARTS_coordinates != None:
            fixed_atom_idx_list = []
            for SMARTSIdx in Prod_SMARTS_coordinates:
                coordinates = Prod_SMARTS_coordinates[SMARTSIdx]
                atomIdx = SMARTSIdx_to_atomIdx[SMARTSIdx]
                fixed_atom_idx_list.append(atomIdx)
                main_Prod_molecule.Atoms[atomIdx].Coordinates = coordinates
            main_Prod_molecule.OptimizeGeometry_pybel_UFF(
                file_path=output_mol2_file,
                fixed_atoms=fixed_atom_idx_list,
                energy_tol=1e-6,
            )
        # If windows system, the XTB plugin ASE will be used
        if platform.system() == "Windows":
            if OptimiseASE == True:
                if FixAtomsReacAndProd == False:
                    main_Prod_molecule.OptimizeGeometry_ASE_GFN(
                        max_steps=1000,
                        energy_tol=GFN_FF_en_tol,
                        method="GFN-FF",
                        fixed_atoms=fixed_atom_idx_list,
                    )
                elif FixAtomsReacAndProd == True:
                    main_Prod_molecule.OptimizeGeometry_ASE_GFN(
                        max_steps=1000,
                        energy_tol=GFN_FF_en_tol,
                        method="GFN-FF",
                        fixed_atoms=fixed_atom_idx_list,
                    )
        # If linux system, use the xtb module in command line
        elif platform.system() == "Linux":
            main_Prod_molecule.OptimizeGeometry_ORCA_XTB2(
                max_steps=1000,
                multiplicity=1,
            )
        print(
            """
#########################
Opt of Products Complete
#########################
"""
        )
        return main_Reac_molecule, main_Prod_molecule

    def BuildReaction(
        self,
        Reactants=dict,
        SMARTS_reactant_to_TS=str,
        SMARTS_TS_to_product=str,
        Stereochemical_Information=dict or None,
        TS_SMARTS_coordinates=dict,
        Reac_SMARTS_coordinates=None,
        Prod_SMARTS_coordinates=None,
        output_mol2_file="TSBuildOutput.mol2",
        NumberOfTranslationSteps=50,
        TranslationDistance=5,
        FixAtomsReacAndProd=False,
        OptimiseASE=True,
    ):
        """
        Resturns TS with product and reactant geometries
        configured with the correct geometries and atom indicies
        """
        # Match reactants with reactants SMARTS string
        # Convert SMILES to Molecule Objects
        Reactants_SMILEStoMoleculeDict = self.SMILESToFragments(FragmentDict=Reactants)

        # Create Fragment combinations for the complexes
        Reactants_Combinations = self.CreateFragmentCombinations(
            Fragment_MoleculeDict=Reactants_SMILEStoMoleculeDict
        )
        # Get Reaction SMARTS
        reactants_SMARTS_list = SMARTS_reactant_to_TS.split(">>")[0].split(".")
        products_SMARTS_list = SMARTS_TS_to_product.split(">>")[-1].split(".")
        TS_state_SMARTS = SMARTS_reactant_to_TS.split(">>")[-1]

        # Iterate through each reaction combination produced
        MolSetObj = ReadWriteFiles()
        for reaction_idx, comb in enumerate(Reactants_Combinations):
            # Step 1) Match reactants atom idx to SMART strings
            comb = deepcopy(comb)
            reactant_SMART_matches_dict = self.MatchReactantsToSMARTS(
                reaction_combination=comb,
                reactants_SMARTS_list=reactants_SMARTS_list,
            )
            # If SMARTS matching has returned more than one match
            multiple_matches = self.CheckForMultipleSMARTSMatches(
                SMARTS_matches_dict=reactant_SMART_matches_dict
            )
            if multiple_matches == True:
                # If stereochemical information about the reaction has being provided
                # Use information to discriminate the matches
                # To extract the correct match for the reaction
                if type(Stereochemical_Information) == dict:
                    # Refactor this fuction later to break it down
                    reactant_SMART_matches_dict = (
                        self.FindCorrectAtomsBasedOnStereoInfo(
                            Stereochemical_Information=Stereochemical_Information,
                            reactant_SMARTS_matches_dict=reactant_SMART_matches_dict,
                        )
                    )
            # Return error message if multiple matches are still returned
            multiple_matches = self.CheckForMultipleSMARTSMatches(
                SMARTS_matches_dict=reactant_SMART_matches_dict
            )
            if multiple_matches == True:
                print(
                    """
#####################################################
ERROR: Not enough StereoChemical information is given
Multiple SMARTS to molecule matches are returned
#####################################################
"""
                )
                continue
            # Align Molcules with SMARTS TS coordinates
            (
                reactant_SMART_matches_dict,
                no_SMART_matches,
                SMARTS_idx_to_AtomIdx_molObj_dict,
            ) = self.AlignMoleculeWithSMARTSCoordinates_v2(
                reactant_SMART_matches_dict=reactant_SMART_matches_dict,
                SMARTSIdx_Coordinates=TS_SMARTS_coordinates,
            )
            if no_SMART_matches == True:
                print(
                    """
#####################################################
ERROR: Incorrect StereoChemical information or
Incorrect SMARTS information given, so does not match
up with chemical structure provided
Multiple SMARTS to molecule matches are returned
#####################################################
"""
                )
                continue
            (
                main_TS_molecule,
                atom_label_traj_dict,
                fixed_atom_idx_list,
                SMARTSIdx_to_atomIdx,
            ) = self.PrepareTranslationToBuildTS(
                TS_SMARTS_coordinates=TS_SMARTS_coordinates,
                SMARTS_idx_to_AtomIdx_molObj_dict=SMARTS_idx_to_AtomIdx_molObj_dict,
                reactant_SMART_matches_dict=reactant_SMART_matches_dict,
                reaction_idx=reaction_idx,
                number_of_steps=NumberOfTranslationSteps,
                translation_distance=TranslationDistance,
            )
            # Optimisation parameters
            GFN_FF_en_tol = 1e-4
            # Pre-optimise before translation to build TS structure
            main_TS_molecule.OptimizeGeometry_pybel_UFF(
                file_path=output_mol2_file,
                fixed_atoms=fixed_atom_idx_list,
                energy_tol=1e-6,
            )
            print(
                """
################
Pre-Opt Complete
################
"""
            )
            # Optimise along trajectory to build TS structure
            for step in range(0, NumberOfTranslationSteps):
                for atom_label in atom_label_traj_dict:
                    position = atom_label_traj_dict[atom_label][step]
                    main_TS_molecule.AtomsDict[atom_label][1].Coordinates = position
                main_TS_molecule.OptimizeGeometry_pybel_UFF(
                    file_path=output_mol2_file,
                    fixed_atoms=fixed_atom_idx_list,
                    energy_tol=1e-6,
                )
            # If windows system, the XTB plugin ASE will be used
            if platform.system() == "Windows":
                if OptimiseASE == True:
                    main_TS_molecule.OptimizeGeometry_ASE_GFN(
                        fixed_atoms=fixed_atom_idx_list,
                        max_steps=1000,
                        energy_tol=GFN_FF_en_tol,
                        method="GFN-FF",
                    )
            # If linux system, use the xtb module in conda command line
            elif platform.system() == "Linux":
                main_TS_molecule.OptimizeGeometry_ORCA_XTB2(
                    fixed_atoms=fixed_atom_idx_list,
                    max_steps=1000,
                    multiplicity=1,
                )
            print(
                """
##################
Opt of TS Complete
##################
"""
            )
            main_Reac_molecule, main_Prod_molecule = self.FromTSguProduceReacAndProd(
                main_TS_molecule=main_TS_molecule,
                reaction_idx=reaction_idx,
                TS_state_SMARTS=TS_state_SMARTS,
                products_SMARTS_list=products_SMARTS_list,
                SMARTSIdx_to_atomIdx=SMARTSIdx_to_atomIdx,
                reactants_SMARTS_list=reactants_SMARTS_list,
                Reac_SMARTS_coordinates=Reac_SMARTS_coordinates,
                Prod_SMARTS_coordinates=Prod_SMARTS_coordinates,
                output_mol2_file=output_mol2_file,
                GFN_FF_en_tol=1e-6,
                FixAtomsReacAndProd=FixAtomsReacAndProd,
                OptimiseASE=False,
            )
            MolSetObj.MoleculeDict[main_Reac_molecule.Identifier] = main_Reac_molecule
            MolSetObj.MoleculeList.append(main_Reac_molecule)
            MolSetObj.MoleculeDict[main_TS_molecule.Identifier] = main_TS_molecule
            MolSetObj.MoleculeList.append(main_TS_molecule)
            MolSetObj.MoleculeDict[main_Prod_molecule.Identifier] = main_Prod_molecule
            MolSetObj.MoleculeList.append(main_Prod_molecule)
        MolSetObj.WriteMol2File(
            output_mol2_file_name=output_mol2_file,
        )

    def MeasureRMSD_molObjVersusSMARTS(
        self, molObj: Molecule, SMARTSIdx_to_AtomIdx=dict, SMARTSIdx_Coordinates=dict
    ):
        SMARTS_x = []
        SMARTS_y = []
        SMARTS_z = []
        Atom_x = []
        Atom_y = []
        Atom_z = []
        length = 0
        for SMARTSIdx in SMARTSIdx_to_AtomIdx:
            atomIdx = SMARTSIdx_to_AtomIdx[SMARTSIdx]
            try:
                SMARTS_x.append(SMARTSIdx_Coordinates[SMARTSIdx][0])
                SMARTS_y.append(SMARTSIdx_Coordinates[SMARTSIdx][1])
                SMARTS_z.append(SMARTSIdx_Coordinates[SMARTSIdx][2])
                Atom_x.append(molObj.Atoms[atomIdx].Coordinates[0])
                Atom_y.append(molObj.Atoms[atomIdx].Coordinates[1])
                Atom_z.append(molObj.Atoms[atomIdx].Coordinates[2])
                length += 1
            except KeyError:
                pass
        RMSD = np.sqrt(
            (
                ((np.array(SMARTS_x) - np.array(Atom_x)) ** 2)
                + ((np.array(SMARTS_y) - np.array(Atom_y)) ** 2)
                + ((np.array(SMARTS_z) - np.array(Atom_z)) ** 2)
            ).sum()
            / length
        )
        return RMSD

    def AlignMoleculeWithSMARTSCoordinates_v2(
        self,
        reactant_SMART_matches_dict=dict,
        SMARTSIdx_Coordinates=dict,
    ):
        # Align molecule into transition state positions
        # Before Aligning molecules into position
        # Need to create dictionary of SMARTS idx to match idx and molecule object
        # Need to create tracker dictionary
        # To track whereever the molecule needs to be translated or rotated
        SMARTS_idx_to_AtomIdx_molObj_dict = {}
        tracker_dict = {}
        for reactant_id in reactant_SMART_matches_dict:
            reactant_info_dict = reactant_SMART_matches_dict[reactant_id]
            molObj = reactant_info_dict["Reactant"]
            SMARTS_idx_to_molIdx_dict = self.GetSMARTSIdxToReactantAtomIdx(
                SMARTS_string=reactant_info_dict["Reactant_SMARTS"],
                Reactant_Match_Tuple=reactant_info_dict["Matches"][0],
            )
            for SMARTS_idx in SMARTS_idx_to_molIdx_dict:
                AtomIdx = SMARTS_idx_to_molIdx_dict[SMARTS_idx]
                SMARTS_idx_to_AtomIdx_molObj_dict[SMARTS_idx] = {
                    "Reactant_ID": reactant_id,
                    "Reactant": molObj,
                    "Atom_Idx": AtomIdx,
                }
            tracker_dict[reactant_id] = {
                "Translated?": "Not_Translated",
                "Yesterdays_Atom_Idx": None,
            }
        # With tracker dictionary and smarts idx to atom idx and mol object dict
        # align molecule to TS atomic positions
        no_SMART_matches = False
        for TS_SMART_idx in SMARTSIdx_Coordinates:
            try:
                atomIdx_molObj_dict = SMARTS_idx_to_AtomIdx_molObj_dict[TS_SMART_idx]
                template_coordinate = SMARTSIdx_Coordinates[TS_SMART_idx]
                molObj = atomIdx_molObj_dict["Reactant"]
                atomIdx = atomIdx_molObj_dict["Atom_Idx"]
                reactant_id = atomIdx_molObj_dict["Reactant_ID"]
                if tracker_dict[reactant_id]["Translated?"] == "Not_Translated":
                    molObj.TranslateMolecule(
                        TranslationVector=molObj.Atoms[atomIdx].Coordinates
                    )
                    molObj.TranslateMolecule(TranslationVector=template_coordinate * -1)
                    tracker_dict[reactant_id]["Translated?"] = "Translated"
                    tracker_dict[reactant_id]["Yesterdays_Atom_Idx"] = atomIdx
                elif tracker_dict[reactant_id]["Translated?"] == "Translated":
                    yesterdays_atom_idx = tracker_dict[reactant_id][
                        "Yesterdays_Atom_Idx"
                    ]
                    molObj.CalcNegThetaCalcCrossRotateMoleculeAroundCentre(
                        StartingPosition=molObj.Atoms[atomIdx].Coordinates,
                        EndPosition=template_coordinate,
                        RotationCentre=molObj.Atoms[yesterdays_atom_idx].Coordinates,
                    )
                    tracker_dict[reactant_id]["Yesterdays_Atom_Idx"] = atomIdx
            except KeyError:
                no_SMART_matches = True
                return (
                    reactant_SMART_matches_dict,
                    no_SMART_matches,
                    SMARTS_idx_to_AtomIdx_molObj_dict,
                )
        # Now that Molecule is aligned with cooridinates via simple translation then rotation algorithm
        # Translate atoms into positions at 0.1 Angstrom step sizes
        for reactant_id in reactant_SMART_matches_dict:
            atomIdx_tuple = reactant_SMART_matches_dict[reactant_id]["Matches"][0]
            Reactant_SMARTS = reactant_SMART_matches_dict[reactant_id][
                "Reactant_SMARTS"
            ]
            molObj = reactant_SMART_matches_dict[reactant_id]["Reactant"]
            SMARTSIdx_to_AtomIdx = self.GetSMARTSIdxToReactantAtomIdx(
                SMARTS_string=Reactant_SMARTS,
                Reactant_Match_Tuple=atomIdx_tuple,
            )
            RMSD = self.MeasureRMSD_molObjVersusSMARTS(
                molObj=molObj,
                SMARTSIdx_to_AtomIdx=SMARTSIdx_to_AtomIdx,
                SMARTSIdx_Coordinates=SMARTSIdx_Coordinates,
            )
            if RMSD != 0:
                # move atoms into positions
                # Calculate trajectories between SMARTS positions and atom positions
                # Step 1) Calculate direction vectors and their magnitudes
                pre_direction_vector_dict = {}
                for SMARTSIdx in SMARTSIdx_to_AtomIdx:
                    temp_dict = {}
                    AtomIdx = SMARTSIdx_to_AtomIdx[SMARTSIdx]
                    AtomLabel = molObj.Atoms[AtomIdx].Label
                    SMARTS_coordinates = SMARTSIdx_Coordinates[SMARTSIdx]
                    atom_coordinates = molObj.Atoms[AtomIdx].Coordinates
                    direction_vector = SMARTS_coordinates - atom_coordinates
                    direction_vector_mag = np.linalg.norm(direction_vector)
                    direction_vector_norm = direction_vector / direction_vector_mag
                    temp_dict["Direction_Vector_Magnitude"] = direction_vector_mag
                    temp_dict["Direction_Vector_Normalised"] = direction_vector_norm
                    pre_direction_vector_dict[AtomLabel] = temp_dict
                # Step 2) Find the number of steps required, stepsize is 0.1 Angstrom
                stepsize = 0.1
                stepcount_list = []
                for AtomLabel in pre_direction_vector_dict:
                    length_of_travel = pre_direction_vector_dict[AtomLabel][
                        "Direction_Vector_Magnitude"
                    ]
                    stepcount_list.append(int(length_of_travel / stepsize))
                number_of_steps = max(stepcount_list)
                # Step 3) Calculate the trajectories
                atom_label_traj_dict = {}
                for AtomLabel in pre_direction_vector_dict:
                    trajectory_list = []
                    length_of_travel = pre_direction_vector_dict[AtomLabel][
                        "Direction_Vector_Magnitude"
                    ]
                    stepsize = length_of_travel / number_of_steps
                    start = molObj.AtomsDict[AtomLabel][1].Coordinates
                    direction = pre_direction_vector_dict[AtomLabel][
                        "Direction_Vector_Normalised"
                    ]
                    for stepnumber in range(1, number_of_steps + 1):
                        trajectory_list.append(
                            molObj.TranslateAtomByCertainDistanceBasedOnDirection(
                                start, direction, stepsize * stepnumber
                            )
                        )
                    atom_label_traj_dict[AtomLabel] = trajectory_list
                # Step 4) Optimise the molecule into position via the trajectories
                fixed_atom_idx_list = [
                    molObj.AtomsDict[AtomLabel][0] for AtomLabel in atom_label_traj_dict
                ]
                for step in range(0, number_of_steps):
                    for atom_label in atom_label_traj_dict:
                        position = atom_label_traj_dict[atom_label][step]
                        molObj.AtomsDict[atom_label][1].Coordinates = position
                    molObj.OptimizeGeometry_pybel_UFF(
                        file_path="",
                        fixed_atoms=fixed_atom_idx_list,
                        energy_tol=1e-6,
                    )
                # Step 4) Move atoms into position
                pass
        return (
            reactant_SMART_matches_dict,
            no_SMART_matches,
            SMARTS_idx_to_AtomIdx_molObj_dict,
        )


class ExercuteWorkFlow:
    def __init__(self):
        pass

    def BuildReaction_FindTSWithNEB(
        self,
        Reactants=dict,
        SMARTS_reactant_to_TS=str,
        SMARTS_TS_to_product=str,
        Stereochemical_Information=dict or None,
        TS_SMARTS_coordinates=dict,
        Reaction_Name=str,
        Output_File_Name=str,
        Path=str,
        github_token="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",
        time="01:00:00",
        RAM_per_CPU=2,
        CPU_num=1,
    ):
        main_bash_string = "git clone https://mace12345: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@github.com/mace12345/Project\n"
        # Match reactants with reactants SMARTS string
        # Convert SMILES to Molecule Objects
        reaction_builder = MoleculeBuilder()
        Reactants_SMILEStoMoleculeDict = reaction_builder.SMILESToFragments(
            FragmentDict=Reactants
        )
        # Create Fragment combinations for the complexes
        Reactants_Combinations = reaction_builder.CreateFragmentCombinations(
            Fragment_MoleculeDict=Reactants_SMILEStoMoleculeDict
        )
        # Get Reaction SMARTS
        reactants_SMARTS_list = SMARTS_reactant_to_TS.split(">>")[0].split(".")
        products_SMARTS_list = SMARTS_TS_to_product.split(">>")[-1].split(".")
        TS_state_SMARTS = SMARTS_reactant_to_TS.split(">>")[-1]
        try:
            os.mkdir(f"{Path}/{Output_File_Name}")
        except FileExistsError:
            pass
        for idx, comb in enumerate(Reactants_Combinations):
            # Write folder bash script
            # Step 1) download and rename https://raw.githubusercontent.com/mace12345/Project/refs/heads/main/MoleculeHandler.py?token=GHSAT0AAAAAAC4ZYHZL4EP3ZRZOS6AR56KYZ4ODN4A
            # Step 2) python build reaction then create orca input files
            # Step 3) run TSNEB calc with orca
            # Step 4) Check results with another python file for TS frequency displacements
            # Step 5) write log file with results
            local_bash = f"""#!/bin/bash
#SBATCH --job-name=TSNEB-J{idx}
#SBATCH --time={time}
#SBATCH --mem={RAM_per_CPU}G
#SBATCH --ntasks={CPU_num}
#SBATCH --cpus-per-task=1

git clone https://mace12345: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@github.com/mace12345/Project
mv Project/MoleculeHandler.py MoleculeHandler.py

module load miniforge
conda activate chem-env

python MakePreTSNEBMolFile.py
rm *_Temp.*
rm *_Temp_trj.xyz

/users/cmsma/orca/orca_6_0_0_shared_openmpi416/orca Reaction_0.inp > Reaction_0.out
/users/cmsma/orca/orca_6_0_0_shared_openmpi416/orca_pltvib Reaction_0.hess 1 2 3 4 5 6
rm *.tmp
rm *.gbw
rm *.interp
rm *.bas*
rm *.carthess
rm *.opt
rm *.allxyz
rm *.gradient
rm *.mol
rm *.xtbrestart
rm *.charges
rm *.txt
rm *.hostnames
rm *.vibspectrum
rm *.xtberr
rm *.bibtex
rm *.log

rm -rf Project
rm MoleculeHandler.py
"""
            # Write mol2 files of the reactants
            try:
                os.mkdir(f"{Path}/{Output_File_Name}/{Reaction_Name}_{idx}")
            except FileExistsError:
                pass
            reactants_string = """Reactants={
"""
            for reactant_id in comb:
                write_mol2_file = ReadWriteFiles()
                reactant_MolObj = comb[reactant_id]
                write_mol2_file.MoleculeDict[reactant_id] = reactant_MolObj
                write_mol2_file.MoleculeList.append(reactant_MolObj)
                write_mol2_file.WriteMol2File(
                    output_mol2_file_name=f"{Path}/{Output_File_Name}/{Reaction_Name}_{idx}/{reactant_id}.mol2"
                )
                reactants_string = (
                    reactants_string
                    + f"        '{reactant_id}': "
                    + "{\n"
                    + "            'MOL2FILE': ['"
                    + f"{reactant_id}.mol2"
                    + "'],"
                    + "\n        },\n"
                )
            reactants_string = reactants_string + "    },"
            stereo_string = json.dumps(Stereochemical_Information, indent=4).replace(
                "\n", "\n    "
            )
            serializable_data = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in TS_SMARTS_coordinates.items()
            }

            # Serialize to JSON
            TS_coor_string = (
                json.dumps(serializable_data, indent=4)
                .replace("\n", "\n    ")
                .replace("[", "np.array([")
                .replace("]", "])")
            )
            # Write python file to Build reaction
            build_reaction_string = f"""from MoleculeHandler import MoleculeBuilder as MB
from MoleculeHandler import ReadWriteFiles as RW
import numpy as np

# Build TS-NEB .mol2 for input coordinates for ORCA6 TS-NEB calc
buildreaction = MB()
buildreaction.BuildReaction(
    {reactants_string}
    SMARTS_reactant_to_TS="{SMARTS_reactant_to_TS}",
    SMARTS_TS_to_product="{SMARTS_TS_to_product}",
    Stereochemical_Information={stereo_string},
    TS_SMARTS_coordinates={TS_coor_string},
    output_mol2_file="PreTSNEB.mol2",
)

# Prepare PreTSNEB.mol2 for ORCA6 input
readwrite = RW()
readwrite.ReadMol2File(mol2_file="PreTSNEB.mol2")
readwrite.Write_ORCA6_TSNEB_Input_For_ExercuteWorkFlow(
    multiplicity=1,
    method="XTB2",
    basis_set="",
    keywords="Freq",
    PreOpt=True,
    NumImages=8,
    ORCA_exp_path="/users/cmsma/orca/orca_6_0_0_shared_openmpi416/orca",
)
"""
            with open(
                f"{Path}/{Output_File_Name}/{Reaction_Name}_{idx}/MakePreTSNEBMolFile.py",
                "w",
            ) as f:
                f.write(build_reaction_string)
            with open(
                f"{Path}/{Output_File_Name}/{Reaction_Name}_{idx}/local_bash.sh",
                "w",
            ) as f:
                f.write(local_bash)

    def BuildReactionOnAIRE(
        self,
        Reactants=dict,
        SMARTS_reactant_to_TS=str,
        SMARTS_TS_to_product=str,
        Stereochemical_Information=dict or None,
        TS_SMARTS_coordinates=dict,
        Reaction_Name=str,
        Output_File_Name=str,
        Path=str,
        github_token="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",
        time="01:00:00",
        RAM_per_CPU=2,
        CPU_num=1,
    ):
        main_bash_string = "git clone https://mace12345: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@github.com/mace12345/Project\n"
        # Match reactants with reactants SMARTS string
        # Convert SMILES to Molecule Objects
        reaction_builder = MoleculeBuilder()
        Reactants_SMILEStoMoleculeDict = reaction_builder.SMILESToFragments(
            FragmentDict=Reactants
        )
        # Create Fragment combinations for the complexes
        Reactants_Combinations = reaction_builder.CreateFragmentCombinations(
            Fragment_MoleculeDict=Reactants_SMILEStoMoleculeDict
        )
        # Get Reaction SMARTS
        reactants_SMARTS_list = SMARTS_reactant_to_TS.split(">>")[0].split(".")
        products_SMARTS_list = SMARTS_TS_to_product.split(">>")[-1].split(".")
        TS_state_SMARTS = SMARTS_reactant_to_TS.split(">>")[-1]
        try:
            os.mkdir(f"{Path}/{Output_File_Name}")
        except FileExistsError:
            pass
        for idx, comb in enumerate(Reactants_Combinations):
            # Write folder bash script
            # Step 1) download and rename https://raw.githubusercontent.com/mace12345/Project/refs/heads/main/MoleculeHandler.py?token=GHSAT0AAAAAAC4ZYHZL4EP3ZRZOS6AR56KYZ4ODN4A
            # Step 2) python build reaction then create orca input files
            # Step 3) run TSNEB calc with orca
            # Step 4) Check results with another python file for TS frequency displacements
            # Step 5) write log file with results
            local_bash = f"""#!/bin/bash
#SBATCH --job-name=RB-{idx}
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

git clone https://mace12345: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@github.com/mace12345/Project
mv Project/MoleculeHandler.py MoleculeHandler.py

module load miniforge
conda activate chem-env

python MakePreTSNEBMolFile.py

rm *_Temp.*
rm *_Temp_trj.xyz
rm -rf Project
rm MoleculeHandler.py
"""
            # Write mol2 files of the reactants
            try:
                os.mkdir(f"{Path}/{Output_File_Name}/{Reaction_Name}_{idx}")
            except FileExistsError:
                pass
            reactants_string = """Reactants={
"""
            for reactant_id in comb:
                write_mol2_file = ReadWriteFiles()
                reactant_MolObj = comb[reactant_id]
                write_mol2_file.MoleculeDict[reactant_id] = reactant_MolObj
                write_mol2_file.MoleculeList.append(reactant_MolObj)
                write_mol2_file.WriteMol2File(
                    output_mol2_file_name=f"{Path}/{Output_File_Name}/{Reaction_Name}_{idx}/{reactant_id}.mol2"
                )
                reactants_string = (
                    reactants_string
                    + f"        '{reactant_id}': "
                    + "{\n"
                    + "            'MOL2FILE': ['"
                    + f"{reactant_id}.mol2"
                    + "'],"
                    + "\n        },\n"
                )
            reactants_string = reactants_string + "    },"
            stereo_string = json.dumps(Stereochemical_Information, indent=4).replace(
                "\n", "\n    "
            )
            serializable_data = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in TS_SMARTS_coordinates.items()
            }

            # Serialize to JSON
            TS_coor_string = (
                json.dumps(serializable_data, indent=4)
                .replace("\n", "\n    ")
                .replace("[", "np.array([")
                .replace("]", "])")
            )
            # Write python file to Build reaction
            build_reaction_string = f"""from MoleculeHandler import MoleculeBuilder as MB
from MoleculeHandler import ReadWriteFiles as RW
import numpy as np

# Build TS-NEB .mol2 for input coordinates for ORCA6 TS-NEB calc
buildreaction = MB()
buildreaction.BuildReaction(
    {reactants_string}
    SMARTS_reactant_to_TS="{SMARTS_reactant_to_TS}",
    SMARTS_TS_to_product="{SMARTS_TS_to_product}",
    Stereochemical_Information={stereo_string},
    TS_SMARTS_coordinates={TS_coor_string},
    output_mol2_file="PreTSNEB.mol2",
)
"""
            with open(
                f"{Path}/{Output_File_Name}/{Reaction_Name}_{idx}/MakePreTSNEBMolFile.py",
                "w",
            ) as f:
                f.write(build_reaction_string)
            with open(
                f"{Path}/{Output_File_Name}/{Reaction_Name}_{idx}/local_bash.sh",
                "w",
            ) as f:
                f.write(local_bash)

    def CustomFunctionForKarendip(self, mol2_file=str):
        molecules = ReadWriteFiles()
        molecules.ReadMol2File(
            mol2_file=mol2_file,
        )
        results_df = pd.DataFrame()
        for molecule in molecules.MoleculeList:
            # Find nitrogen atom (N.3 or N.pl)
            # Neighbours must be C.ar and 2x C.3
            for atom in molecule.Atoms:
                if atom.SybylType == "N.3" or atom.SybylType == "N.pl3":
                    natoms = molecule.GetAtomNeighbours(AtomLabel=atom.Label)
                    # C.3 count == 2
                    c3_count = 0
                    # C.ar count == 1
                    Car_count = 0
                    Car_adj_count = 0
                    for natom in natoms:
                        natom = molecule.AtomsDict[natom][1]
                        if natom.SybylType == "C.3" and c3_count == 0:
                            C3_atom1 = natom
                            c3_count += 1
                        elif natom.SybylType == "C.3" and c3_count == 1:
                            C3_atom2 = natom
                            c3_count += 1
                        elif natom.SybylType == "C.ar" and Car_count == 0:
                            Car_atom = natom
                            Car_count += 1
                            for ar_natom in molecule.GetAtomNeighbours(
                                AtomLabel=Car_atom.Label
                            ):
                                ar_natom = molecule.AtomsDict[ar_natom][1]
                                if (
                                    ar_natom.SybylType.split(".")[-1] == "ar"
                                    and Car_adj_count == 0
                                ):
                                    Car_adj1_atom = ar_natom
                                    Car_adj_count += 1
                                elif (
                                    ar_natom.SybylType.split(".")[-1] == "ar"
                                    and Car_adj_count == 1
                                ):
                                    Car_adj2_atom = ar_natom
                                    Car_adj_count += 1
                    if c3_count == 2 and Car_count == 1 and Car_adj_count == 2:
                        N3_or_pl3_atom = atom
                        break

            # Measure sp3 character angle
            C31_C32_angle = np.degrees(
                molecule.FindBondAngle(
                    a=C3_atom1.Coordinates,
                    b=N3_or_pl3_atom.Coordinates,
                    c=C3_atom2.Coordinates,
                )
            )
            results_df.loc[molecule.Identifier, "Atom Labels in C.3-N-C.3 angle"] = (
                f"{C3_atom1.Label}-{N3_or_pl3_atom.Label}-{C3_atom2.Label}"
            )
            results_df.loc[molecule.Identifier, "C.3-N-C.3 bond angle / deg"] = round(
                C31_C32_angle, 4
            )
            Car_C31_angle = np.degrees(
                molecule.FindBondAngle(
                    a=C3_atom1.Coordinates,
                    b=N3_or_pl3_atom.Coordinates,
                    c=Car_atom.Coordinates,
                )
            )
            results_df.loc[molecule.Identifier, "Atom Labels in C.3-N-C.ar angle"] = (
                f"{C3_atom1.Label}-{N3_or_pl3_atom.Label}-{Car_atom.Label}"
            )
            results_df.loc[molecule.Identifier, "C.3-N-C.ar bond angle / deg"] = round(
                Car_C31_angle, 4
            )
            Car_C32_angle = np.degrees(
                molecule.FindBondAngle(
                    a=C3_atom2.Coordinates,
                    b=N3_or_pl3_atom.Coordinates,
                    c=Car_atom.Coordinates,
                )
            )
            results_df.loc[molecule.Identifier, "Atom Labels in C.ar-N-C.3 angle"] = (
                f"{Car_atom.Label}-{N3_or_pl3_atom.Label}-{C3_atom2.Label}"
            )
            results_df.loc[molecule.Identifier, "C.ar-N-C.3 bond angle / deg"] = round(
                Car_C32_angle, 4
            )
            results_df.loc[molecule.Identifier, "N's sp3 character angle sum / deg"] = (
                round(C31_C32_angle + Car_C31_angle + Car_C32_angle, 4)
            )
            # Measure sp3 character lone pair vector magnitude
            C31_vector = C3_atom1.Coordinates - N3_or_pl3_atom.Coordinates
            C31_norm_vector = C31_vector / np.linalg.norm(C31_vector)
            C32_vector = C3_atom2.Coordinates - N3_or_pl3_atom.Coordinates
            C32_norm_vector = C32_vector / np.linalg.norm(C32_vector)
            Car_vector = Car_atom.Coordinates - N3_or_pl3_atom.Coordinates
            Car_norm_vector = Car_vector / np.linalg.norm(Car_vector)
            N_lone_pair_vector = (
                C31_norm_vector + C32_norm_vector + Car_norm_vector
            ) * -1
            N_lone_pair_mag = np.linalg.norm(N_lone_pair_vector)
            results_df.loc[
                molecule.Identifier, "N's sp3 character resultant lone pair mag / AU"
            ] = round(N_lone_pair_mag, 4)
            # Find Cross Product of C.ar atom adjacent to N
            Car_cross = np.cross(
                Car_adj1_atom.Coordinates - Car_atom.Coordinates,
                Car_adj2_atom.Coordinates - Car_atom.Coordinates,
            )
            # Find Angle between C.ar cross product and N sp3 lone pair
            phi = np.degrees(
                molecule.FindBondAngle(
                    a=N_lone_pair_vector,
                    b=Car_cross,
                    c=None,
                )
            )
            if phi > 90:
                phi = 180 - phi
            results_df.loc[
                molecule.Identifier, "N-lone-pair-vector C.ar-cross-product Phi / deg"
            ] = round(phi, 4)
            # Find Angle between C.ar cross product and C.3-N-C.3 cross product
            C31_N_C32_cross = np.cross(
                C3_atom1.Coordinates - N3_or_pl3_atom.Coordinates,
                C3_atom2.Coordinates - N3_or_pl3_atom.Coordinates,
            )
            phi = np.degrees(
                molecule.FindBondAngle(
                    a=C31_N_C32_cross,
                    b=Car_cross,
                    c=None,
                )
            )
            if phi > 90:
                phi = 180 - phi
            results_df.loc[
                molecule.Identifier,
                "C.3-N-C.3-cross-product C.ar-cross-product Phi / deg",
            ] = round(phi, 4)
            # Find Dihedral between C.ar cross product, C.ar, N and C.3-N-C.3 cross product
            dihedral_1 = abs(
                np.degrees(
                    molecule.FindTorsionAngle(
                        a=C31_N_C32_cross + N3_or_pl3_atom.Coordinates,
                        b=N3_or_pl3_atom.Coordinates,
                        c=Car_atom.Coordinates,
                        d=Car_cross + Car_atom.Coordinates,
                    )
                )
            )
            if dihedral_1 > 90:
                dihedral_1 = 180 - dihedral_1
            results_df.loc[
                molecule.Identifier,
                "C.ar-cross-product C.ar N C.3-N-C.3-cross-product / deg",
            ] = round(dihedral_1, 4)
            # Find Dihedral between C.ar cross product, C.ar, N and N lone pair vector
            dihedral_2 = abs(
                np.degrees(
                    molecule.FindTorsionAngle(
                        a=N_lone_pair_vector + N3_or_pl3_atom.Coordinates,
                        b=N3_or_pl3_atom.Coordinates,
                        c=Car_atom.Coordinates,
                        d=Car_cross + Car_atom.Coordinates,
                    )
                )
            )
            if dihedral_2 > 90:
                dihedral_2 = 180 - dihedral_2
            results_df.loc[
                molecule.Identifier,
                "C.ar-cross-product C.ar N N-lone-pair-vector / deg",
            ] = round(dihedral_1, 4)
            # Find 2D phi angle between C.ar cross product and N lone pair vector
            molecule.TranslateMolecule(TranslationVector=Car_atom.Coordinates)
            for _ in range(0, 100):
                molecule.CalcNegThetaCalcCrossRotateMolecule(
                    StartingPosition=Car_cross, EndPosition=np.array([1, 0, 0])
                )
                molecule.CalcNegThetaCalcCrossRotateMolecule(
                    StartingPosition=N3_or_pl3_atom.Coordinates,
                    EndPosition=np.array([0, 1, 0]),
                )
                Car_cross = np.cross(
                    Car_adj1_atom.Coordinates - Car_atom.Coordinates,
                    Car_adj2_atom.Coordinates - Car_atom.Coordinates,
                )
            C31_vector = C3_atom1.Coordinates - N3_or_pl3_atom.Coordinates
            C31_norm_vector = C31_vector / np.linalg.norm(C31_vector)
            C32_vector = C3_atom2.Coordinates - N3_or_pl3_atom.Coordinates
            C32_norm_vector = C32_vector / np.linalg.norm(C32_vector)
            Car_vector = Car_atom.Coordinates - N3_or_pl3_atom.Coordinates
            Car_norm_vector = Car_vector / np.linalg.norm(Car_vector)
            N_lone_pair_vector = (
                C31_norm_vector + C32_norm_vector + Car_norm_vector
            ) * -1
            Car_cross[2] = 0
            N_lone_pair_vector[2] = 0
            phi_2D = np.degrees(
                molecule.FindBondAngle(
                    a=N_lone_pair_vector,
                    b=Car_cross,
                    c=None,
                )
            )
            if phi_2D > 90:
                phi_2D = 180 - phi_2D
            results_df.loc[
                molecule.Identifier,
                "N-lone-pair-vector C.ar-cross-product Phi-2D (z-component = 0) / deg",
            ] = round(phi_2D, 4)
        results_df.reset_index(inplace=True)
        results_df.rename(columns={"index": "Identifier"}, inplace=True)
        results_df.set_index("Identifier", inplace=True)
        results_df.to_csv(mol2_file.split(".")[0] + ".csv")


class ReadWriteFiles:
    def __init__(self):
        self.bond_types_to_bond_order_dict = {
            "1": 1,
            "2": 2,
            "3": 3,
            "am": None,
            "ar": 1.5,
            "du": 1,
            "un": 1,
            "nc": 0,
        }
        self.MoleculeList = []
        self.MoleculeDict = {}
        self.mol2_file_path = None

    def ReadMol2File(self, mol2_file=str):
        new_file_path = ""
        for string in mol2_file.split("/")[:-1]:
            new_file_path = new_file_path + f"{string}/"
        self.mol2_file_path = new_file_path
        with open(mol2_file, "r") as f:
            file = f.read()

        molecule_list = []
        molecule_string_list = [i for i in file.split("@<TRIPOS>MOLECULE\n") if i != ""]
        if len(molecule_string_list) == 0:
            print(
                """
##############
.mol2 is empty
##############
"""
            )
        for molecule_string in molecule_string_list:
            molucule_info_string = molecule_string.split("@<TRIPOS>ATOM\n")[0]
            molecule_atom_string = molecule_string.split("@<TRIPOS>ATOM\n")[-1].split(
                "@"
            )[0]
            molecule_bond_string = molecule_string.split("@<TRIPOS>BOND\n")[-1].split(
                "@"
            )[0]

            identifier = molucule_info_string.split("\n")[0]
            atom_bond_number = [
                i for i in molucule_info_string.split("\n")[1].split(" ") if i != ""
            ]
            number_of_atoms = int(atom_bond_number[0])
            number_of_bonds = int(atom_bond_number[1])
            number_of_substructures = int(atom_bond_number[2])

            molecule_atom_list = [
                [i for i in j.split(" ") if i != ""]
                for j in molecule_atom_string.split("\n")
            ]

            atom_list = []
            for atom in molecule_atom_list:
                if len(atom) == 0:
                    continue
                atom_list.append(
                    Atom(
                        Label=atom[1],
                        Coordinates=np.array(
                            [
                                float(atom[2]),
                                float(atom[3]),
                                float(atom[4]),
                            ]
                        ),
                        SybylType=atom[5],
                        AtomicSymbol=atom[5].split(".")[0],
                        SubstructureIndex=atom[6],
                        SubstructureName=atom[7],
                        FormalCharge=int(float(atom[8])),
                    )
                )

            molecule_bond_list = [
                [i for i in j.split(" ") if i != ""]
                for j in molecule_bond_string.split("\n")
            ]
            if len(molecule_bond_list[0]) == 4:
                connectivity_matrix = np.zeros((number_of_atoms, number_of_atoms))
                bond_order_matrix = np.zeros((number_of_atoms, number_of_atoms))
                bond_type_matrix = np.zeros((number_of_atoms, number_of_atoms)).astype(
                    str
                )
                for bond in molecule_bond_list:
                    if len(bond) == 0:
                        continue
                    atom1_index = int(bond[1])
                    atom2_index = int(bond[2])
                    bond_type = bond[3]
                    connectivity_matrix[atom1_index - 1][atom2_index - 1] = 1
                    connectivity_matrix[atom2_index - 1][atom1_index - 1] = 1
                    # Bond Order for "N.am", "O.am" and "C.am" atoms will need to be sorted
                    if bond_type == "am":
                        if (
                            atom_list[atom1_index - 1].AtomicSymbol == "C"
                            and atom_list[atom2_index - 1].AtomicSymbol == "N"
                        ):
                            bond_order_matrix[atom1_index - 1][atom2_index - 1] = 1
                        elif (
                            atom_list[atom2_index - 1].AtomicSymbol == "C"
                            and atom_list[atom1_index - 1].AtomicSymbol == "N"
                        ):
                            bond_order_matrix[atom1_index - 1][atom2_index - 1] = 1
                        if (
                            atom_list[atom1_index - 1].AtomicSymbol == "C"
                            and atom_list[atom2_index - 1].AtomicSymbol == "O"
                        ):
                            bond_order_matrix[atom1_index - 1][atom2_index - 1] = 2
                        elif (
                            atom_list[atom2_index - 1].AtomicSymbol == "C"
                            and atom_list[atom1_index - 1].AtomicSymbol == "O"
                        ):
                            bond_order_matrix[atom1_index - 1][atom2_index - 1] = 2
                    else:
                        bond_order_matrix[atom1_index - 1][atom2_index - 1] = (
                            self.bond_types_to_bond_order_dict[bond_type]
                        )
                        bond_order_matrix[atom2_index - 1][atom1_index - 1] = (
                            self.bond_types_to_bond_order_dict[bond_type]
                        )
                    bond_type_matrix[atom1_index - 1][atom2_index - 1] = bond_type
                    bond_type_matrix[atom2_index - 1][atom1_index - 1] = bond_type
            else:
                connectivity_matrix = None
                bond_order_matrix = None
                bond_type_matrix = None

            molecule_list.append(
                Molecule(
                    Identifier=identifier,
                    NumberOfAtoms=number_of_atoms,
                    NumberOfBonds=number_of_bonds,
                    Atoms=atom_list,
                    AtomsDict={
                        atom.Label: [atom_idx, atom]
                        for atom_idx, atom in enumerate(atom_list)
                    },
                    ConnectivityMatrix=connectivity_matrix,
                    BondOrderMatrix=bond_order_matrix,
                    BondTypeMatrix=bond_type_matrix,
                    NumberOfSubstructures=number_of_substructures,
                )
            )

        self.MoleculeList = molecule_list
        self.MoleculeDict = {
            molecule.Identifier: molecule for molecule in molecule_list
        }

    def ReadMol2Directory(self, mol2_direcotry=str):
        mol2_list = os.listdir(mol2_direcotry)
        main_write = ReadWriteFiles()
        for mol2_file in mol2_list:
            temp_read = ReadWriteFiles()
            temp_read.ReadMol2File(mol2_file=f"{mol2_direcotry}/{mol2_file}")
            molObj = temp_read.MoleculeList[0]
            molObj.Identifier = mol2_file.split(".")[0]
            main_write.MoleculeDict[molObj.Identifier] = molObj
            main_write.MoleculeList.append(molObj)
        main_write.WriteMol2File(output_mol2_file_name=f"{mol2_direcotry}.mol2")

    def WriteMol2File(
        self,
        output_mol2_file_name=str,
    ):
        mol2_string = ""
        for molecule in self.MoleculeList:
            mol2_string = (
                mol2_string
                + f"""@<TRIPOS>MOLECULE
{molecule.Identifier}
{str(molecule.NumberOfAtoms)} {str(molecule.NumberOfBonds)} {str(molecule.NumberOfSubstructures)} 0 0
SMALL
USER_CHARGES
****
Generated With Molecule.py

@<TRIPOS>ATOM
"""
            )
            for idx, atom in enumerate(molecule.Atoms):
                mol2_string = (
                    mol2_string
                    + f"""{str(idx + 1)} {atom.Label} {str(round(atom.Coordinates[0], 7))} {str(round(atom.Coordinates[1], 7))} {str(round(atom.Coordinates[2], 7))} {atom.SybylType} {str(atom.SubstructureIndex)} SUB1 {str(atom.FormalCharge)}
"""
                )
            if type(molecule.ConnectivityMatrix) == np.ndarray:
                mol2_string = mol2_string + "@<TRIPOS>BOND\n"
                bond_idx = 1
                for atom1_idx in range(0, molecule.BondTypeMatrix.shape[0]):
                    for atom2_idx in range(atom1_idx, molecule.BondTypeMatrix.shape[0]):
                        if molecule.ConnectivityMatrix[atom1_idx][atom2_idx] != 0:
                            mol2_string = (
                                mol2_string
                                + f"""{str(bond_idx)} {str(atom1_idx + 1)} {str(atom2_idx + 1)} {molecule.BondTypeMatrix[atom1_idx][atom2_idx]}
"""
                            )
                            bond_idx += 1
        with open(output_mol2_file_name, "w") as f:
            f.write(mol2_string)
            f.close()

    def WriteMol2FileIntoDirectory(self, mol2_directory_name=str):
        try:
            os.mkdir(mol2_directory_name)
        except FileExistsError:
            pass
        for molecule in self.MoleculeList:
            temp_readwrite = ReadWriteFiles()
            temp_readwrite.MoleculeList.append(molecule)
            temp_readwrite.MoleculeDict[molecule.Identifier] = molecule
            temp_readwrite.WriteMol2File(output_mol2_file_name=f"{mol2_directory_name}/{molecule.Identifier}.mol2")

    def ReadSMILEScsv(
        self,
        SMILES_csv_file=str,
    ):
        df = pd.read_csv(SMILES_csv_file)
        for idx in df.index:
            identifier = df.loc[idx, "identifier"]
            smiles_string = df.loc[idx, "SMILES"]
            molecule = Molecule.SMILESToMolecule(
                self, SMILES_string=smiles_string, identifier=identifier
            )
            molecule.NormaliseAtomLabels()
            self.MoleculeList.append(molecule)
            self.MoleculeDict[identifier] = molecule

    def WriteXYZBlock(self, molecule: Molecule):
        xyz_block = ""
        for atom in molecule.Atoms[:-1]:
            xyz_block = (
                xyz_block
                + f"{atom.AtomicSymbol} {atom.Coordinates[0]} {atom.Coordinates[1]} {atom.Coordinates[2]}\n"
            )
        xyz_block = (
            xyz_block
            + f"{molecule.Atoms[-1].AtomicSymbol} {molecule.Atoms[-1].Coordinates[0]} {molecule.Atoms[-1].Coordinates[1]} {molecule.Atoms[-1].Coordinates[2]}"
        )
        return xyz_block

    def Write_ORCA6_TSNEB_Input_For_ExercuteWorkFlow(
        self,
        multiplicity=int,
        method="XTB2",
        basis_set="",
        keywords="Freq",
        PreOpt=True,
        NumImages=8,
        ORCA_exp_path="/users/cmsma/orca/orca_6_0_0_shared_openmpi416/orca",
    ):
        # Produce dictionary containing reactions with their seperate images
        reactions_dict = {}
        for reaction_image in self.MoleculeList:
            reaction_name = reaction_image.Identifier.split("_")[0]
            for string in reaction_image.Identifier.split("_")[1:-1]:
                reaction_name = reaction_name + f"_{string}"
            reactions_dict[reaction_name] = {
                "Reac": None,
                "TSgu": None,
                "Prod": None,
            }
        for reaction_image in self.MoleculeList:
            reaction_name = reaction_image.Identifier.split("_")[0]
            for string in reaction_image.Identifier.split("_")[1:-1]:
                reaction_name = reaction_name + f"_{string}"
            reaction_image_type = reaction_image.Identifier.split("_")[-1]
            reactions_dict[reaction_name][reaction_image_type] = reaction_image

        for reaction_id in reactions_dict:
            reaction_images = reactions_dict[reaction_id]
            reac_image = reaction_images["Reac"]
            reac_id = reac_image.Identifier
            TSgu_image = reaction_images["TSgu"]
            TSgu_id = TSgu_image.Identifier
            prod_image = reaction_images["Prod"]
            prod_id = prod_image.Identifier
            # Write xyz files and get formal charge
            reac_xyz = self.WriteXYZBlock(molecule=reac_image)
            TSgu_xyz = self.WriteXYZBlock(molecule=TSgu_image)
            prod_xyz = self.WriteXYZBlock(molecule=prod_image)
            number_of_atoms = reac_image.NumberOfAtoms
            TSgu_xyz = f"{number_of_atoms}\n\n" + TSgu_xyz
            prod_xyz = f"{number_of_atoms}\n\n" + prod_xyz
            with open(
                self.mol2_file_path + f"{prod_id}.xyz",
                "w",
            ) as f:
                f.write(prod_xyz)
                f.close()
            with open(
                self.mol2_file_path + f"{TSgu_id}.xyz",
                "w",
            ) as f:
                f.write(TSgu_xyz)
                f.close()
            formal_charge = reac_image.GetMoleculeFormalCharge()
            # Write ORCA6 input file
            orca_input = f"""!{method} {basis_set} NEB-TS {keywords}

%NEB
NImages {NumImages}
PREOPT {str(PreOpt).title()}
NEB_END_XYZFILE "{prod_id}.xyz"
NEB_TS_XYZFILE "{TSgu_id}.xyz"
END

*xyz {formal_charge} {multiplicity}
{reac_xyz}
*

""".replace(
                "  ", " "
            )
            with open(
                self.mol2_file_path + f"{reaction_id}.inp",
                "w",
            ) as f:
                f.write(orca_input)
                f.close()

    def WriteORCA6TSNEBInput(
        self,
        multiplicity=int,
        method="XTB2",
        basis_set="",
        keywords="Freq",
        PreOpt=True,
        CPU_num=8,
        RAM_per_CPU=1,
        time="48:00:00",
        NumImages=8,
        NumTSNEBIterations=2000,
        output_directory_name=str,
        ORCA_exp_path="/users/cmsma/orca/orca_6_0_0_shared_openmpi416/orca",
        user_name_path="/users/cmsma/",
    ):
        # Produce dictionary containing reactions with their seperate images
        reactions_dict = {}
        for reaction_image in self.MoleculeList:
            reaction_name = reaction_image.Identifier.split("_")[0]
            for string in reaction_image.Identifier.split("_")[1:-1]:
                reaction_name = reaction_name + f"_{string}"
            reactions_dict[reaction_name] = {
                "Reac": None,
                "TSgu": None,
                "Prod": None,
            }
        for reaction_image in self.MoleculeList:
            reaction_name = reaction_image.Identifier.split("_")[0]
            for string in reaction_image.Identifier.split("_")[1:-1]:
                reaction_name = reaction_name + f"_{string}"
            reaction_image_type = reaction_image.Identifier.split("_")[-1]
            reactions_dict[reaction_name][reaction_image_type] = reaction_image

        # Produce TS-NEB input files per reaction
        try:
            os.makedirs(output_directory_name)
        except FileExistsError:
            pass
        bash_sbatch_string = ""
        for reaction_id in reactions_dict:
            try:
                os.makedirs(f"{output_directory_name}/{reaction_id}")
            except FileExistsError:
                pass
            reaction_images = reactions_dict[reaction_id]
            reac_image = reaction_images["Reac"]
            reac_id = reac_image.Identifier
            TSgu_image = reaction_images["TSgu"]
            TSgu_id = TSgu_image.Identifier
            prod_image = reaction_images["Prod"]
            prod_id = prod_image.Identifier
            # Write xyz files and get formal charge
            reac_xyz = self.WriteXYZBlock(molecule=reac_image)
            TSgu_xyz = self.WriteXYZBlock(molecule=TSgu_image)
            prod_xyz = self.WriteXYZBlock(molecule=prod_image)
            number_of_atoms = reac_image.NumberOfAtoms
            TSgu_xyz = f"{number_of_atoms}\n\n" + TSgu_xyz
            prod_xyz = f"{number_of_atoms}\n\n" + prod_xyz
            with open(
                f"{output_directory_name}/{reaction_id}/{prod_id}.xyz",
                "w",
            ) as f:
                f.write(prod_xyz)
                f.close()
            with open(
                f"{output_directory_name}/{reaction_id}/{TSgu_id}.xyz",
                "w",
            ) as f:
                f.write(TSgu_xyz)
                f.close()
            formal_charge = reac_image.GetMoleculeFormalCharge()
            # Write ORCA6 input file
            if CPU_num == 1:
                orca_input = f"""!{method} {basis_set} NEB-TS {keywords}

%NEB
NImages {NumImages}
PREOPT {str(PreOpt).title()}
NEB_END_XYZFILE "{prod_id}.xyz"
NEB_TS_XYZFILE "{TSgu_id}.xyz"
END

*xyz {formal_charge} {multiplicity}
{reac_xyz}
*

""".replace(
                    "  ", " "
                )
                submission_file = f"""#!/bin/bash
#SBATCH --job-name={reaction_id}
#SBATCH --time={time}
#SBATCH --mem={RAM_per_CPU*CPU_num}G
#SBATCH --ntasks={CPU_num}
#SBATCH --cpus-per-task=1

{ORCA_exp_path} {reaction_id}.inp > {reaction_id}.out

"""
            else:
                orca_input = f"""!{method} {basis_set} NEB-TS {keywords} PAL{CPU_num}

%NEB
NImages {NumImages}
PREOPT {str(PreOpt).title()}
MAXITER {NumTSNEBIterations}
NEB_END_XYZFILE "{prod_id}.xyz"
NEB_TS_XYZFILE "{TSgu_id}.xyz"
END

*xyz {formal_charge} {multiplicity}
{reac_xyz}
*

""".replace(
                    "  ", " "
                )
                output_directory_name_AIRE = output_directory_name.split("/")[-1]
                submission_file = f"""#!/bin/bash
#SBATCH --job-name={reaction_id}
#SBATCH --time={time}
#SBATCH --mem={RAM_per_CPU*CPU_num}G
#SBATCH --ntasks={CPU_num}
#SBATCH --cpus-per-task=1

module load openmpi

# Ensure OpenMPI uses the allocated SLURM resources
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_orte_default_hostfile=$SLURM_JOB_NODELIST

# Create a scratch directory and navigate to it
SCRATCH_DIR=/scratch/$USER/$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# Copy input files to the scratch directory
cp {user_name_path}{output_directory_name_AIRE}/{reaction_id}/{reaction_id}.inp $SCRATCH_DIR
cp {user_name_path}{output_directory_name_AIRE}/{reaction_id}/*.xyz $SCRATCH_DIR

# Run ORCA with MPI
mpirun --bind-to core --map-by slot -np $SLURM_CPUS_PER_TASK {ORCA_exp_path} {reaction_id}.inp > {reaction_id}.out

# Copy results back to permanent storage
cp {reaction_id}.out {user_name_path}{output_directory_name_AIRE}/{reaction_id}/
cp *.xyz {user_name_path}{output_directory_name_AIRE}/{reaction_id}/

# Clean up the scratch directory
rm -rf $SCRATCH_DIR
"""
            if bash_sbatch_string == "":
                bash_sbatch_string = f"""{bash_sbatch_string}cd {reaction_id}
dos2unix {reaction_id}.sh
sbatch {reaction_id}.sh

"""
            else:
                bash_sbatch_string = f"""{bash_sbatch_string}cd ../{reaction_id}
dos2unix {reaction_id}.sh
sbatch {reaction_id}.sh

"""
            with open(
                f"{output_directory_name}/{reaction_id}/{reaction_id}.inp",
                "w",
            ) as f:
                f.write(orca_input)
                f.close()
            # Write .sh submission file
            with open(
                f"{output_directory_name}/{reaction_id}/{reaction_id}.sh",
                "w",
            ) as f:
                f.write(submission_file)
                f.close()
        with open(
            f"{output_directory_name}/bash_sbatch.sh",
            "w",
        ) as f:
            f.write(bash_sbatch_string)
            f.close()

    def ReadORCA6TSNEBOutput(
        self,
        output_directory_name=str,
        NumImages=8,
    ):
        dirs = os.listdir(output_directory_name)
        dirs = [i for i in dirs if len(i.split(".")) == 1]
        new_molecule_set = ReadWriteFiles()
        for dir in dirs:
            new_TS_molObj = deepcopy(self.MoleculeDict[f"{dir}_TSgu"])
            reaction_dir = os.listdir(f"{output_directory_name}/{dir}")
            # Find Converged TS
            # If no converged TS, find MEP saddle point
            if f"{dir}_NEB-TS_converged.xyz" in reaction_dir:
                with open(
                    f"{output_directory_name}/{dir}/{dir}_NEB-TS_converged.xyz", "r"
                ) as f:
                    TS_xyz = f.read()
                    f.close()
                TS_xyz = [
                    [j for j in i.split(" ") if j != ""]
                    for i in TS_xyz.split("\n")[2:]
                    if len([j for j in i.split(" ") if j != ""]) == 4
                ]
                for atomObj, xyz in zip(new_TS_molObj.Atoms, TS_xyz):
                    atomObj.Coordinates = np.array(
                        [
                            float(xyz[1]),
                            float(xyz[2]),
                            float(xyz[3]),
                        ]
                    )
            elif f"{dir}_MEP_trj.xyz" in reaction_dir:
                with open(f"{output_directory_name}/{dir}/{dir}_MEP_trj.xyz", "r") as f:
                    TS_MEP_xyz = f.read()
                    f.close()
                TS_MEP_xyz = TS_MEP_xyz.split(f"Coordinates from ORCA-job {dir}_MEP E")[
                    1:
                ]
                TS_MEP_xyz = [
                    [[j for j in i.split(" ") if j != ""] for i in xyz.split("\n")[:-2]]
                    for xyz in TS_MEP_xyz
                ]
                en_xyz_list = [[float(i[0][0]), i[1:]] for i in TS_MEP_xyz]
                en_xyz_list = sorted(en_xyz_list, key=lambda x: x[0], reverse=True)
                for atomObj, xyz in zip(new_TS_molObj.Atoms, en_xyz_list[0][1]):
                    atomObj.Coordinates = np.array(
                        [
                            float(xyz[1]),
                            float(xyz[2]),
                            float(xyz[3]),
                        ]
                    )
            # Find Optimised Product
            new_Prod_molObj = deepcopy(self.MoleculeDict[f"{dir}_Prod"])
            with open(f"{output_directory_name}/{dir}/{dir}_product.xyz", "r") as f:
                Prod_xyz = f.read()
                f.close()
            Prod_xyz = [
                [j for j in i.split(" ") if j != ""]
                for i in Prod_xyz.split("\n")[2:]
                if len([j for j in i.split(" ") if j != ""]) == 4
            ]
            for atomObj, xyz in zip(new_Prod_molObj.Atoms, Prod_xyz):
                atomObj.Coordinates = np.array(
                    [
                        float(xyz[1]),
                        float(xyz[2]),
                        float(xyz[3]),
                    ]
                )
            # Find Optimised Reactant
            new_Reac_molObj = deepcopy(self.MoleculeDict[f"{dir}_Reac"])
            with open(f"{output_directory_name}/{dir}/{dir}_reactant.xyz", "r") as f:
                Reac_xyz = f.read()
                f.close()
            Reac_xyz = [
                [j for j in i.split(" ") if j != ""]
                for i in Reac_xyz.split("\n")[2:]
                if len([j for j in i.split(" ") if j != ""]) == 4
            ]
            for atomObj, xyz in zip(new_Reac_molObj.Atoms, Reac_xyz):
                atomObj.Coordinates = np.array(
                    [
                        float(xyz[1]),
                        float(xyz[2]),
                        float(xyz[3]),
                    ]
                )
            # Add molecules to molecule set
            new_molecule_set.MoleculeDict[new_Reac_molObj.Identifier] = new_Reac_molObj
            new_molecule_set.MoleculeList.append(new_Reac_molObj)
            new_molecule_set.MoleculeDict[new_TS_molObj.Identifier] = new_TS_molObj
            new_molecule_set.MoleculeList.append(new_TS_molObj)
            new_molecule_set.MoleculeDict[new_Prod_molObj.Identifier] = new_Prod_molObj
            new_molecule_set.MoleculeList.append(new_Prod_molObj)
        new_molecule_set.WriteMol2File(
            output_mol2_file_name=f"{output_directory_name}_ORCAOutput.mol2"
        )
        # Write individual Product, TS and Reactant MOL2 files
        reac_molecule_set = ReadWriteFiles()
        TSgu_molecule_set = ReadWriteFiles()
        prod_molecule_set = ReadWriteFiles()
        for molecule in new_molecule_set.MoleculeList:
            if molecule.Identifier.split("_")[-1] == "Reac":
                new_id = molecule.Identifier.split("_")[0]
                for string in molecule.Identifier.split("_")[1:-1]:
                    new_id = f"{new_id}_{string}"
                molecule.Identifier = new_id
                reac_molecule_set.MoleculeDict[molecule.Identifier] = molecule
                reac_molecule_set.MoleculeList.append(molecule)
            elif molecule.Identifier.split("_")[-1] == "TSgu":
                new_id = molecule.Identifier.split("_")[0]
                for string in molecule.Identifier.split("_")[1:-1]:
                    new_id = f"{new_id}_{string}"
                molecule.Identifier = new_id
                TSgu_molecule_set.MoleculeDict[molecule.Identifier] = molecule
                TSgu_molecule_set.MoleculeList.append(molecule)
            elif molecule.Identifier.split("_")[-1] == "Prod":
                new_id = molecule.Identifier.split("_")[0]
                for string in molecule.Identifier.split("_")[1:-1]:
                    new_id = f"{new_id}_{string}"
                molecule.Identifier = new_id
                prod_molecule_set.MoleculeDict[molecule.Identifier] = molecule
                prod_molecule_set.MoleculeList.append(molecule)
        reac_molecule_set.WriteMol2File(
            output_mol2_file_name=f"{output_directory_name}/Reactants.mol2"
        )
        prod_molecule_set.WriteMol2File(
            output_mol2_file_name=f"{output_directory_name}/Products.mol2"
        )
        TSgu_molecule_set.WriteMol2File(
            output_mol2_file_name=f"{output_directory_name}/TSguess.mol2"
        )

    def WriteORCA6Input(
        self,
        multiplicity=int,
        method=str,
        basis_set=str,
        output_dir_name=str,
        keyword="",
        CPU_num=int,
        RAM_per_CPU=int,
        ORCA_exe_dir="/users/cmsma/orca/orca_6_0_0_shared_openmpi416/orca",
        user_name_path="/users/cmsma/",
        time="48:00:00",
        resource_req_csv=None,
        scratch_space=10,
    ):
        try:
            os.mkdir(output_dir_name)
        except FileExistsError:
            pass
        sbatch_string = ""
        if resource_req_csv != None:
            df = pd.read_csv(resource_req_csv)
            df.set_index("Identifier", inplace=True)
        else:
            pass
        for molecule in self.MoleculeList:
            xyz_block = self.WriteXYZBlock(molecule=molecule)
            if resource_req_csv != None:
                CPU_time = df.loc[molecule.Identifier, "Total CPU time Est Req / sec"]
                wall_time = CPU_time / CPU_num
                hours = int(wall_time / (60 * 60))
                minuets = int(
                    ((wall_time / (60 * 60)) - int(wall_time / (60 * 60))) * 60
                )
                seconds = int(
                    (
                        (((wall_time / (60 * 60)) - int(wall_time / (60 * 60))) * 60)
                        - int(
                            ((wall_time / (60 * 60)) - int(wall_time / (60 * 60))) * 60
                        )
                    )
                    * 60
                )
                if len(str(hours)) == 1:
                    hours = f"0{hours}"
                if len(str(minuets)) == 1:
                    minuets = f"0{minuets}"
                if len(str(seconds)) == 1:
                    seconds = f"0{seconds}"
                time = f"{hours}:{minuets}:{seconds}"
                RAM_total = df.loc[molecule.Identifier, "Total RAM Est Req / GB"]
                RAM_per_CPU = RAM_total / CPU_num
            if CPU_num == 1:
                orca_input_string = f"""! {method} {basis_set} {keyword}

*xyz {molecule.GetMoleculeFormalCharge()} {multiplicity}
{xyz_block}
*
"""
            elif CPU_num > 1:
                if (
                    keyword == "GOAT"
                    or keyword == "GOAT-EXPLORE"
                    or keyword == "GOAT-REACT"
                ):
                    orca_input_string = f"""! {method} {basis_set} {keyword} PAL{CPU_num}

%maxcore {int(RAM_per_CPU*750)}

%goat
    nworkers {CPU_num}
end

*xyz {molecule.GetMoleculeFormalCharge()} {multiplicity}
{xyz_block}
*
"""
                else:
                    orca_input_string = f"""! {method} {basis_set} {keyword}

%maxcore {int(RAM_per_CPU*750)}

%pal
    nprocs {CPU_num}
end

*xyz {molecule.GetMoleculeFormalCharge()} {multiplicity}
{xyz_block}
*
"""
            if CPU_num == 1:
                submission_file_string = f"""#!/bin/bash
#SBATCH --job-name={molecule.Identifier}
#SBATCH --time={time}
#SBATCH --mem={int(RAM_per_CPU*CPU_num)}G
#SBATCH --ntasks={CPU_num}
#SBATCH --cpus-per-task=1

{ORCA_exe_dir} {molecule.Identifier}.inp > {molecule.Identifier}.out

"""
            elif CPU_num * RAM_per_CPU <= 700:
                submission_file_string = f"""#!/bin/bash
#SBATCH --job-name={molecule.Identifier}
#SBATCH --time={time}
#SBATCH --mem={int(RAM_per_CPU*CPU_num)}G
#SBATCH --ntasks={CPU_num}
#SBATCH --cpus-per-task=1

module load openmpi

# Ensure OpenMPI uses the allocated SLURM resources
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_orte_default_hostfile=$SLURM_JOB_NODELIST

# Create a scratch directory and navigate to it
SCRATCH_DIR=/scratch/$USER/$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# Copy input files to the scratch directory
cp {user_name_path}{output_dir_name.split("/")[-1]}/{molecule.Identifier}.inp $SCRATCH_DIR

# Run ORCA with MPI
mpirun --bind-to core --map-by slot -np $SLURM_CPUS_PER_TASK {ORCA_exe_dir} {molecule.Identifier}.inp > {molecule.Identifier}.out

# Copy results back to permanent storage
cp {molecule.Identifier}.out {user_name_path}{output_dir_name.split("/")[-1]}/
cp *.xyz {user_name_path}{output_dir_name.split("/")[-1]}/

# Clean up the scratch directory
rm -rf $SCRATCH_DIR
"""
            elif CPU_num * RAM_per_CPU > 700:
                submission_file_string = f"""#!/bin/bash
#SBATCH --job-name={molecule.Identifier}
#SBATCH --time={time}
#SBATCH --partition=himem
#SBATCH --mem={RAM_per_CPU*CPU_num}G
#SBATCH --nodes=1
#SBATCH --ntasks={CPU_num}
#SBATCH --cpus-per-task=1

module load openmpi

# Ensure OpenMPI uses the allocated SLURM resources
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_orte_default_hostfile=$SLURM_JOB_NODELIST

# Create a scratch directory and navigate to it
SCRATCH_DIR=/scratch/$USER/$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# Copy input files to the scratch directory
cp {user_name_path}{output_dir_name.split("/")[-1]}/{molecule.Identifier}.inp $SCRATCH_DIR

# Run ORCA with MPI
mpirun --bind-to core --map-by slot -np $SLURM_CPUS_PER_TASK {ORCA_exe_dir} {molecule.Identifier}.inp > {molecule.Identifier}.out

# Copy results back to permanent storage
cp {molecule.Identifier}.out {user_name_path}{output_dir_name.split("/")[-1]}/
cp *.xyz {user_name_path}{output_dir_name.split("/")[-1]}/

# Clean up the scratch directory
rm -rf $SCRATCH_DIR
"""
            with open(f"{output_dir_name}/{molecule.Identifier}.inp", "w") as f:
                f.write(orca_input_string)
                f.close()
            with open(f"{output_dir_name}/{molecule.Identifier}.sh", "w") as f:
                f.write(submission_file_string)
                f.close()
            sbatch_string = sbatch_string + f"sbatch {molecule.Identifier}.sh\n"
        with open(f"{output_dir_name}/bash_sbatch.sh", "w") as f:
            f.write(sbatch_string)
            f.close()

    def WriteCRESTInput(
        self,
        multiplicity=int,
        method="gfn2",
        keyword=str,
        CPU_num=4,
        RAM_per_CPU=1,
        output_dir_name=str,
        user_name_path="/users/cmsma/",
        crest_exe_path="crest/",
        time="48:00:00",
        resource_req_csv=None,
        scratch_space=10,
    ):
        try:
            os.mkdir(output_dir_name)
        except FileExistsError:
            pass
        sbatch_string = ""
        for molecule in self.MoleculeList:
            xyz_block = self.WriteXYZBlock(molecule=molecule)
            xyz_input_string = f"""{len(molecule.Atoms)}

{xyz_block}"""
            submission_file_string = f"""#!/bin/bash
#SBATCH --job-name={molecule.Identifier}
#SBATCH --time={time}
#SBATCH --mem={int(RAM_per_CPU*CPU_num)}G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPU_num}

chmod +x {user_name_path}{crest_exe_path}crest
export PATH=$PATH:{user_name_path}{crest_exe_path}

export OMP_NUM_THREADS={CPU_num}

# Create a scratch directory and navigate to it
SCRATCH_DIR=/scratch/$USER/$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# Copy input files to the scratch directory
cp {user_name_path}{output_dir_name.split("/")[-1]}/{molecule.Identifier}/{molecule.Identifier}.xyz $SCRATCH_DIR

crest {molecule.Identifier}.xyz > {molecule.Identifier}.out -T {CPU_num} --{method} --chrg {molecule.GetMoleculeFormalCharge()}"""
            if multiplicity != 1:
                submission_file_string = (
                    f"{submission_file_string} --uhf {(multiplicity-1)/2}"
                )
            submission_file_string = f"""{submission_file_string}

# Copy results back to permanent storage
cp {molecule.Identifier}.out {user_name_path}{output_dir_name.split("/")[-1]}/{molecule.Identifier}/
cp *.xyz {user_name_path}{output_dir_name.split("/")[-1]}/{molecule.Identifier}/

# Clean up the scratch directory
rm -rf $SCRATCH_DIR
"""
            try:
                os.mkdir(f"{output_dir_name}/{molecule.Identifier}")
            except FileExistsError:
                pass
            with open(
                f"{output_dir_name}/{molecule.Identifier}/{molecule.Identifier}.xyz",
                "w",
            ) as f:
                f.write(xyz_input_string)
                f.close()
            with open(
                f"{output_dir_name}/{molecule.Identifier}/{molecule.Identifier}.sh", "w"
            ) as f:
                f.write(submission_file_string)
                f.close()
            if sbatch_string == "":
                sbatch_string = (
                    sbatch_string
                    + f"""cd {molecule.Identifier}
dos2unix {molecule.Identifier}.sh
sbatch {molecule.Identifier}.sh

"""
                )
            else:
                sbatch_string = (
                    sbatch_string
                    + f"""cd ../{molecule.Identifier}
dos2unix {molecule.Identifier}.sh
sbatch {molecule.Identifier}.sh

"""
                )
        with open(f"{output_dir_name}/bash_sbatch.sh", "w") as f:
            f.write(sbatch_string)
            f.close()

    def ReadCRESTOutputDirectory(
        self,
        output_dir_name=str,
        results_files_name=str,
        SaveOnlyCRESTBEST=True,
    ):
        list_dir = os.listdir(output_dir_name)
        list_dir = [i for i in list_dir if len(i.split(".")) == 1]
        results_df = pd.DataFrame()
        new_molecule_set = ReadWriteFiles()
        for dir in list_dir:
            Identifier = dir
            dir_contents = os.listdir(f"{output_dir_name}/{dir}")
            new_molecule = deepcopy(self.MoleculeDict[dir])
            try:
                with open(
                    f"{output_dir_name}/{dir}/{dir}.out", "r", encoding="utf-8"
                ) as f:
                    out_file = f.read()
                    f.close()
                results_df.loc[Identifier, "Successful Run?"] = ".out file produced"
            except FileNotFoundError:
                results_df.loc[Identifier, "Successful Run?"] = "No .out file"
                continue
            if SaveOnlyCRESTBEST == True:
                with open(f"{output_dir_name}/{dir}/crest_best.xyz", "r") as f:
                    crest_best = f.read()
                    f.close()
            elif SaveOnlyCRESTBEST == False:
                with open(f"{output_dir_name}/{dir}/crest_conformers.xyz", "r") as f:
                    crest_conformers = f.read()
                    f.close()
                numberOfAtoms = new_molecule.NumberOfAtoms
                crest_conformers = [
                    i for i in crest_conformers.split(f"\n  {numberOfAtoms}\n")
                ]
                crest_conformers = [i.split("\n") for i in crest_conformers if i != ""]
                crest_conformers = [
                    [[k for k in j.split(" ") if k != ""] for j in i]
                    for i in crest_conformers
                ]
                crest_conformers = [
                    [j for j in i if len(j) == 4] for i in crest_conformers
                ]
                for idx, crest_conformer_xyz in enumerate(crest_conformers):
                    new_molecule = deepcopy(self.MoleculeDict[dir])
                    new_identifier = f"{dir}_CREST{idx}"
                    new_molecule.Identifier = new_identifier
                    for crest_xyz, atomObj in zip(
                        crest_conformer_xyz, new_molecule.Atoms
                    ):
                        atomObj.Coordinates = np.array(
                            [
                                float(crest_xyz[1]),
                                float(crest_xyz[2]),
                                float(crest_xyz[3]),
                            ]
                        )
                    new_molecule_set.MoleculeDict[Identifier] = new_molecule
                    new_molecule_set.MoleculeList.append(new_molecule)
                continue
            crest_best_xyz = [
                [j for j in i.split(" ") if j != ""] for i in crest_best.split("\n")
            ]
            crest_best_xyz = [i for i in crest_best_xyz if len(i) == 4]
            for crest_xyz, atomObj in zip(crest_best_xyz, new_molecule.Atoms):
                atomObj.Coordinates = np.array(
                    [
                        float(crest_xyz[1]),
                        float(crest_xyz[2]),
                        float(crest_xyz[3]),
                    ]
                )
            new_molecule_set.MoleculeDict[Identifier] = new_molecule
            new_molecule_set.MoleculeList.append(new_molecule)
        new_molecule_set.WriteMol2File(
            output_mol2_file_name=f"{results_files_name}_CRESTOutput.mol2"
        )

    def ReadORCA6OuputDirectory(
        self,
        output_dir_name=str,
        results_files_name=str,
    ):
        list_dir = os.listdir(output_dir_name)
        out_files = [i for i in list_dir if i.split(".")[-1] == "out"]
        out_files = [i for i in out_files if i.split("-")[0] != "slurm"]
        results_df = pd.DataFrame()
        for out_file in out_files:
            identifier = out_file.split(".")[0]
            with open(f"{output_dir_name}/{out_file}", "r") as f:
                out_file = f.read()
                f.close()
            molObj = self.MoleculeDict[identifier]
            # Find Single Point Energy
            try:
                sp_en = out_file.split("FINAL SINGLE POINT ENERGY")[-1]
                sp_en = float(
                    [i for i in sp_en.split("\n")[0].split(" ") if i != ""][0]
                )
                results_df.loc[identifier, "Single Point Energy / HF"] = sp_en
            except IndexError:
                results_df.loc[identifier, "Single Point Energy / HF"] = np.nan
            except ValueError:
                results_df.loc[identifier, "Single Point Energy / HF"] = np.nan
            # Find Freq energies
            # Find CPU time
            try:
                CPU_num = out_file.split("Program running with")
                if len(CPU_num) > 1:
                    CPU_num = int(CPU_num[-1].split("parallel MPI-processes")[0])
                else:
                    CPU_num = 1
                time = out_file.split("TOTAL RUN TIME:")[-1]
                time = [i for i in time.split(" ") if i != ""]
                time = (
                    (int(time[0]) * 24 * 60 * 60)
                    + (int(time[2]) * 60 * 60)
                    + (int(time[4]) * 60)
                    + (int(time[6]))
                    + (float(time[8]) / 1000)
                )
                CPU_time = time * CPU_num
                results_df.loc[identifier, "CPU time / sec"] = CPU_time
            except ValueError:
                results_df.loc[identifier, "CPU time / sec"] = np.nan
            except IndexError:
                results_df.loc[identifier, "CPU time / sec"] = np.nan
            # Find Memory usage
            try:
                max_mem_list = [
                    i.split("MB")[0] for i in out_file.split("Maximum memory")[1:]
                ]
                mem_list = []
                for i in max_mem_list:
                    if len(i.split(":")) == 2:
                        mem = float(i.split(":")[-1])
                        mem_list.append(mem)
                results_df.loc[identifier, "Maximum RAM mem per CPU / MB"] = max(
                    mem_list
                )
            except ValueError:
                results_df.loc[identifier, "Maximum RAM mem per CPU / MB"] = np.nan
            # Find molecular weight and update coordinates
            try:
                xyz_block = [
                    [j for j in i.split(" ") if j != ""]
                    for i in out_file.split("CARTESIAN COORDINATES (ANGSTROEM)")[-1]
                    .split("\n\n")[0]
                    .split("\n")[2:]
                ]
                for line, atomObj in zip(xyz_block, molObj.Atoms):
                    atomic_symbol = line[0]
                    x = float(line[1])
                    y = float(line[2])
                    z = float(line[3])
                    atomObj.Coordinates = np.array([x, y, z])
                weight = molObj.GetMoleculeWeight()
                results_df.loc[identifier, "Molecular Weight / g mol-1"] = weight
            except:
                results_df.loc[identifier, "Molecular Weight / g mol-1"] = np.nan
            # If PES calculation, find and plot PES
            if len(out_file.split("RELAXED SURFACE SCAN RESULTS")) == 2:
                relaxed_PES = out_file.split("RELAXED SURFACE SCAN RESULTS")[-1]
                relaxed_PES_actual_en = relaxed_PES.split(
                    "The Calculated Surface using the 'Actual Energy'"
                )[-1].split("\n\n")[0]
                relaxed_PES_actual_en = [
                    [j for j in i.split(" ") if j != ""]
                    for i in relaxed_PES_actual_en.split("\n")
                ]
                relaxed_PES_actual_en = [
                    i for i in relaxed_PES_actual_en if len(i) == 2
                ]
                PES_df = pd.DataFrame()
                for idx, data_point in enumerate(relaxed_PES_actual_en):
                    PES_df.loc[idx, "Geom Measurment"] = data_point[0]
                    PES_df.loc[idx, "Actual Energy"] = data_point[1]
                PES_df.to_csv(f"{output_dir_name}/{identifier}_PES.csv")

            # Find Number of Basis sets used
            try:
                basis_set_num = int(
                    out_file.split("Number of basis functions")[-1]
                    .split("\n")[0]
                    .split("...")[1]
                )
                results_df.loc[identifier, "Number of Basis Sets Used"] = basis_set_num
            except IndexError:
                results_df.loc[identifier, "Number of Basis Sets Used"] = np.nan
        results_df.reset_index(inplace=True)
        results_df.rename(columns={"index": "Identifier"}, inplace=True)
        results_df.set_index("Identifier", inplace=True)
        results_df.to_csv(f"{results_files_name}_ORCA6Output.csv")
        self.WriteMol2File(
            output_mol2_file_name=f"{results_files_name}_ORCA6Output.mol2"
        )

    def ReadG16OutputDirectory(
        self,
        output_dir_name=str,
        results_csv_name=str,
    ):
        list_dir = os.listdir(output_dir_name)
        log_files = [i for i in list_dir if i.split(".")[-1] == "log"]
        results_df = pd.DataFrame()
        new_molecules = ReadWriteFiles()
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "mol2")
        for log_file in log_files:
            Identifier = log_file.split(".")[0]
            with open(f"{output_dir_name}/{log_file}", "r") as f:
                log_file = f.read()
                f.close()
            results = log_file.split(
                "----------------------------------------------------------------------"
            )[-1]
            results = results.replace("\n", "")
            results = results.replace(" ", "")
            results = results.split("\\")
            results = [i.split(",") for i in results]
            # Find Energies
            for item in results:
                if len(item[0].split("=")) == 2:
                    if item[0].split("=")[0] == "HF":
                        results_df.loc[Identifier, "Single Point Energy / HF"] = item[
                            0
                        ].split("=")[1]
                    elif item[0].split("=")[0] == "ETot":
                        results_df.loc[
                            Identifier, "Electronic, ZPE and Thermal / HF"
                        ] = item[0].split("=")[1]
                    elif item[0].split("=")[0] == "HTot":
                        results_df.loc[Identifier, "Enthalpy Energy / HF"] = item[
                            0
                        ].split("=")[1]
                    elif item[0].split("=")[0] == "GTot":
                        results_df.loc[Identifier, "Gibbs Free Energy / HF"] = item[
                            0
                        ].split("=")[1]
            # Find 6 lowest Energies
            freqencies = log_file.split("Frequencies --")[1:]
            freqencies = [i.split("\n")[0] for i in freqencies]
            freqencies = [[j for j in i.split(" ") if j != ""] for i in freqencies]
            freq_new = []
            for freq_triple in freqencies:
                freq_new = freq_new + freq_triple
            freqencies = freq_new
            for i in range(0, 6):
                results_df.loc[Identifier, f"Vib{i} / cm-1"] = freqencies[i]
            # Find Coordinates
            coordinates = [i for i in results if len(i) == 4]
            # Update coordinates
            try:
                molecule = self.MoleculeDict[Identifier]
            except KeyError:
                # If can not update coordinates
                # Make new .mol2 file with open babel
                atomic_symbol = coordinates[0][0]
                x = float(coordinates[0][1])
                y = float(coordinates[0][2])
                z = float(coordinates[0][3])
                atom = Atom(
                    Label=f"{atomic_symbol}1",
                    Coordinates=np.array([x, y, z]),
                    SybylType=atomic_symbol,
                    AtomicSymbol=atomic_symbol,
                    FormalCharge=0,
                )
                molecule = Molecule(
                    Atoms=[atom],
                    AtomsDict={atom.Label: [0, atom]},
                    NumberOfAtoms=1,
                    NumberOfBonds=0,
                    ConnectivityMatrix=None,
                    BondOrderMatrix=None,
                    BondTypeMatrix=None,
                    NumberOfSubstructures=1,
                    Identifier=Identifier,
                )
                for idx, coordinate in enumerate(coordinates[1:]):
                    idx += 2
                    atomic_symbol = coordinate[0]
                    x = float(coordinate[1])
                    y = float(coordinate[2])
                    z = float(coordinate[3])
                    molecule.AddAtom(
                        AtomicSymbol=atomic_symbol,
                        SybylType=atomic_symbol,
                        FormalCharge=0,
                        Coordinates=np.array([x, y, z]),
                        Label=f"{atomic_symbol}{idx}",
                        Adjust_Bond_Matrices=False,
                    )
                xyz_string = f"""{molecule.NumberOfAtoms}

{self.WriteXYZBlock(molecule=molecule)}"""
                with open(f"{output_dir_name}_TEMP.xyz", "w") as f:
                    f.write(xyz_string)
                    f.close()
                # Read XYZ file
                mol = ob.OBMol()
                obConversion.ReadFile(mol, f"{output_dir_name}_TEMP.xyz")
                # Set formal charges for atoms
                for atom in ob.OBMolAtomIter(mol):
                    atom.SetPartialCharge(
                        float(atom.GetFormalCharge())
                    )  # Replace partial with formal charge
                # Write to MOL2 file
                obConversion.WriteFile(mol, f"{output_dir_name}_TEMP.mol2")
                # ReRead MOL2 file in Molecule() and Atom() format
                molecule = ReadWriteFiles()
                molecule.ReadMol2File(mol2_file=f"{output_dir_name}_TEMP.mol2")
                molecule = molecule.MoleculeList[0]
                molecule.Identifier = Identifier
                molecule.NormaliseAtomLabels()
                new_molecules.MoleculeDict[Identifier] = molecule
                new_molecules.MoleculeList.append(molecule)
        del obConversion
        del mol
        os.remove(f"{output_dir_name}_TEMP.xyz")
        os.remove(f"{output_dir_name}_TEMP.mol2")
        new_molecules.WriteMol2File(
            output_mol2_file_name=f"{output_dir_name}_g16Output.mol2",
        )
        results_df.to_csv(results_csv_name)

    def GetMoleculeWeights(
        self,
        results_csv_name=str,
    ):
        weight_df = pd.DataFrame()
        for molObj in self.MoleculeList:
            weight_df.loc[molObj.Identifier, "Weight / g mol-1"] = (
                molObj.GetMoleculeWeight()
            )
        weight_df.to_csv(results_csv_name)
        pass

    def SaveMoleculeBasedOnWeight(
        self,
        results_csv_name=str,
        output_mol2_file_name=str,
        min_weight=float,
        max_weight=float,
    ):
        weight_df = pd.DataFrame()
        new_mol2_file = ReadWriteFiles()
        for molObj in self.MoleculeList:
            if molObj.GetMoleculeWeight() >= min_weight:
                if molObj.GetMoleculeWeight() <= max_weight:
                    weight_df.loc[molObj.Identifier, "Weight / g mol-1"] = (
                        molObj.GetMoleculeWeight()
                    )
                    new_mol2_file.MoleculeDict[molObj.Identifier] = molObj
                    new_mol2_file.MoleculeList.append(molObj)
        weight_df.to_csv(results_csv_name)
        new_mol2_file.WriteMol2File(output_mol2_file_name=output_mol2_file_name)

    def SaveMoleculeBasedOnCharge(
        self,
        output_mol2_file_name=str,
        output_csv_file=str,
        charge=int,
    ):
        new_mol2_file = ReadWriteFiles()
        df = pd.DataFrame()
        for molObj in self.MoleculeList:
            if molObj.GetMoleculeFormalCharge() == charge:
                new_mol2_file.MoleculeDict[molObj.Identifier] = molObj
                new_mol2_file.MoleculeList.append(molObj)
                df.loc[molObj.Identifier, "Charge"] = molObj.GetMoleculeFormalCharge()
        new_mol2_file.WriteMol2File(output_mol2_file_name=output_mol2_file_name)
        df.to_csv(output_csv_file)

    def GetMoleculeChargeCSV(
        self,
        output_results_csv=str,
    ):
        df = pd.DataFrame()
        for molObj in self.MoleculeList:
            df.loc[molObj.Identifier, "Charge"] = molObj.GetMoleculeFormalCharge()
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Identifier"}, inplace=True)
        df.set_index("Identifier", inplace=True)
        df.to_csv(output_results_csv)

    def GetCompResourceEstimation(
        self, results_csv_name=str, basis_set=str, method_type=str
    ):
        df = pd.DataFrame()
        for molecule in self.MoleculeList:
            size = molecule.GetTotalMoleculeBasisSetSize(basis_set=basis_set)
            df.loc[molecule.Identifier, f"{basis_set} size"] = size
            if method_type == "DLPNO-CCSD(T)_SP":
                mem = (size * 407) + 102210
                df.loc[molecule.Identifier, "Total RAM Est Req / GB"] = int(mem * 0.001)
                cpu_time = (size * 142) - 15360
                if cpu_time < 14400:
                    cpu_time = 14400
                df.loc[molecule.Identifier, "Total CPU time Est Req / sec"] = cpu_time
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Identifier"}, inplace=True)
        df.set_index("Identifier", inplace=True)
        df.to_csv(results_csv_name)

    def CreateNewMol2FileBasedOnCSV(
        self,
        input_csv_file=str,
        output_mol2_file_name=str,
    ):
        NewMol2File = ReadWriteFiles()
        df = pd.read_csv(input_csv_file)
        print(self.MoleculeDict)
        for identifier in df["Identifier"]:
            print(identifier)
            try:
                NewMol2File.MoleculeList.append(self.MoleculeDict[identifier + "_0"])
                NewMol2File.MoleculeDict[identifier] = self.MoleculeDict[
                    identifier + "_0"
                ]
            except KeyError:
                print(f"Could not find {identifier}")
        NewMol2File.WriteMol2File(
            output_mol2_file_name=output_mol2_file_name,
        )

    def MeasureBondAnglesAroundAtomCentre(
        self,
        AtomicSymbol="Li",
        results_csv_file_name=str,
    ):
        df = pd.DataFrame()
        Index = 0
        for molecule in self.MoleculeList:
            Identifier = molecule.Identifier
            for atom in molecule.Atoms:
                if atom.AtomicSymbol == AtomicSymbol:
                    natomsLabels = molecule.GetAtomNeighbours(AtomLabel=atom.Label)
                    coordination_number = len(natomsLabels)
                    for idx, natomLabel1 in enumerate(natomsLabels):
                        for natomLabel2 in natomsLabels[idx + 1 :]:
                            theta = round(
                                np.degrees(
                                    molecule.FindBondAngle(
                                        a=molecule.AtomsDict[natomLabel1][
                                            1
                                        ].Coordinates,
                                        b=atom.Coordinates,
                                        c=molecule.AtomsDict[natomLabel2][
                                            1
                                        ].Coordinates,
                                    )
                                ),
                                2,
                            )
                            df.loc[Index, "Identifier"] = Identifier
                            df.loc[Index, "Coordination Number"] = coordination_number
                            df.loc[Index, "AtomLabel1-LiLabel-AtomLabel2"] = (
                                f"{natomLabel1}-{atom.Label}-{natomLabel2}"
                            )
                            df.loc[Index, "Theta / deg"] = theta
                            Index += 1
        df.to_csv(results_csv_file_name)
        pass


if __name__ == "__main__":

    filepath = "/users/cmsma/MolBuild"
    readwrite = ReadWriteFiles()
    readwrite.ReadMol2File(
        mol2_file=f"{filepath}/HALCROW_Lig.mol2"
    )
    readwrite.WriteMol2FileIntoDirectory(
        mol2_directory_name=f"{filepath}/HALCROW_Lig"
    )

    """
    MB = MoleculeBuilder()
    MB.BuildMetalComplex(
        MetalCenter="Fe",
        MetalOxidation=2,
        FragmentDict={
            "Fragment1": {
                "SMILES": {
                    "NNN": "O=[N+]([O-])c1ccc(C=Cc2cc(-n3cccn3)nc(-n3cccn3)c2)cc1",
                },
                "BindingInformation": [
                    [np.array([0, 0, 2]), "N.ar"],
                    [np.array([-2, 0, 0]), "N.ar"],
                    [np.array([0, 0, -2]), "N.ar"],
                ],
                "StereoChemicalInformation": None,
            },
            "Fragment2": {
                "SMILES": {
                    "NNN": "O=[N+]([O-])c1ccc(C=Cc2cc(-n3cccn3)nc(-n3cccn3)c2)cc1",
                },
                "BindingInformation": [
                    [np.array([0, 2, 0]), "N.ar"],
                    [np.array([2, 0, 0]), "N.ar"],
                    [np.array([0, -2, 0]), "N.ar"],
                ],
                "StereoChemicalInformation": None,
            },
        },
        Output_mol2_file=f"HALCROW-X-cisCHCHC6H4NO24.mol2",
        GFN2xTB_Opt_FinalStructure=False,
    )
    """
