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

from MoleculeHandler import Molecule, ReadWriteFiles

if __name__ == "__main__":

    readwrite = ReadWriteFiles()
    readwrite.ReadMol2File(
        mol2_file="C://Users//cm21sb//OneDrive - University of Leeds//Year 4//Sophie Blanch//code//tetracyclic_new//sandwiches//combined_ligands.mol2"
    )

    for molecule in readwrite.MoleculeList:
        molecule2 = deepcopy(molecule)
        molecule.RotateMolecule(rotation_axis=np.array([0, 1, 0]), theta=np.pi)
        molecule2.TranslateMolecule(TranslationVector=np.array([0, 0, 6]))
        molecule.AddAtom(Label="Li1", AtomicSymbol="Li", SybylType="Li", Coordinates=np.array([0, 0, -3]), FormalCharge=1)
        molecule.AddMolecule(Molecule=molecule2)

    readwrite.WriteMol2File(output_mol2_file_name="combined_sandwiches_v2.mol2")