#Necessary imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions as Reactions
import sys,re,random
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
import itertools
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import inchi
import pandas as pd
import zipfile
import os

# Define stepwise reaction function that does the following for each list in combinations: 
# amino acid 1 + amino acid 2 = intermediate1
# intermediate1 + amino acid 3 = intermediate2
# intermediate2 + amino acid 4 = product (linear peptide)

def stepwise_reaction(amino_acids_list):
    intermediate1 = amide_bond_reaction.RunReactants([amino_acids_list[0], amino_acids_list[1]])
    intermediate1 =intermediate1[0][0]
    Chem.SanitizeMol(intermediate1)
    intermediate2 = amide_bond_reaction.RunReactants([intermediate1, amino_acids_list[2]])
    intermediate2 = intermediate2[0][0]
    Chem.SanitizeMol(intermediate2)
    product = amide_bond_reaction.RunReactants([intermediate2, amino_acids_list[3]])
    product = product[0][0]
    Chem.SanitizeMol(product)
    return product

# Produce xyz co-ordinate files

def smiles_to_xyz(smiles_string, output_xyz_file):
    # Convert SMILES to an rdkit molecule
    mol = Chem.MolFromSmiles(smiles_string)
    
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    
    # Add hydrogens, rdkit uff optimise, and generate 3D coordinates
    mol = Chem.AddHs(mol)  # Add hydrogens to the molecule
    AllChem.EmbedMolecule(mol)  # Generate 3D coordinates
    AllChem.UFFOptimizeMolecule(mol)  # Optimize the molecule's geometry using UFF
    
    # Extract the atomic positions and atomic symbols
    conf = mol.GetConformer()  # Get the conformation with 3D coordinates
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # Get atom symbols
    coords = conf.GetPositions()  # Get atom coordinates (in angstroms)
    
    # Write XYZ file
    with open(f"{output_xyz_file}.inp", 'w') as xyz_file:
        charge = AllChem.rdmolops.GetFormalCharge(mol)
        xyz_file.write(f"! XTB2 OPT ALPB(WATER)\n\n")
        xyz_file.write(f"*xyz {charge} 1\n")
        
        for atom, coord in zip(atoms, coords):
            x, y, z = coord
            xyz_file.write(f"{atom} {x:.4f} {y:.4f} {z:.4f}\n")

        xyz_file.write(f"*")
    
    print(f"XYZ file has been created: {output_xyz_file}")

# Generate matching amino acid codes and rdkit molecules
def generate_combination(combinations, code_combinations)->tuple:
    for combination, code in zip(combinations, code_combinations):
        yield combination, code

if __name__ == "__main__":

    # Define the SMILES strings for the 20 amino acids including
    # charges for amino acid side chains that are charged in water, pH 7
    amino_acids = {
                'A': 'NC(C)C(O)=O',                      # Alanine (A)
                'G': 'NCC(O)=O',                         # Glycine (G)
                'S': 'NC(CO)C(O)=O',                     # Serine (S)
                'V': 'NC(C(C)C)C(O)=O',                  # Valine (V)
                'L': 'NC(CC(C)C)C(O)=O',                 # Leucine (L)
                'I': 'NC(C(C)CC)C(O)=O',                 # Isoleucine (I)
                'T': 'NC(C(C)O)C(O)=O',                  # Threonine (T)
                'D': 'N[C@@H](CC([O-])=O)C(O)=O',        # Aspartic Acid (D)
                'E': 'NC(C(O)=O)CCC([O-])=O',            # Glutamic Acid (E)
                'F': 'NC(CC1=CC=CC=C1)C(O)=O',           # Phenylalanine (F)
                'Y': 'NC(CC1=CC=C(O)C=C1)C(O)=O',        # Tyrosine (Y)
                'W': 'NC(Cc(c[nH]1)c2c1cccc2)C(O)=O',    # Tryptophan (W)
                'C': 'N[C@@H](CS)C(O)=O',                # Cysteine (C)
                'M': 'NC(CCSC)C(O)=O',                   # Methionine (M)
                'N': 'N[C@@H](CC(N)=O)C(O)=O',           # Asparagine (N)
                'Q': 'NC(CCC(N)=O)C(O)=O',               # Glutamine (Q)
                'K': 'NC(C(O)=O)CCCC[NH3+]',             # Lysine (K)
                'R': 'N[C@H](C(O)=O)CCCNC(N)=[NH2+]',    # Arginine (R)
                'H': 'N[C@@H](Cc1c[nH]cn1)C(O)=O',       # Histidine (H)
                'P': 'OC(C1NCCC1)=O'                     #Proline
    }

    # Convert the SMILES strings of amino acids into molecule (mol) objects
    amino_acid_mols = {name: Chem.MolFromSmiles(smiles) for name, smiles in amino_acids.items()}

    # Define reaction SMARTs for stepwise reaction - specifying which carbon is reacting with which nitrogen by describing their environments
    amide_bond_smart = "[N;$([NX3]-[CX4H1,CX4H2]-[C](=[O])[OH,NX3H1,NX3H0]);!$([N]-[C]=[O]):1].[C;$([C]([OH])(=[O])-[CX4H1,CX4H2]-[NH0,NH1,NH2]):2](=[O:3])[O]>>[NX3:1]-[C:2](=[O:3])"
    amide_bond_reaction = Reactions.ReactionFromSmarts(amide_bond_smart)

    # Produce all combinations of 4 amino acids out of the common 20
    combinations = list(itertools.product(amino_acid_mols.values(), repeat=4))
    code_combinations = [''.join(combo) for combo in itertools.product(amino_acid_mols.keys(), repeat=4)]

    # Set up cyclisation reaction SMARTs specifying which carbon is reacting with which nitrogen by describing their environments
    amide_cyclisation_bond_smart = "([N;$([NX3]-[CX4H1,CX4H2]-[C](=[O])[OH,NX3H1,NX3H0]);!$([N]-[C]=[O]):1].[C;$([C]([OH])(=[O])-[CX4H1,CX4H2]-[NH0,NH1,NH2]):2](=[O:3])[O])>>[NX3:1]-[C:2](=[O:3])"
    ring_closing_reaction = Reactions.ReactionFromSmarts(amide_cyclisation_bond_smart)

    # Define a set to remove duplicate codes from the list
    seen_inchis = set()
    
    # Perform linear peptide synthesis and cyclisation for each combination of 4 amino acids
    for combination, code in generate_combination(combinations, code_combinations):
        product = stepwise_reaction(combination)
        if product is not None:
            cyclic_product = ring_closing_reaction.RunReactants((product, ))
            if cyclic_product is not None:
                cyclic_product = cyclic_product[0][0]
                
                # Use inchi keys to remove duplicates using previously defined set
                pep_inchi = inchi.MolToInchiKey(cyclic_product)
                if pep_inchi not in seen_inchis:
                    seen_inchis.add(pep_inchi)
                    cyc_smiles = Chem.MolToSmiles(cyclic_product)

                    # Write the xyz input files for all unique tetracyclic peptides
                    smiles_to_xyz(cyc_smiles, code)
                
