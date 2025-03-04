import numpy as np
import os
import shutil
from scipy.spatial import distance

def read_xyz(filename):
    """ Reads an XYZ file and extracts atomic symbols and coordinates. """
    with open(filename, 'r') as f:
        lines = f.readlines()

    if not lines:
        print(f"Warning: {filename} is empty.")
        return [], []
    try:
        num_atoms = int(lines[0].strip())  # First line: number of atoms
    except ValueError:
        print(f"Error: Invalid number of atoms in {filename}.")
        return [], []
            
    atoms = []
    coords = []

    for line in lines[2:num_atoms+2]:  # Skip first two lines (header and comment)
        parts = line.split()
        atoms.append(parts[0])  # Atomic symbol
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])  # XYZ coordinates
    
    return atoms, np.array(coords)

def calculate_bond_length(coord1, coord2):
    """ Computes Euclidean distance between two points in 3D space. """
    return distance.euclidean(coord1, coord2)

def find_li_o_bonds(xyz_file):
    """ Finds and calculates bond lengths between Li and all O atoms. """
    atoms, coords = read_xyz(xyz_file)

    # Find indices of Li and O atoms
    li_indices = [i for i, atom in enumerate(atoms) if atom == "Li"]
    o_indices = [i for i, atom in enumerate(atoms) if atom == "O"]

    if not li_indices:
        print("No lithium atom found in the XYZ file.")
        return
    if not o_indices:
        print("No oxygen atoms found in the XYZ file.")
        return

    # Assuming only one Li atom, take the first found
    li_index = li_indices[0]  
    li_coord = coords[li_index]

    # Calculate bond lengths
    bond_lengths = {}
    for o_index in o_indices:
        o_coord = coords[o_index]
        bond_length = calculate_bond_length(li_coord, o_coord)
        bond_lengths[(li_index, o_index)] = bond_length

    return bond_lengths

def filter_bonds_by_length(bonds_lengths, threshold):
    bond_values = bond_lengths.values()
    bond_count = sum(1 for length in bond_values if length < threshold)
    return bond_count

if __name__ == "__main__":

    xyz_dir = "//nobackup//cm21sb//sandwich_xyz//"

    output_dirs = {
        0: os.path.join(xyz_dir, "zero_bonds"),
        1: os.path.join(xyz_dir, "one_bond"),
        2: os.path.join(xyz_dir, "two_bonds"),
        3: os.path.join(xyz_dir, "three_bonds"),
        4: os.path.join(xyz_dir, "four_bonds"),
        "more": os.path.join(xyz_dir, "more_than_four_bonds")
    }

    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)


    for file in os.listdir(xyz_dir):
        filepath = os.path.join(xyz_dir, file)
        code = str(file)[:4]
        
        if file.endswith('.xyz'):
            bond_lengths = find_li_o_bonds(filepath)

            if bond_lengths:
                count = filter_bonds_by_length(bond_lengths, 2.5)

                print(f"{code} has {count} bond(s) less than 2.5 Å ")
                
                if count in output_dirs:
                    destination_dir = output_dirs[count]
                else:
                    destination_dir = output_dirs["more"]

                shutil.move(filepath, os.path.join(destination_dir, file))
                print(f"Moved {file} to {destination_dir}")
                       

