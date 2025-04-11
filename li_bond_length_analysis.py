# Necessary imports
import numpy as np
import os
from scipy.spatial import distance
import matplotlib.pyplot as plt
from collections import Counter


if __name__ == "__main__":

    # Define xyz directory
    xyz_dir = "//nobackup//cm21sb//sandwich_xyz//zero_bonds//"
    # xyz_dir = "C://Users//cm21sb//OneDrive - University of Leeds//Year 4//Sophie Blanch//code//tetracyclic_new//sandwiches//random_sandwiches"

    # Establish amino acid code list
    AA_codes = []

    # Count the xyz files in certain directories, this example is for zero Li-O bonds
    for file in os.listdir(xyz_dir):
        if file.endswith(".xyz"):
            code = str(file)[:4]

            AA_codes.extend(list(code))
    
    freq = Counter(AA_codes)

    labels, values = zip(*freq.items())

    plt.figure(figsize=(8, 5))

    sorted_labels = sorted(labels)
    sorted_values = [freq[label] for label in sorted_labels]

    # Produce a bar chart
    plt.bar(sorted_labels, sorted_values, color='skyblue', edgecolor='black')
    plt.xlabel("Amino Acids")
    plt.ylabel("Frequency")
    plt.title("Amino Acid Frequency Histogram for Complexes with \n no Li-O Distances Shorter than 2.5 Å")
    plt.savefig("amino_acid_histogram_0.png", format="png")

    plt.close()

