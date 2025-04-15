# Necessary imports
import os
import pandas as pd
import re

# Define paths
file_list_path = "/nobackup/cm21sb/pm6_energy/random_aux_files.txt"
source_dir = "/nobackup/cm21sb/pm6_energy/"

# Load filenames from list
with open(file_list_path, "r") as f:
    target_files = [line.strip() for line in f if line.strip()]

# Establish empty list for frequencies
data = []

# Map filenames to full paths by walking subdirectories
all_files = {}
for root, dirs, files in os.walk(source_dir):
    for name in files:
        all_files[name] = os.path.join(root, name)

# Prcoess each file name
for filename in target_files:
    if filename in all_files:
        file_path = all_files[filename]
    else:
        print(f"File not found: {filename}")
        continue

    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        neg_freqs = []
        capture = False

        for line in lines:
            if "VIB._FREQ:CM" in line:
                capture = True
                continue

            if capture:
                if re.search(r'[A-Za-z]', line):  # stop if non-numeric line found
                    break
                try:
                    freqs = [float(val) for val in line.strip().split()]
                    negs = [f for f in freqs if f < 0]
                    neg_freqs.extend(negs)
                except ValueError:
                    continue

        if neg_freqs:
            data.append({"File": filename, "Negative Frequencies": neg_freqs})

# Write csv
df = pd.DataFrame(data)
df.to_csv("/nobackup/cm21sb/pm6_negative_frequencies.csv", index=False)

print("Done! Negative frequencies saved to pm6_negative_frequencies.csv")
