# For XTB files
import os
import pandas as pd

# === Paths ===
file_list_path = "/nobackup/cm21sb/xtb_energy/sophie/sophie/jobs/random_out_files.txt"
source_dir = "/nobackup/cm21sb/xtb_energy/sophie/sophie/jobs/"

# === Load filenames from the list ===
with open(file_list_path, "r") as f:
    target_files = [line.strip() for line in f if line.strip()]

# === Data container ===
data = []

# === Process each file ===
# Map filenames to full paths by walking all subdirectories
all_files = {}
for root, dirs, files in os.walk(source_dir):
    for name in files:
        all_files[name] = os.path.join(root, name)

# Now check against target list
for filename in target_files:
    if filename in all_files:
        file_path = all_files[filename]
        
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            neg_freqs = []

            for line in lines:
                if line.strip().startswith("eigval :"):
                    values = line.strip().split()[2:]  # skip "eigval :"
                    for val in values:
                        try:
                            freq = float(val)
                            if freq < 0:
                                neg_freqs.append(freq)
                        except ValueError:
                            continue  # Ignore bad values

            # Save only if any negative frequencies were found
            if neg_freqs:
                data.append({"File": filename, "Negative Frequencies": neg_freqs})
    else:
        print(f"File not found: {filename}")

# === Write to CSV ===
df = pd.DataFrame(data)
df.to_csv("/nobackup/cm21sb/xtb_negative_frequencies.csv", index=False)

print("Done! Negative frequencies saved to xtb_negative_frequencies.csv")