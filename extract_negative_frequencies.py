from retrieve_descriptors import MopacFile
import os
import pandas as pd

input_directory = "/nobackup/cm21sb/sandwich_xyz/mopac/more_than_four_bonds_mopac/"
# input_directory = "C:/Users/cm21sb/OneDrive - University of Leeds/Year 4/Sophie Blanch/code/test_analysis/negative_freqs"

successful_codes = []
negative_freq_data = []

for file in os.listdir(input_directory):
    if file.endswith(".out"):
        code = str(file)[:4]
        mopac = MopacFile(os.path.join(input_directory, file))

        try:
            freq_dict = mopac.get_frequencies()
            frequencies = freq_dict.keys()
            negative_freqs = [float(freq) for freq in frequencies if freq < 0]

            if negative_freqs:
                negative_freq_data.append({"Code": code, "Negative Frequencies": negative_freqs})

            else:
                successful_codes.append(code) # index codes list by directory name

        except:
            pass


negative_freq_df = pd.DataFrame(negative_freq_data)
negative_freq_df.to_csv("negative_frequencies_more_than_four_bonds.csv", index=False)

# Save the successful codes to a CSV file
pd.DataFrame({"Code": successful_codes}).to_csv("successful_codes_more_than_four_bonds.csv", index=False)

print("CSV files written successfully!")