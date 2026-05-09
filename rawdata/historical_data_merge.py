import pandas as pd
import os
import glob

# Path to your folder containing yearly CSV files
folder_path = "rawdata/historical_price_data"

# Get all CSV files in the folder
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Sort files to ensure chronological order
all_files = sorted(all_files)

print("Files found:")
for file in all_files:
    print(file)

# Read and combine all files
df_list = []

for file in all_files:
    df = pd.read_csv(file)
    df_list.append(df)

# Concatenate all dataframes
merged_df = pd.concat(df_list, ignore_index=True)

# Optional but recommended: remove duplicate rows (if any)
merged_df.drop_duplicates(inplace=True)

# Convert Date column to datetime (IMPORTANT)
merged_df['DATE'] = pd.to_datetime(merged_df['DATE'], dayfirst=True)

# Sort by Date just to be safe
merged_df = merged_df.sort_values(by='DATE')

# Reset index
merged_df.reset_index(drop=True, inplace=True)

# Save final merged file
output_path = "rawdata/HDFCBANK_2015_2026_merged.csv"
merged_df.to_csv(output_path, index=False)

print("\nMerge Complete.")
print(f"Final shape: {merged_df.shape}")
print(f"Saved to: {output_path}")