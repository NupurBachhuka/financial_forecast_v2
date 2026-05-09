import pandas as pd
import os
import glob

# Folder containing NIFTY BANK yearly CSVs
folder_path = "rawdata/niftybank_index_data"

# Get all CSV files
all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

print("Files found:")
for file in all_files:
    print(file)

df_list = []

for file in all_files:
    df = pd.read_csv(file)

    # Clean column names (remove hidden spaces)
    df.columns = df.columns.str.strip()

    df_list.append(df)

# Merge all years
merged_df = pd.concat(df_list, ignore_index=True)

# Remove duplicates (safety)
merged_df.drop_duplicates(inplace=True)

# Print column names once for verification
print("\nColumns detected:")
print(merged_df.columns)

merged_df['Date'] = pd.to_datetime(merged_df['Date'], dayfirst=True)

# Sort chronologically
merged_df = merged_df.sort_values(by='Date')

# Reset index
merged_df.reset_index(drop=True, inplace=True)

# Save merged file
output_path = "rawdata/NIFTYBANK_2015_2026_merged.csv"
merged_df.to_csv(output_path, index=False)

print("\nMerge Complete.")
print("Final Shape:", merged_df.shape)
print("Saved at:", output_path)