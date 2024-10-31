import h5py
import pandas as pd

file_path = '20240910_SI.h5'  # Replace with your file path
with h5py.File(file_path, 'r') as f:
    # Print all top-level keys
    keys = list(f.keys())
    print("Keys in the file:", keys)


list_df = []

for k in keys:

    with h5py.File(file_path, 'r') as f:
        # Navigate through groups if needed
        data = pd.read_hdf(file_path, key=k)


    list_df.append(data)

x=1

