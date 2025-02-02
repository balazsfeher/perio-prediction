# Machine learning predicts patient clinical responses to periodontal treatment
# Synthetic dataset generator 
#
# Balazs Feher, Eduardo H. de Souza Oliveira, Poliana Duarte, Andreas A. Werdich, William V. Giannobile, Magda Feres
# Harvard School of Dental Medicine
# balazs_feher@hsdm.harvard.edu
# 
# Last updated February 3, 2025

import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

file_path = "data_full.csv"
df_real = pd.read_csv(file_path)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_real)

# Add constraints to amoxicillin and metronidazole dosages (CTGAN automatically does it correctly for treatment duration)
constraints = [
    {
        "constraint_class": "ScalarRange",
        "constraint_parameters": {
            "column_name": "Tx_AMX",
            "low_value": 0,
            "high_value": 1.5,
            "strict_boundaries": False  # Allows 0 and 1.5
        }
    },
    {
        "constraint_class": "ScalarRange",
        "constraint_parameters": {
            "column_name": "Tx_MTZ",
            "low_value": 0,
            "high_value": 1.2,
            "strict_boundaries": False  # Allows 0, 0.75, and 1.2
        }
    }
]

synthesizer = CTGANSynthesizer(metadata, epochs=500, verbose=True)
synthesizer.add_constraints(constraints) 
synthesizer.fit(df_real)

num_rows = len(df_real)
df_synthetic = synthesizer.sample(num_rows)

def nearest_valid_value(value, allowed_values):
    return min(allowed_values, key=lambda x: abs(x - value))

df_synthetic["Tx_AMX"] = df_synthetic["Tx_AMX"].apply(lambda x: nearest_valid_value(x, [0, 1.5]))
df_synthetic["Tx_MTZ"] = df_synthetic["Tx_MTZ"].apply(lambda x: nearest_valid_value(x, [0, 0.75, 1.2])) # Remove 1.2 option for North American/European dataset

synthetic_file_path = "synthdata_internal.csv"
df_synthetic.to_csv(synthetic_file_path, index=False)

print(f"Synthetic dataset saved to {synthetic_file_path}")