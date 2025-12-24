#!/usr/bin/env python3
"""Test Camelot table extraction on the Security Category table."""

import camelot
from pathlib import Path

pdf_path = Path("examples/data/sample_soc2.pdf")

print("=" * 80)
print("CAMELOT LATTICE MODE (detects table borders)")
print("=" * 80)

try:
    # Lattice mode uses table borders to detect structure
    tables = camelot.read_pdf(str(pdf_path), pages='27', flavor='lattice')
    print(f"\nFound {len(tables)} tables")
    
    if tables:
        for i, table in enumerate(tables):
            print(f"\n--- Table {i} ---")
            df = table.df
            print(f"Shape: {df.shape} (rows x cols)")
            print(f"Accuracy: {table.accuracy}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            print(f"\nColumn emptiness check:")
            for col_idx in range(df.shape[1]):
                non_empty = df.iloc[:, col_idx].notna().sum()
                non_empty_non_blank = df.iloc[:, col_idx].apply(lambda x: bool(str(x).strip()) if x is not None else False).sum()
                print(f"  Col {col_idx}: {non_empty_non_blank} non-empty cells")
except Exception as e:
    print(f"Lattice mode failed: {e}")

print("\n" + "=" * 80)
print("CAMELOT STREAM MODE (detects text positioning)")
print("=" * 80)

try:
    # Stream mode uses text position and spacing
    tables = camelot.read_pdf(str(pdf_path), pages='27', flavor='stream')
    print(f"\nFound {len(tables)} tables")
    
    if tables:
        for i, table in enumerate(tables):
            print(f"\n--- Table {i} ---")
            df = table.df
            print(f"Shape: {df.shape} (rows x cols)")
            print(f"Accuracy: {table.accuracy}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            print(f"\nColumn emptiness check:")
            for col_idx in range(df.shape[1]):
                non_empty_non_blank = df.iloc[:, col_idx].apply(lambda x: bool(str(x).strip()) if x is not None else False).sum()
                print(f"  Col {col_idx}: {non_empty_non_blank} non-empty cells")
except Exception as e:
    print(f"Stream mode failed: {e}")
