#!/usr/bin/env python3
"""Detailed analysis of Camelot stream extraction."""

import camelot
from pathlib import Path

pdf_path = Path("examples/data/sample_soc2.pdf")

tables = camelot.read_pdf(str(pdf_path), pages='27', flavor='stream')

if tables:
    table = tables[0]
    df = table.df
    
    print(f"Shape: {df.shape}")
    print(f"Accuracy: {table.accuracy}\n")
    
    print("All rows and columns:")
    print("=" * 100)
    
    for idx, row in df.iterrows():
        print(f"\nRow {idx}:")
        for col_idx in range(df.shape[1]):
            cell = row[col_idx]
            if cell and str(cell).strip():
                display = str(cell).replace('\n', ' ')[:80]
                print(f"  Col {col_idx}: {display}")
