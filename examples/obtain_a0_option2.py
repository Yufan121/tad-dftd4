#!/usr/bin/env python3
"""
Extract alpha_0 from isolated atoms (Option 2: Direct method).

This script extracts isolated atom polarizabilities directly from D4 reference
data where refscount=0, providing a clean baseline without charge corrections.
This is the same as Option 1 but presented as a simpler alternative approach.
"""

import torch
from tad_dftd4.reference import d4 as reference

print("="*80)
print("Extracting alpha_0 from Isolated Atoms (Option 2)")
print("="*80)
print("\nMethod: Direct extraction from isolated atom references")
print("This gives baseline polarizabilities without charge/environment corrections")

# Load reference data
refalpha = reference.refalpha  # Shape: (n_elements, 7, 23)
refscount = reference.refscount  # Shape: (n_elements, 7)
refascale = reference.refascale  # Shape: (n_elements, 7)

n_elements = refalpha.shape[0]
max_elements = 119
nfreq = 23

print(f"\nReference data covers {n_elements} elements")
print(f"Extracting isolated atoms (refscount = 0) for each element")

# Initialize alpha_0
alpha_0 = torch.zeros((max_elements, nfreq), dtype=torch.float64)

print("\n" + "-"*80)
print("Extracting isolated atom polarizabilities...")
print("-"*80)

element_symbols = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

# Extract for each element
elements_found = 0
for Z in range(n_elements):
    # Find isolated atom references (refscount = 0)
    isolated_mask = (refscount[Z] == 0)
    isolated_indices = isolated_mask.nonzero(as_tuple=True)[0]
    
    if len(isolated_indices) > 0:
        # Use the first isolated reference
        ref_idx = isolated_indices[0].item()
        
        # Extract: alpha_0 = refascale * refalpha
        alpha_0[Z, :] = refascale[Z, ref_idx] * refalpha[Z, ref_idx, :]
        
        if Z > 0:  # Skip dummy element
            elements_found += 1
            sym = element_symbols[Z] if Z < len(element_symbols) else "?"
            alpha_sum = alpha_0[Z, :].sum().item()
            print(f"  Z={Z:3d} ({sym:>2s}): alpha_sum = {alpha_sum:12.6f}")

print(f"\n✓ Extracted {elements_found} elements with isolated atom references")

# Generate output
print("\n" + "="*80)
print("COPYABLE OUTPUT FOR params.py")
print("="*80)
print("\n# Add this to src/tad_dftd4/reference/d4/params.py:\n")

print("# Base polarizabilities from isolated atoms (refscount=0)")
print(f"# Shape: (119, 23) - extracted from {n_elements} elements, padded with zeros")
print("# Method: Direct extraction without charge/environment corrections")
print("# 23 imaginary frequency points")
print("alpha_0 = torch.tensor(")
print("    [")

for Z in range(max_elements):
    values = alpha_0[Z, :].tolist()
    
    if Z == 0:
        comment = "  # dummy"
    elif Z < len(element_symbols):
        comment = f"  # {element_symbols[Z]}"
    else:
        comment = ""
    
    formatted_values = ", ".join([f"{v:23.14e}" for v in values])
    print(f"        [{formatted_values}],{comment}")

print("    ],")
print("    dtype=torch.float64,")
print(")")

# Save to file
output_file = "alpha_0_isolated.txt"
with open(output_file, 'w') as f:
    f.write("# Base polarizabilities from isolated atoms\n")
    f.write(f"# Shape: (119, 23) - extracted from {n_elements} elements, padded with zeros\n")
    f.write("# Method: Direct extraction from refscount=0 references\n")
    f.write("# Format: One line per element, 23 frequency points\n\n")
    
    f.write("alpha_0 = torch.tensor(\n")
    f.write("    [\n")
    for Z in range(max_elements):
        values = alpha_0[Z, :].tolist()
        if Z < len(element_symbols):
            comment = f"  # {Z:3d} {element_symbols[Z]:2s}"
        else:
            comment = f"  # {Z:3d}"
        
        formatted_values = ", ".join([f"{v:23.14e}" for v in values])
        f.write(f"        [{formatted_values}],{comment}\n")
    
    f.write("    ],\n")
    f.write("    dtype=torch.float64,\n")
    f.write(")\n")

print(f"\n✓ Output also saved to: {output_file}")

# Summary
nonzero_count = (alpha_0.abs().sum(dim=1) > 1e-10).sum().item()
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nElements with non-zero alpha_0: {nonzero_count}")
print(f"Total elements in output: {max_elements} (padded with zeros)")
print("\nThese are baseline isolated atom polarizabilities")
print("suitable for use as starting values in noref mode.")
print("\n" + "="*80)
print("Done! You can:")
print("1. Copy the output above into params.py")
print(f"2. Use {output_file} as reference")
print("3. Use these values with dynamic_alpha_delta_w for corrections")
print("="*80)

