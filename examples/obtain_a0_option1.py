#!/usr/bin/env python3
"""
Extract alpha_0 (base polarizabilities) from D4 reference data.

This script extracts isolated atom polarizabilities from the reference data
by finding reference systems with no surrounding atoms (refscount = 0).

The output format is ready to copy-paste into params.py.
"""

import torch
from tad_dftd4.reference import d4 as reference

print("="*80)
print("Extracting alpha_0 from D4 Reference Data")
print("="*80)
print("\nMethod: Extract from isolated atom references (refscount = 0)")
print("This gives quantum-mechanical isolated atom polarizabilities")
print("at 23 imaginary frequency points.")

# Load reference data
refalpha = reference.refalpha  # Shape: (n_elements, 7, 23)
refscount = reference.refscount  # Shape: (n_elements, 7)
refascale = reference.refascale  # Shape: (n_elements, 7)

print(f"\nReference data shapes:")
print(f"  refalpha:  {refalpha.shape}")
print(f"  refscount: {refscount.shape}")
print(f"  refascale: {refascale.shape}")

# Get actual number of elements from data
n_elements = refalpha.shape[0]
print(f"\nNumber of elements in reference data: {n_elements}")

# Initialize alpha_0 with zeros (pad to 119 for consistency with data structures)
max_elements = 119
alpha_0 = torch.zeros((max_elements, 23), dtype=torch.float64)

# Statistics
elements_with_isolated = []
elements_without_isolated = []
elements_with_multiple = []

print("\n" + "-"*80)
print("Extracting isolated atom polarizabilities...")
print("-"*80)
print(f"Note: Extracting from {n_elements} elements, padding to {max_elements} for consistency")

# Element symbols for reporting
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

# Extract for each element (only loop through available data)
for Z in range(n_elements):
    # Find reference systems with no surrounding atoms
    isolated_mask = (refscount[Z] == 0)
    isolated_indices = isolated_mask.nonzero(as_tuple=True)[0]
    
    if len(isolated_indices) > 0:
        # Use the first isolated reference (usually index 0)
        ref_idx = isolated_indices[0].item()
        
        # Extract and scale
        raw_alpha = refalpha[Z, ref_idx, :]
        scale = refascale[Z, ref_idx]
        alpha_0[Z, :] = scale * raw_alpha
        
        # Statistics
        if Z > 0:  # Skip dummy element
            elements_with_isolated.append((Z, element_symbols[Z], len(isolated_indices)))
            if len(isolated_indices) > 1:
                elements_with_multiple.append((Z, element_symbols[Z], len(isolated_indices)))
    else:
        if Z > 0:  # Skip dummy element
            elements_without_isolated.append((Z, element_symbols[Z]))

# Report statistics
print(f"\n✓ Elements with isolated atom references: {len(elements_with_isolated)}")
print(f"✗ Elements without isolated references: {len(elements_without_isolated)}")
print(f"⚠ Elements with multiple isolated references: {len(elements_with_multiple)}")

if elements_without_isolated:
    print(f"\nElements without isolated references:")
    for Z, sym in elements_without_isolated[:10]:  # Show first 10
        print(f"  Z={Z:3d} ({sym:2s})")
    if len(elements_without_isolated) > 10:
        print(f"  ... and {len(elements_without_isolated) - 10} more")

# Check if any are non-zero
nonzero_elements = (alpha_0.abs().sum(dim=1) > 1e-10).sum().item()
print(f"\nElements with non-zero alpha_0: {nonzero_elements}")

# Show examples
print("\n" + "-"*80)
print("Example extracted values:")
print("-"*80)
example_elements = [1, 6, 7, 8, 16]  # H, C, N, O, S
for Z in example_elements:
    if Z < len(element_symbols):
        sym = element_symbols[Z]
        alpha_sum = alpha_0[Z, :].abs().sum().item()
        if alpha_sum > 1e-10:
            print(f"\n{sym} (Z={Z}):")
            print(f"  First 5 frequencies: {alpha_0[Z, :5].tolist()}")
            print(f"  Sum over all frequencies: {alpha_sum:.6f}")
        else:
            print(f"\n{sym} (Z={Z}): All zeros")

# Generate output in params.py format
print("\n" + "="*80)
print("COPYABLE OUTPUT FOR params.py")
print("="*80)
print("\n# Add this to src/tad_dftd4/reference/d4/params.py:\n")

print("# Base polarizabilities extracted from isolated atom references")
print(f"# Shape: (119, 23) - padded with zeros for elements {n_elements}-118")
print(f"# Extracted from {n_elements} elements with refscount=0 (isolated atoms)")
print("# 23 imaginary frequency points")
print("alpha_0 = torch.tensor(")
print("    [")

# Format as the params.py style (one element per row with 23 values)
for Z in range(max_elements):
    values = alpha_0[Z, :].tolist()
    
    # Format comment
    if Z == 0:
        comment = "  # dummy"
    elif Z < len(element_symbols):
        comment = f"  # {element_symbols[Z]}"
    else:
        comment = ""
    
    # Format values (scientific notation with consistent width)
    formatted_values = ", ".join([f"{v:23.14e}" for v in values])
    print(f"        [{formatted_values}],{comment}")

print("    ],")
print("    dtype=torch.float64,")
print(")")

# Save to file
output_file = "alpha_0_extracted.txt"
with open(output_file, 'w') as f:
    f.write("# Base polarizabilities extracted from D4 reference data\n")
    f.write(f"# Shape: (119, 23) - extracted from {n_elements} elements, padded with zeros\n")
    f.write("# Method: Extracted from isolated atom references (refscount=0)\n")
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

# Additional analysis: Show which elements can benefit from noref mode
print("\n" + "="*80)
print("ANALYSIS: Suitability for noref mode")
print("="*80)

total_nonzero = 0
total_with_refs = 0

print("\nElements with good isolated atom references (non-zero alpha_0):")
print(f"{'Z':>3s} {'Symbol':>4s} {'Sum(alpha_0)':>15s} {'Max(alpha_0)':>15s}")
print("-" * 45)

for Z in range(1, min(37, n_elements)):  # Show first 36 elements (up to Kr)
    alpha_sum = alpha_0[Z, :].abs().sum().item()
    alpha_max = alpha_0[Z, :].abs().max().item()
    
    if alpha_sum > 1e-10:
        total_nonzero += 1
        sym = element_symbols[Z] if Z < len(element_symbols) else "?"
        print(f"{Z:3d} {sym:>4s} {alpha_sum:15.6e} {alpha_max:15.6e}")
    
    if (refscount[Z] > 0).any():
        total_with_refs += 1

print(f"\n✓ {total_nonzero} elements have non-zero isolated atom polarizabilities")
print(f"✓ These are good candidates for noref mode with alpha_0 as baseline")

print("\n" + "="*80)
print("Done! You can now:")
print("1. Copy the output above into params.py")
print("2. Use alpha_0_extracted.txt as reference")
print("3. Use these alpha_0 values in noref mode:")
print("   param = Param(alpha_0=alpha_0, dynamic_alpha_delta_w=corrections)")
print("="*80)

