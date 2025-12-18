
# if use no-ref mode, program uses alphe_base
# if not, then not using that 



# SPDX-Identifier: CC0-1.0
import tad_mctc as mctc
import torch

import tad_dftd4 as d4

numbers = mctc.convert.symbol_to_number(
    symbols="C C C C N C S H H H H H".split()
)

# coordinates in Bohr
positions = torch.tensor(
    [
        [-2.56745685564671, -0.02509985979910, 0.00000000000000],
        [-1.39177582455797, +2.27696188880014, 0.00000000000000],
        [+1.27784995624894, +2.45107479759386, 0.00000000000000],
        [+2.62801937615793, +0.25927727028120, 0.00000000000000],
        [+1.41097033661123, -1.99890996077412, 0.00000000000000],
        [-1.17186102298849, -2.34220576284180, 0.00000000000000],
        [-2.39505990368378, -5.22635838332362, 0.00000000000000],
        [+2.41961980455457, -3.62158019253045, 0.00000000000000],
        [-2.51744374846065, +3.98181713686746, 0.00000000000000],
        [+2.24269048384775, +4.24389473203647, 0.00000000000000],
        [+4.66488984573956, +0.17907568006409, 0.00000000000000],
        [-4.60044244782237, -0.17794734637413, 0.00000000000000],
    ]
, requires_grad=True)

# total charge of the system
charge = torch.tensor(0.0)

# TPSSh-D4-ATM parameters
c6_delta = positions.new_zeros((positions.shape[0], positions.shape[0]))
# c6_delta[0, 1] = 0.001
dynamic_alpha_delta = positions.new_zeros((positions.shape[0]))
dynamic_alpha_delta[0] = 0.1

param = d4.damping.Param(
    s6=positions.new_tensor(1.0),
    s8=positions.new_tensor(1.85897750),
    s9=positions.new_tensor(1.0),
    a1=positions.new_tensor(0.44286966),
    a2=positions.new_tensor(4.60230534),
    c6_delta=c6_delta,
    dynamic_alpha_delta=dynamic_alpha_delta,
)

# parameters can also be obtained using the functional name:
# param = d4.get_params(method="d4", functional="tpssh")

energy1 = d4.dftd4(numbers, positions, charge, param)

# class-based interface
disp = d4.dispersion.DispD4()
energy2 = disp.calculate(numbers, positions, charge, param)

torch.set_printoptions(precision=10)

# ref = torch.tensor(
#     [
#         -0.0020841344,
#         -0.0018971195,
#         -0.0018107513,
#         -0.0018305695,
#         -0.0021737693,
#         -0.0019484236,
#         -0.0022788253,
#         -0.0004080658,
#         -0.0004261866,
#         -0.0004199839,
#         -0.0004280768,
#         -0.0005108935,
#     ]
# )

# if not torch.allclose(energy1, ref, atol=1e-8):
#     print("Energy does not match for energy1!")
#     print("energy1:", energy1)
#     print("ref:", ref)
# if not torch.allclose(energy2, ref, atol=1e-8):
#     print("Energy does not match for energy2!")
#     print("energy2:", energy2)
#     print("ref:", ref)

# compute gradient of energy with respect to positions
(grad,) = torch.autograd.grad(energy1.sum(), positions)
print(f"forces: {-grad}")

print(f"energy1: {energy1}")
print(f"energy2: {energy2}")
# tensor([-0.0020841344, -0.0018971195, -0.0018107513, -0.0018305695,
#         -0.0021737693, -0.0019484236, -0.0022788253, -0.0004080658,
#         -0.0004261866, -0.0004199839, -0.0004280768, -0.0005108935])










print("\n" + "="*70)
print("Testing new alpha_mode='noref' with dynamic_alpha_delta_w")
print("="*70)

# Test the new noref mode with dynamic_alpha_delta_w
# Shape: (natom, 23) for 23 imaginary frequency points
natom = positions.shape[0]
nfreq = 23

# Create dummy dynamic_alpha_delta_w 
# For testing: use small uniform corrections at all frequencies
dynamic_alpha_delta_w = positions.new_ones((natom, nfreq)) * 0.0

print(f"\nNumber of atoms: {natom}")
print(f"Number of frequency points: {nfreq}")
print(f"dynamic_alpha_delta_w shape: {dynamic_alpha_delta_w.shape}")
print(f"dynamic_alpha_delta_w values: uniform 0.1 at all frequencies")

param_noref = d4.damping.Param(
    s6=positions.new_tensor(1.0),
    s8=positions.new_tensor(1.85897750),
    s9=positions.new_tensor(1.0),
    a1=positions.new_tensor(0.44286966),
    a2=positions.new_tensor(4.60230534),
    dynamic_alpha_delta_w=dynamic_alpha_delta_w,
)

# Calculate energy with noref mode
energy_noref = d4.dftd4(numbers, positions, charge, param_noref, alpha_mode="noref")

print(f"\nEnergy (noref mode):\n{energy_noref}")
print(f"Total energy (noref): {energy_noref.sum().item():.10f} Hartree")
print(f"Total energy (reference): {energy1.sum().item():.10f} Hartree")
print(f"Difference: {(energy_noref - energy1).sum().item():.10f} Hartree")

# Test gradient with noref mode
positions_noref = positions.clone().requires_grad_(True)
energy_noref_grad = d4.dftd4(numbers, positions_noref, charge, param_noref, alpha_mode="noref") # controls c6 integration using ref or not 
(grad_noref,) = torch.autograd.grad(energy_noref_grad.sum(), positions_noref)
print(f"\nForces (noref mode): {-grad_noref}")
print(f"Forces shape: {grad_noref.shape}")
print("✓ Gradient computation successful with noref mode!")





# # Test with user-supplied alpha_0
# print("\n" + "="*70)
# print("Testing with User-Supplied alpha_0 (Base Polarizabilities)")
# print("="*70)

# # Create custom alpha_0: shape (119, 23) for all elements at 23 frequencies
# # For this example, we'll set non-zero values only for elements present in our molecule
# max_atomic_number = 119
# custom_alpha_0 = positions.new_zeros((max_atomic_number, nfreq))

# # Get unique elements in our molecule
# unique_elements = torch.unique(numbers)
# print(f"\nElements in molecule: {unique_elements.tolist()}")
# element_symbols = mctc.convert.number_to_symbol(unique_elements)
# print(f"Corresponding symbols: {element_symbols}")

# # Set custom base polarizabilities for each element
# # In a real application, these would come from calculations or experiments
# # Here we use dummy values that vary by frequency
# for z in unique_elements:
#     # Create frequency-dependent base polarizabilities
#     # Higher frequencies get smaller values (typical behavior)
#     freq_decay = torch.linspace(1.0, 0.3, nfreq)
#     custom_alpha_0[z, :] = z.item() * 0.5 * freq_decay  # Scale by atomic number

# print(f"\ncustom_alpha_0 shape: {custom_alpha_0.shape}")
# print(f"Non-zero elements: {(custom_alpha_0 != 0).sum().item()} / {custom_alpha_0.numel()}")
# print(f"Example alpha_0 for Carbon (Z=6): {custom_alpha_0[6, :5]}... (first 5 frequencies)")
# print(f"Example alpha_0 for Nitrogen (Z=7): {custom_alpha_0[7, :5]}... (first 5 frequencies)")

# # Create param with both custom alpha_0 and dynamic corrections
# param_custom_alpha = d4.damping.Param(
#     s6=positions.new_tensor(1.0),
#     s8=positions.new_tensor(1.85897750),
#     s9=positions.new_tensor(1.0),
#     a1=positions.new_tensor(0.44286966),
#     a2=positions.new_tensor(4.60230534),
#     alpha_0=custom_alpha_0,  # User-supplied base polarizabilities
#     dynamic_alpha_delta_w=dynamic_alpha_delta_w,  # Additional corrections
# )

# # Calculate energy with custom alpha_0
# energy_custom_alpha = d4.dftd4(
#     numbers, positions, charge, param_custom_alpha, alpha_mode="noref"
# )

# print(f"\nEnergy (with custom alpha_0):\n{energy_custom_alpha}")
# print(f"Total energy (custom alpha_0): {energy_custom_alpha.sum().item():.10f} Hartree")
# print(f"Total energy (default alpha_0): {energy_noref.sum().item():.10f} Hartree")
# print(f"Difference: {(energy_custom_alpha - energy_noref).sum().item():.10f} Hartree")

# # Show the calculation breakdown
# print("\n" + "-"*70)
# print("Summary of noref mode options:")
# print("-"*70)
# print("1. Default: alpha_0=zeros, corrections via dynamic_alpha_delta_w")
# print("2. Custom: user-supplied alpha_0 + corrections via dynamic_alpha_delta_w")
# print("\nThe total polarizability is: alpha_total = alpha_0 + dynamic_alpha_delta_w")
# print("Then C6 is computed via Casimir-Polder integration: C6 = ∫ alpha_A * alpha_B dω")
# print("✓ All tests completed successfully!")
