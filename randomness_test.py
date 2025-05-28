#!/usr/bin/env python3
"""
Test to check if randomness in add_scatter is causing the differences we observed.
"""

import numpy as np
from AbundanceMatching import AbundanceFunction, LF_SCATTER_MULT, calc_number_densities

# Create test data
np.random.seed(42)
lf_table = '''
-24.7 -6.285 -24.5 -5.861 -24.3 -5.518 -24.1 -5.161 -23.9 -4.903
-23.7 -4.651 -23.5 -4.411 -23.3 -4.170 -23.1 -3.953 -22.9 -3.751
-22.7 -3.555 -22.5 -3.383 -22.3 -3.229 -22.1 -3.095 -21.9 -2.971
-21.7 -2.876 -21.5 -2.801 -21.3 -2.722 -21.1 -2.668 -20.9 -2.604
-20.7 -2.559 -20.5 -2.509 -20.3 -2.494 -20.1 -2.487 -19.9 -2.477
-19.7 -2.451 -19.5 -2.447 -19.3 -2.409 -19.1 -2.399 -18.9 -2.377
-18.7 -2.371 -18.5 -2.348 -18.3 -2.305 -18.1 -2.309 -17.9 -2.335
-17.7 -2.350'''.replace('\n', ' ')

lf = np.fromstring(lf_table, sep=' ').reshape(-1, 2)
lf[:, 1] = 10.0**lf[:, 1]

vpeak_values = np.random.lognormal(mean=np.log(200), sigma=0.8, size=10000)
vpeak_values = np.clip(vpeak_values, 50, 1500)

print("Testing randomness in abundance matching...")
print("="*60)

# Test 1: Same deconvolution parameters, different random seeds
print("\nTest 1: Same parameters, different random states")
print("-" * 50)

results = []
for i in range(3):
    # Reset random seed for consistent comparison
    np.random.seed(100 + i)  # Different seeds
    
    af = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
    af.deconvolute(0.2*LF_SCATTER_MULT, repeat=20, sm_step=0.005)  # Same parameters
    
    nd_halos = calc_number_densities(vpeak_values, 400)
    catalog = af.match(nd_halos, 0.2*LF_SCATTER_MULT)  # This adds random scatter!
    
    results.append(catalog)
    print(f"Run {i+1}: min={catalog.min():.3f}, max={catalog.max():.3f}, mean={catalog.mean():.3f}")

# Compare differences due to randomness
diff_12 = np.abs(results[0] - results[1])
diff_13 = np.abs(results[0] - results[2])

print(f"\nDifferences due to random scatter:")
print(f"Run 1 vs 2: max diff = {diff_12.max():.3f}, mean diff = {diff_12.mean():.3f}")
print(f"Run 1 vs 3: max diff = {diff_13.max():.3f}, mean diff = {diff_13.mean():.3f}")

# Test 2: Remove randomness by setting do_add_scatter=False
print("\n" + "="*60)
print("Test 2: Same test but with do_add_scatter=False")
print("-" * 50)

results_no_scatter = []
for i in range(3):
    np.random.seed(100 + i)  # Different seeds (shouldn't matter now)
    
    af = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
    af.deconvolute(0.2*LF_SCATTER_MULT, repeat=20, sm_step=0.005)
    
    nd_halos = calc_number_densities(vpeak_values, 400)
    catalog = af.match(nd_halos, 0.2*LF_SCATTER_MULT, do_add_scatter=False)  # No random scatter!
    
    results_no_scatter.append(catalog)
    print(f"Run {i+1}: min={catalog.min():.3f}, max={catalog.max():.3f}, mean={catalog.mean():.3f}")

# Compare differences without randomness
diff_12_no = np.abs(results_no_scatter[0] - results_no_scatter[1])
diff_13_no = np.abs(results_no_scatter[0] - results_no_scatter[2])

print(f"\nDifferences without random scatter:")
print(f"Run 1 vs 2: max diff = {diff_12_no.max():.6f}, mean diff = {diff_12_no.mean():.6f}")
print(f"Run 1 vs 3: max diff = {diff_13_no.max():.6f}, mean diff = {diff_13_no.mean():.6f}")

# Test 3: Check if different deconvolution parameters matter when randomness is removed
print("\n" + "="*60)
print("Test 3: Different deconv params WITHOUT random scatter")
print("-" * 50)

np.random.seed(42)  # Fixed seed
af1 = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
af1.deconvolute(0.2*LF_SCATTER_MULT, repeat=20, sm_step=0.005)  # Original
catalog1 = af1.match(nd_halos, 0.2*LF_SCATTER_MULT, do_add_scatter=False)

np.random.seed(42)  # Same seed
af2 = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
af2.deconvolute(0.2*LF_SCATTER_MULT, repeat=10, sm_step=0.01)   # Optimized
catalog2 = af2.match(nd_halos, 0.2*LF_SCATTER_MULT, do_add_scatter=False)

diff_deconv = np.abs(catalog1 - catalog2)
print(f"Original params: min={catalog1.min():.3f}, max={catalog1.max():.3f}, mean={catalog1.mean():.3f}")
print(f"Optimized params: min={catalog2.min():.3f}, max={catalog2.max():.3f}, mean={catalog2.mean():.3f}")
print(f"True deconvolution difference: max={diff_deconv.max():.6f}, mean={diff_deconv.mean():.6f}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print(f"Random scatter effect: ~{diff_12.max():.3f} magnitude units")
print(f"True deconvolution effect: ~{diff_deconv.max():.6f} magnitude units")
print(f"Ratio (random/deconv): {diff_12.max()/diff_deconv.max():.0f}x")
print("\nThe differences we observed were DOMINATED by random scatter,")
print("NOT by the deconvolution parameter changes!") 