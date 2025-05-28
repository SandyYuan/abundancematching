#!/usr/bin/env python3
"""
Comprehensive test for AbundanceMatching package.

This script can:
1. Run abundance matching with diagnostic plots (--plot)
2. Save reference catalogs for regression testing (--save-reference)
3. Run regression tests against saved references (default)

Usage:
    python regression_test.py                    # Run regression test
    python regression_test.py --save-reference   # Create reference catalog
    python regression_test.py --plot             # Interactive with plots
    python regression_test.py --plot --save-reference  # Both plots and save reference
"""

import numpy as np
import argparse
import time
from AbundanceMatching import AbundanceFunction, LF_SCATTER_MULT, calc_number_densities, add_scatter, rematch

def create_test_data(n_halos=1000000):
    """Create the same test data as test_readme.py"""
    # Same luminosity function data
    _lf_table = '''
    -24.7 -6.285
    -24.5 -5.861
    -24.3 -5.518
    -24.1 -5.161
    -23.9 -4.903
    -23.7 -4.651
    -23.5 -4.411
    -23.3 -4.170
    -23.1 -3.953
    -22.9 -3.751
    -22.7 -3.555
    -22.5 -3.383
    -22.3 -3.229
    -22.1 -3.095
    -21.9 -2.971
    -21.7 -2.876
    -21.5 -2.801
    -21.3 -2.722
    -21.1 -2.668
    -20.9 -2.604
    -20.7 -2.559
    -20.5 -2.509
    -20.3 -2.494
    -20.1 -2.487
    -19.9 -2.477
    -19.7 -2.451
    -19.5 -2.447
    -19.3 -2.409
    -19.1 -2.399
    -18.9 -2.377
    -18.7 -2.371
    -18.5 -2.348
    -18.3 -2.305
    -18.1 -2.309
    -17.9 -2.335
    -17.7 -2.350'''.strip().replace('\n', ' ')
    
    _lf = np.fromstring(_lf_table, sep=' ').reshape(-1, 2)
    _lf[:, 1] = 10.0**_lf[:, 1]
    
    # Same mock halo data (reproducible with same seed)
    np.random.seed(42)
    vpeak_values = np.random.lognormal(mean=np.log(200), sigma=0.8, size=n_halos)
    vpeak_values = np.clip(vpeak_values, 50, 1500)
    halos = np.array([(v,) for v in vpeak_values], dtype=[('vpeak', 'f8')])
    
    print(f"Created {len(halos)} mock halos")
    print(f"vpeak range: {halos['vpeak'].min():.1f} - {halos['vpeak'].max():.1f} km/s")
    print(f"Mean vpeak: {halos['vpeak'].mean():.1f} km/s")
    
    return _lf, halos

def run_abundance_matching(lf, halos, box_size=100, scatter=0.2, show_plots=False):
    """Run the abundance matching workflow with optional plotting"""
    # Create abundance function

    af = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))

    if show_plots:
        import matplotlib.pyplot as plt
        # Create subplots for better visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Check the abundance function
        ax1.semilogy(lf[:,0], lf[:,1], 'o', label='Original LF data', markersize=6, alpha=0.7)
        x = np.linspace(-27, -5, 101)
        ax1.semilogy(x, af(x), '-', label='Interpolated/Extrapolated', linewidth=2)
        ax1.set_xlabel('Absolute Magnitude')
        ax1.set_ylabel('Number Density [Mpc⁻³ mag⁻¹]')
        ax1.set_title('Galaxy Luminosity Function')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Deconvolute
    print("  Running deconvolution...")
    start_time = time.time()
    remainder = af.deconvolute(scatter*LF_SCATTER_MULT, 20)
    deconv_time = time.time() - start_time
    print(f"    Completed in {deconv_time:.3f} seconds")

    if show_plots:
        # Plot 2: Deconvolution quality check
        x, nd = af.get_number_density_table()
        ax2.plot(x, remainder/nd, 'r-', linewidth=2, label=f'Scatter = {scatter}')
        ax2.set_xlabel('Absolute Magnitude')
        ax2.set_ylabel('Fractional Remainder')
        ax2.set_title('Deconvolution Quality Check')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    # Get number densities
    print("  Calculating number densities (includes sorting)...")
    start_time = time.time()
    nd_halos = calc_number_densities(halos['vpeak'], box_size)
    nd_time = time.time() - start_time
    print(f"    Completed in {nd_time:.3f} seconds")
    
    # Do abundance matching
    catalog = af.match(nd_halos)
    catalog_sc = af.match(nd_halos, scatter*LF_SCATTER_MULT)

    print("  Running abundance matching (deconvolved)...")
    start_time = time.time()
    catalog_deconv = af.match(nd_halos, scatter*LF_SCATTER_MULT, False)
    deconv_time = time.time() - start_time
    print(f"    Completed in {deconv_time:.3f} seconds")
    
    return {
        'halos_vpeak': halos['vpeak'],
        'nd_halos': nd_halos,
        'catalog_no_scatter': catalog,
        'catalog_with_scatter': catalog_sc,
        'catalog_deconv': catalog_deconv,
        'scatter_value': scatter,
        'box_size': box_size,
        'n_halos': len(halos)
    }

def print_catalog_summary(results):
    """Print detailed summary of what's in the catalog"""
    print(f"\n" + "="*60)
    print("REFERENCE CATALOG CONTENTS")
    print("="*60)
    
    print(f"Input Parameters:")
    print(f"  Number of halos: {results['n_halos']}")
    print(f"  Box size: {results['box_size']} Mpc/h")
    print(f"  Scatter value: {results['scatter_value']}")
    
    print(f"\nHalo Properties:")
    vpeak = results['halos_vpeak'] 
    print(f"  vpeak range: {vpeak.min():.1f} - {vpeak.max():.1f} km/s")
    print(f"  vpeak mean/std: {vpeak.mean():.1f} ± {vpeak.std():.1f} km/s")
    print(f"  vpeak median: {np.median(vpeak):.1f} km/s")
    
    print(f"\nNumber Densities:")
    nd = results['nd_halos']
    print(f"  Range: {nd.min():.2e} - {nd.max():.2e} Mpc⁻³")
    print(f"  Mean: {nd.mean():.2e} Mpc⁻³")
    
    print(f"\nCatalog Results (Absolute Magnitudes):")
    catalogs = {
        'No scatter': results['catalog_no_scatter'],
        'With scatter': results['catalog_with_scatter'], 
        'Deconvolved': results['catalog_deconv']
    }
    
    for name, catalog in catalogs.items():
        print(f"  {name:12s}: min={catalog.min():.3f}, max={catalog.max():.3f}, mean={catalog.mean():.3f}, std={catalog.std():.3f}")
    
    print(f"\nArray shapes and data types:")
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:20s}: {value.shape} {value.dtype}")
        else:
            print(f"  {key:20s}: {type(value).__name__} = {value}")
    
    print("="*60)

def compare_results(new_results, reference_file='reference_catalog.npz', tolerance=1e-10):
    """Compare new results with reference"""
    try:
        ref_data = np.load(reference_file)
        
        catalogs_to_check = ['catalog_no_scatter', 'catalog_with_scatter', 'catalog_deconv']
        all_match = True
        
        print(f"\nRegression test results (tolerance: {tolerance}):")
        print("=" * 60)
        
        for cat_name in catalogs_to_check:
            if cat_name in ref_data:
                diff = np.abs(new_results[cat_name] - ref_data[cat_name])
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                matches = max_diff <= tolerance
                
                status = "PASS" if matches else "FAIL"
                print(f"{cat_name:20s}: {status:4s} (max: {max_diff:.2e}, mean: {mean_diff:.2e})")
                
                if not matches:
                    all_match = False
                    print(f"  First 5 differences: {diff[:5]}")
                    print(f"  Indices of max diff: {np.argmax(diff)}")
            else:
                print(f"{cat_name:20s}: MISSING from reference")
                all_match = False
        
        # Check metadata
        meta_checks = ['scatter_value', 'box_size', 'n_halos']
        print(f"\nMetadata checks:")
        for meta in meta_checks:
            if meta in ref_data:
                match = new_results[meta] == ref_data[meta]
                print(f"  {meta:15s}: {'PASS' if match else 'FAIL'}")
                if not match:
                    all_match = False
                    print(f"    New: {new_results[meta]}, Ref: {ref_data[meta]}")
        
        print("=" * 60)
        print(f"Overall result: {'PASS - All tests passed!' if all_match else 'FAIL - Some tests failed!'}")
        return all_match
        
    except FileNotFoundError:
        print(f"Reference file '{reference_file}' not found!")
        print("Run with --save-reference to create it first.")
        return False
    except Exception as e:
        print(f"Error in comparison: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Comprehensive AbundanceMatching test and regression tool')
    parser.add_argument('--save-reference', action='store_true', 
                       help='Save current results as reference catalog')
    parser.add_argument('--plot', action='store_true',
                       help='Show diagnostic plots')
    parser.add_argument('--tolerance', type=float, default=1e-10,
                       help='Numerical tolerance for comparison (default: 1e-10)')
    parser.add_argument('--reference-file', default='reference_catalog.npz',
                       help='Reference catalog file (default: reference_catalog.npz)')
    parser.add_argument('--n-halos', type=int, default=1000000,
                       help='Number of mock halos to generate (default: 1000000)')
    
    args = parser.parse_args()
    
    print("Creating test data...")
    lf, halos = create_test_data(args.n_halos)
    
    print("Running abundance matching...")
    results = run_abundance_matching(lf, halos, show_plots=args.plot)
    
    if args.save_reference:
        print_catalog_summary(results)
        np.savez(args.reference_file, **results)
        print(f"\nReference catalog saved to '{args.reference_file}'")
        print("You can now run regression tests against this reference.")
    
    if not args.save_reference:
        success = compare_results(results, args.reference_file, args.tolerance)
        exit(0 if success else 1)
    elif args.plot:
        print("\nShowing plots (close plot windows to continue)")

if __name__ == "__main__":
    main() 