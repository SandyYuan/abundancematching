#!/usr/bin/env python3
"""
Test script to compare original vs optimized deconvolution parameters.

This script compares:
1. Runtime differences between original (repeat=20, sm_step=0.005) vs optimized (repeat=10, sm_step=0.01)
2. Accuracy differences in the final abundance matching results
3. Quality of deconvolution (remainder analysis)

Usage:
    python parameter_optimization_test.py                    # Run comparison test
    python parameter_optimization_test.py --plot             # Show diagnostic plots
    python parameter_optimization_test.py --n-halos 100000  # Use fewer halos for faster testing
"""

import numpy as np
import argparse
import time
from AbundanceMatching import AbundanceFunction, LF_SCATTER_MULT, calc_number_densities

def create_test_data(n_halos=1000000):
    """Create the same test data as regression_test.py"""
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
    
    return _lf, halos

def AM_original(scatter, alpha, af, Vvir, Vmax, box_size):
    """Original AM function with default parameters"""
    # Deconvolution with original parameters
    remainder = af.deconvolute(scatter*LF_SCATTER_MULT, 20)  # repeat=20, sm_step=0.005 (default)
    x, nd = af.get_number_density_table()
    
    # Define halo proxy & calculate halo number density
    halo_proxy = Vvir * (Vmax/Vvir)**alpha
    nd_halos = calc_number_densities(halo_proxy, box_size)
    
    # Do abundance matching with some scatter
    catalog_sc = af.match(nd_halos, scatter*LF_SCATTER_MULT)
    return catalog_sc, remainder, nd_halos

def AM_optimized(scatter, alpha, af, Vvir, Vmax, box_size):
    """Optimized AM function with modified parameters"""
    # Deconvolution with optimized parameters
    remainder = af.deconvolute(scatter*LF_SCATTER_MULT, repeat=10, sm_step=0.01)  # FAST!
    x, nd = af.get_number_density_table()
    
    # Define halo proxy & calculate halo number density
    halo_proxy = Vvir * (Vmax/Vvir)**alpha
    nd_halos = calc_number_densities(halo_proxy, box_size)
    
    # Do abundance matching with some scatter
    catalog_sc = af.match(nd_halos, scatter*LF_SCATTER_MULT)
    return catalog_sc, remainder, nd_halos

def AM_conservative(scatter, alpha, af, Vvir, Vmax, box_size):
    """Conservative optimization - less aggressive parameter changes"""
    # Deconvolution with conservative parameters (between original and aggressive)
    remainder = af.deconvolute(scatter*LF_SCATTER_MULT, repeat=15, sm_step=0.007)  # More conservative
    x, nd = af.get_number_density_table()
    
    # Define halo proxy & calculate halo number density
    halo_proxy = Vvir * (Vmax/Vvir)**alpha
    nd_halos = calc_number_densities(halo_proxy, box_size)
    
    # Do abundance matching with some scatter
    catalog_sc = af.match(nd_halos, scatter*LF_SCATTER_MULT)
    return catalog_sc, remainder, nd_halos

def run_parameter_comparison(lf, halos, box_size=400, scatter=0.2, alpha=0.5, show_plots=False):
    """Run both original and optimized versions and compare results"""
    
    print(f"\n" + "="*80)
    print("PARAMETER OPTIMIZATION COMPARISON TEST (FIXED RANDOM SEED)")
    print("="*80)
    print(f"Test parameters:")
    print(f"  Number of halos: {len(halos)}")
    print(f"  Box size: {box_size} Mpc/h")
    print(f"  Scatter: {scatter}")
    print(f"  Alpha: {alpha}")
    print(f"  LF_SCATTER_MULT: {LF_SCATTER_MULT}")
    print(f"  Random seed: FIXED for all tests (eliminates scatter randomness)")
    
    # Test 1: Original parameters
    print(f"\n" + "-"*50)
    print("RUNNING ORIGINAL VERSION")
    print("-"*50)
    print("Parameters: repeat=20, sm_step=0.005 (default)")
    
    np.random.seed(12345)  # Fixed seed for reproducibility
    af_orig = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
    
    start_time = time.time()
    catalog_orig, remainder_orig, nd_halos_orig = AM_original(
        scatter, alpha, af_orig, halos['vpeak'], halos['vpeak'], box_size
    )
    time_orig = time.time() - start_time
    
    print(f"  Total runtime: {time_orig:.3f} seconds")
    print(f"  Catalog range: {catalog_orig.min():.3f} to {catalog_orig.max():.3f}")
    print(f"  Remainder stats: min={remainder_orig.min():.2e}, max={remainder_orig.max():.2e}")
    
    # Test 2: Optimized parameters  
    print(f"\n" + "-"*50)
    print("RUNNING OPTIMIZED VERSION")
    print("-"*50)
    print("Parameters: repeat=10, sm_step=0.01")
    
    np.random.seed(12345)  # SAME fixed seed for fair comparison!
    af_opt = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
    
    start_time = time.time()
    catalog_opt, remainder_opt, nd_halos_opt = AM_optimized(
        scatter, alpha, af_opt, halos['vpeak'], halos['vpeak'], box_size
    )
    time_opt = time.time() - start_time
    
    print(f"  Total runtime: {time_opt:.3f} seconds")
    print(f"  Catalog range: {catalog_opt.min():.3f} to {catalog_opt.max():.3f}")
    print(f"  Remainder stats: min={remainder_opt.min():.2e}, max={remainder_opt.max():.2e}")
    
    # Test 3: Conservative optimization
    print(f"\n" + "-"*50)
    print("RUNNING CONSERVATIVE VERSION")
    print("-"*50)
    print("Parameters: repeat=15, sm_step=0.007")
    
    np.random.seed(12345)  # SAME fixed seed for fair comparison!
    af_cons = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
    
    start_time = time.time()
    catalog_cons, remainder_cons, nd_halos_cons = AM_conservative(
        scatter, alpha, af_cons, halos['vpeak'], halos['vpeak'], box_size
    )
    time_cons = time.time() - start_time
    
    print(f"  Total runtime: {time_cons:.3f} seconds")
    print(f"  Catalog range: {catalog_cons.min():.3f} to {catalog_cons.max():.3f}")
    print(f"  Remainder stats: min={remainder_cons.min():.2e}, max={remainder_cons.max():.2e}")
    
    # Test 4: DETERMINISTIC TEST (no random scatter added)
    print(f"\n" + "-"*50)
    print("RUNNING DETERMINISTIC COMPARISON (NO SCATTER)")
    print("-"*50)
    print("This shows PURE deconvolution parameter effects")
    
    af_det_orig = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
    af_det_orig.deconvolute(scatter*LF_SCATTER_MULT, 20)  # Original params
    catalog_det_orig = af_det_orig.match(nd_halos_orig, scatter*LF_SCATTER_MULT, do_add_scatter=False)
    
    af_det_opt = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
    af_det_opt.deconvolute(scatter*LF_SCATTER_MULT, repeat=10, sm_step=0.01)  # Optimized params
    catalog_det_opt = af_det_opt.match(nd_halos_orig, scatter*LF_SCATTER_MULT, do_add_scatter=False)
    
    print(f"  Deterministic original: min={catalog_det_orig.min():.3f}, max={catalog_det_orig.max():.3f}")
    print(f"  Deterministic optimized: min={catalog_det_opt.min():.3f}, max={catalog_det_opt.max():.3f}")
    
    # Comparison Analysis
    print(f"\n" + "-"*50)
    print("COMPARISON ANALYSIS")
    print("-"*50)
    
    speedup_orig = time_orig / time_opt
    speedup_cons = time_orig / time_cons
    print(f"Runtime comparison:")
    print(f"  Original: {time_orig:.3f} seconds")
    print(f"  Optimized: {time_opt:.3f} seconds")
    print(f"  Conservative: {time_cons:.3f} seconds")
    print(f"  Speedup (Original vs Optimized): {speedup_orig:.2f}x")
    print(f"  Speedup (Original vs Conservative): {speedup_cons:.2f}x")
    
    # Catalog accuracy comparison (WITH RANDOM SCATTER - same seed)
    catalog_diff_orig = np.abs(catalog_orig - catalog_opt)
    catalog_rel_diff_orig = catalog_diff_orig / np.abs(catalog_orig)
    
    catalog_diff_cons = np.abs(catalog_orig - catalog_cons)
    catalog_rel_diff_cons = catalog_diff_cons / np.abs(catalog_orig)
    
    # Deterministic comparison (NO RANDOM SCATTER)
    catalog_diff_det = np.abs(catalog_det_orig - catalog_det_opt)
    catalog_rel_diff_det = catalog_diff_det / np.abs(catalog_det_orig)
    
    print(f"\nCatalog accuracy comparison (with same random seed):")
    print(f"  Max absolute difference (Original vs Optimized): {catalog_diff_orig.max():.2e}")
    print(f"  Mean absolute difference (Original vs Optimized): {catalog_diff_orig.mean():.2e}")
    print(f"  RMS difference (Original vs Optimized): {np.sqrt(np.mean(catalog_diff_orig**2)):.2e}")
    print(f"  Max relative difference (Original vs Optimized): {catalog_rel_diff_orig.max():.2e}")
    print(f"  Mean relative difference (Original vs Optimized): {catalog_rel_diff_orig.mean():.2e}")
    
    print(f"\nCatalog accuracy comparison (Original vs Conservative, same random seed):")
    print(f"  Max absolute difference: {catalog_diff_cons.max():.2e}")
    print(f"  Mean absolute difference: {catalog_diff_cons.mean():.2e}")
    print(f"  RMS difference: {np.sqrt(np.mean(catalog_diff_cons**2)):.2e}")
    print(f"  Max relative difference: {catalog_rel_diff_cons.max():.2e}")
    print(f"  Mean relative difference: {catalog_rel_diff_cons.mean():.2e}")
    
    print(f"\nPURE DECONVOLUTION EFFECT (no random scatter):")
    print(f"  Max absolute difference: {catalog_diff_det.max():.2e}")
    print(f"  Mean absolute difference: {catalog_diff_det.mean():.2e}")
    print(f"  RMS difference: {np.sqrt(np.mean(catalog_diff_det**2)):.2e}")
    print(f"  Max relative difference: {catalog_rel_diff_det.max():.2e}")
    print(f"  Mean relative difference: {catalog_rel_diff_det.mean():.2e}")
    
    # Remainder comparison
    remainder_diff_orig = np.abs(remainder_orig - remainder_opt)
    remainder_diff_cons = np.abs(remainder_orig - remainder_cons)
    
    print(f"\nRemainder (deconvolution quality) comparison:")
    print(f"  Max absolute difference (Original vs Optimized): {remainder_diff_orig.max():.2e}")
    print(f"  Mean absolute difference (Original vs Optimized): {remainder_diff_orig.mean():.2e}")
    print(f"  RMS difference (Original vs Optimized): {np.sqrt(np.mean(remainder_diff_orig**2)):.2e}")
    
    print(f"  Max absolute difference (Original vs Conservative): {remainder_diff_cons.max():.2e}")
    print(f"  Mean absolute difference (Original vs Conservative): {remainder_diff_cons.mean():.2e}")
    print(f"  RMS difference (Original vs Conservative): {np.sqrt(np.mean(remainder_diff_cons**2)):.2e}")
    
    # Statistical significance test
    from scipy import stats
    ks_stat_orig, ks_pvalue_orig = stats.ks_2samp(catalog_orig, catalog_opt)
    ks_stat_cons, ks_pvalue_cons = stats.ks_2samp(catalog_orig, catalog_cons)
    ks_stat_det, ks_pvalue_det = stats.ks_2samp(catalog_det_orig, catalog_det_opt)
    
    print(f"\nStatistical comparison (Kolmogorov-Smirnov test):")
    print(f"  KS statistic (Original vs Optimized, with scatter): {ks_stat_orig:.2e}")
    print(f"  p-value (Original vs Optimized, with scatter): {ks_pvalue_orig:.2e}")
    print(f"  KS statistic (Original vs Conservative, with scatter): {ks_stat_cons:.2e}")
    print(f"  p-value (Original vs Conservative, with scatter): {ks_pvalue_cons:.2e}")
    print(f"  KS statistic (Original vs Optimized, deterministic): {ks_stat_det:.2e}")
    print(f"  p-value (Original vs Optimized, deterministic): {ks_pvalue_det:.2e}")
    
    if ks_pvalue_det > 0.01:
        print(f"  Result: Deterministic distributions are statistically similar (p > 0.01)")
    else:
        print(f"  Result: Deterministic distributions are statistically different (p <= 0.01)")
    
    # Plots
    if show_plots:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # Plot 1: Catalog comparison scatter plots
        ax = axes[0, 0]
        ax.scatter(catalog_orig, catalog_opt, alpha=0.5, s=1, label='Original vs Optimized (with scatter)', color='red')
        ax.scatter(catalog_det_orig, catalog_det_opt, alpha=0.5, s=1, label='Original vs Optimized (deterministic)', color='blue')
        ax.plot([catalog_orig.min(), catalog_orig.max()], 
                [catalog_orig.min(), catalog_orig.max()], 'k--', label='Perfect agreement')
        ax.set_xlabel('Original Catalog')
        ax.set_ylabel('Catalog Comparison')
        ax.set_title('Catalog Value Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Catalog difference histograms
        ax = axes[0, 1]
        ax.hist(catalog_diff_orig, bins=50, alpha=0.7, edgecolor='red', label='With scatter', color='red')
        ax.hist(catalog_diff_det, bins=50, alpha=0.7, edgecolor='blue', label='Deterministic', color='blue')
        ax.set_xlabel('Absolute Difference')
        ax.set_ylabel('Count')
        ax.set_title('Catalog Difference Distribution')
        ax.grid(True, alpha=0.3)
        ax.axvline(catalog_diff_orig.mean(), color='red', linestyle='--', label=f'Mean (scatter): {catalog_diff_orig.mean():.2e}')
        ax.axvline(catalog_diff_det.mean(), color='blue', linestyle='--', label=f'Mean (determ): {catalog_diff_det.mean():.2e}')
        ax.legend()
        ax.set_yscale('log')
        
        # Plot 3: Remainder comparison
        x, nd = af_orig.get_number_density_table()
        ax = axes[0, 2]
        ax.plot(x, remainder_orig/nd, 'b-', linewidth=2, label='Original (repeat=20)', alpha=0.8)
        ax.plot(x, remainder_opt/nd, 'r-', linewidth=2, label='Optimized (repeat=10)', alpha=0.8)
        ax.plot(x, remainder_cons/nd, 'g-', linewidth=2, label='Conservative (repeat=15)', alpha=0.8)
        ax.set_xlabel('Absolute Magnitude')
        ax.set_ylabel('Fractional Remainder')
        ax.set_title('Deconvolution Quality Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot 4: Catalog distributions
        ax = axes[1, 0]
        ax.hist(catalog_orig, bins=50, alpha=0.5, label='Original', density=True)
        ax.hist(catalog_opt, bins=50, alpha=0.5, label='Optimized', density=True)
        ax.hist(catalog_cons, bins=50, alpha=0.5, label='Conservative', density=True)
        ax.set_xlabel('Catalog Values')
        ax.set_ylabel('Probability Density')
        ax.set_title('Catalog Value Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Relative difference vs magnitude (deterministic)
        ax = axes[1, 1]
        ax.scatter(catalog_det_orig, catalog_rel_diff_det, alpha=0.7, s=1, color='blue', label='Pure deconvolution effect')
        ax.set_xlabel('Original Catalog Value')
        ax.set_ylabel('Relative Difference')
        ax.set_title('Pure Deconvolution Effect\n(No Random Scatter)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 6: Runtime bar chart
        ax = axes[1, 2]
        methods = ['Original\n(repeat=20)', 'Optimized\n(repeat=10)', 'Conservative\n(repeat=15)']
        times = [time_orig, time_opt, time_cons]
        bars = ax.bar(methods, times, color=['blue', 'red', 'green'], alpha=0.7)
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title(f'Runtime Comparison\nOptimized: {speedup_orig:.2f}x speedup\nConservative: {speedup_cons:.2f}x speedup')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(times),
                   f'{time_val:.3f}s', ha='center', va='bottom')
        
        # Plot 7: Effect magnitude comparison
        ax = axes[2, 0]
        effects = ['Random\nScatter\n(different seeds)', 'Same Seed\n(deconv + scatter)', 'Pure\nDeconvolution\n(no scatter)']
        # Calculate effect from different random seeds (approximate from our previous test)
        random_effect = 2.0  # From our randomness test
        magnitudes = [random_effect, catalog_diff_orig.max(), catalog_diff_det.max()]
        bars = ax.bar(effects, magnitudes, color=['orange', 'red', 'blue'], alpha=0.7)
        ax.set_ylabel('Max Absolute Difference (mag)')
        ax.set_title('Effect Size Comparison')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add value labels
        for bar, val in zip(bars, magnitudes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                   f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    # Summary recommendation
    print(f"\n" + "="*50)
    print("RECOMMENDATION")
    print("="*50)
    
    print(f"PURE DECONVOLUTION EFFECTS (systematic error):")
    if catalog_rel_diff_det.max() < 0.01:  # Less than 1% difference
        print(f"✓ RECOMMENDED: Use optimized parameters")
        print(f"  - {speedup_orig:.2f}x speedup with negligible accuracy loss")
        print(f"  - Max relative error: {catalog_rel_diff_det.max():.2e} (< 1%)")
        recommendation = "RECOMMENDED"
    elif catalog_rel_diff_det.max() < 0.05:  # Less than 5% difference
        print(f"? CONDITIONAL: Consider optimized parameters")
        print(f"  - {speedup_orig:.2f}x speedup but with some accuracy loss")
        print(f"  - Max relative error: {catalog_rel_diff_det.max():.2e} ({catalog_rel_diff_det.max()*100:.1f}%)")
        print(f"  - Check if this accuracy is acceptable for your science case")
        recommendation = "CONDITIONAL"
    else:
        print(f"✗ NOT RECOMMENDED: Stick with original parameters")
        print(f"  - Accuracy loss too large: {catalog_rel_diff_det.max():.2e} ({catalog_rel_diff_det.max()*100:.1f}%)")
        recommendation = "NOT_RECOMMENDED"
    
    print(f"\nNote: Random scatter (when different seeds are used) causes ~2.0 mag differences,")
    print(f"which is {2.0/catalog_diff_det.max():.0f}x larger than the pure deconvolution effect.")
    
    return {
        'time_orig': time_orig,
        'time_opt': time_opt,
        'time_cons': time_cons,
        'speedup_orig': speedup_orig,
        'speedup_cons': speedup_cons,
        'catalog_orig': catalog_orig,
        'catalog_opt': catalog_opt,
        'catalog_cons': catalog_cons,
        'catalog_det_orig': catalog_det_orig,
        'catalog_det_opt': catalog_det_opt,
        'nd_halos': nd_halos_orig,
        'max_abs_diff_orig': catalog_diff_orig.max(),
        'max_abs_diff_cons': catalog_diff_cons.max(),
        'max_abs_diff_det': catalog_diff_det.max(),
        'max_rel_diff_orig': catalog_rel_diff_orig.max(),
        'max_rel_diff_cons': catalog_rel_diff_cons.max(),
        'max_rel_diff_det': catalog_rel_diff_det.max(),
        'mean_abs_diff_orig': catalog_diff_orig.mean(),
        'mean_abs_diff_cons': catalog_diff_cons.mean(),
        'mean_abs_diff_det': catalog_diff_det.mean(),
        'mean_rel_diff_orig': catalog_rel_diff_orig.mean(),
        'mean_rel_diff_cons': catalog_rel_diff_cons.mean(),
        'mean_rel_diff_det': catalog_rel_diff_det.mean(),
        'ks_pvalue_orig': ks_pvalue_orig,
        'ks_pvalue_cons': ks_pvalue_cons,
        'ks_pvalue_det': ks_pvalue_det,
        'remainder_orig': remainder_orig,
        'remainder_opt': remainder_opt,
        'remainder_cons': remainder_cons,
        'recommendation': recommendation
    }

def main():
    parser = argparse.ArgumentParser(description='Compare original vs optimized deconvolution parameters')
    parser.add_argument('--plot', action='store_true',
                       help='Show diagnostic plots')
    parser.add_argument('--n-halos', type=int, default=1000000,
                       help='Number of mock halos to generate (default: 1000000)')
    parser.add_argument('--scatter', type=float, default=0.2,
                       help='Scatter value to test (default: 0.2)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Alpha value to test (default: 0.5)')
    parser.add_argument('--box-size', type=float, default=400,
                       help='Box size in Mpc/h (default: 400)')
    parser.add_argument('--save-reference', type=str, default=None,
                       help='Save catalog results to .npz file for baseline reference (e.g., baseline_catalogs.npz)')
    
    args = parser.parse_args()
    
    print("Creating test data...")
    lf, halos = create_test_data(args.n_halos)
    
    print("Running parameter comparison...")
    results = run_parameter_comparison(
        lf, halos, 
        box_size=args.box_size, 
        scatter=args.scatter, 
        alpha=args.alpha,
        show_plots=args.plot
    )
    
    print(f"\nTest completed successfully!")
    print(f"Final speedup (Original vs Optimized): {results['speedup_orig']:.2f}x")
    print(f"Final speedup (Original vs Conservative): {results['speedup_cons']:.2f}x")
    print(f"Max relative accuracy loss (Original vs Optimized): {results['max_rel_diff_orig']:.2e}")
    print(f"Max relative accuracy loss (Original vs Conservative): {results['max_rel_diff_cons']:.2e}")
    print(f"")
    print(f"PURE DECONVOLUTION EFFECTS (systematic error):")
    print(f"Max relative accuracy loss (deterministic): {results['max_rel_diff_det']:.2e}")
    print(f"Recommendation: {results['recommendation']}")
    
    # Save reference catalogs if requested
    if args.save_reference:
        print(f"\nSaving reference catalogs to {args.save_reference}...")
        
        # Prepare metadata
        metadata = {
            'n_halos': args.n_halos,
            'scatter': args.scatter,
            'alpha': args.alpha,
            'box_size': args.box_size,
            'LF_SCATTER_MULT': LF_SCATTER_MULT,
            'random_seed': 12345,  # The fixed seed we used
            'speedup_orig': results['speedup_orig'],
            'speedup_cons': results['speedup_cons'],
            'max_rel_diff_det': results['max_rel_diff_det'],
            'recommendation': results['recommendation']
        }
        
        # Save all catalog arrays and key results
        np.savez_compressed(args.save_reference,
            # Input data
            vpeak_values=halos['vpeak'],
            lf_mags=lf[:,0],
            lf_phi=lf[:,1],
            
            # Catalog results (with scatter, same random seed)
            catalog_orig=results['catalog_orig'],
            catalog_opt=results['catalog_opt'], 
            catalog_cons=results['catalog_cons'],
            
            # Deterministic catalog results (no scatter)
            catalog_det_orig=results['catalog_det_orig'],
            catalog_det_opt=results['catalog_det_opt'],
            
            # Number densities
            nd_halos=results['nd_halos'],
            
            # Remainder arrays
            remainder_orig=results['remainder_orig'],
            remainder_opt=results['remainder_opt'],
            remainder_cons=results['remainder_cons'],
            
            # Runtime measurements
            time_orig=results['time_orig'],
            time_opt=results['time_opt'], 
            time_cons=results['time_cons'],
            
            # Accuracy metrics
            max_abs_diff_det=results['max_abs_diff_det'],
            max_rel_diff_det=results['max_rel_diff_det'],
            mean_abs_diff_det=results['mean_abs_diff_det'],
            mean_rel_diff_det=results['mean_rel_diff_det'],
            
            # Statistical test results
            ks_pvalue_det=results['ks_pvalue_det'],
            
            # Metadata (as arrays for npz compatibility)
            **{f'meta_{k}': np.array(v) for k, v in metadata.items()}
        )
        
        print(f"Reference catalogs saved successfully!")
        print(f"Contents saved:")
        print(f"  - Input halo catalog ({len(halos)} halos)")
        print(f"  - Luminosity function data")
        print(f"  - Catalog results for all 3 parameter sets (with scatter)")
        print(f"  - Deterministic catalog results (no scatter)")
        print(f"  - Runtime and accuracy metrics")
        print(f"  - All metadata (parameters, random seed, etc.)")
        print(f"\nTo load later: data = np.load('{args.save_reference}')")
        print(f"Access arrays like: data['catalog_orig'], data['catalog_det_opt'], etc.")

if __name__ == '__main__':
    main() 