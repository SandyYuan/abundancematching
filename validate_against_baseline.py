#!/usr/bin/env python3
"""
Enhanced Validation Script with Performance Testing

This script loads the baseline reference catalogs and compares them against
results from potentially optimized abundance matching code. Now includes
comprehensive timing measurements and performance baseline tracking.

Usage:
    python validate_against_baseline.py                    # Quick validation with timing
    python validate_against_baseline.py --detailed         # Detailed comparison
    python validate_against_baseline.py --plot             # With plots
    python validate_against_baseline.py --save-timing baseline_timing.npz  # Save timing baseline
"""

import numpy as np
import argparse
import time
from AbundanceMatching import AbundanceFunction, LF_SCATTER_MULT, calc_number_densities

def load_baseline(filename='baseline_catalogs.npz'):
    """Load baseline reference data"""
    try:
        data = np.load(filename)
        print(f"Loaded baseline data from {filename}")
        print(f"  - Number of halos: {data['meta_n_halos']}")
        print(f"  - Scatter: {data['meta_scatter']}")
        print(f"  - Alpha: {data['meta_alpha']}")
        print(f"  - Random seed: {data['meta_random_seed']}")
        print(f"  - Baseline recommendation: {data['meta_recommendation']}")
        return data
    except FileNotFoundError:
        print(f"Error: Baseline file '{filename}' not found!")
        print("Run: python parameter_optimization_test.py --save-reference baseline_catalogs.npz")
        return None

def load_timing_baseline(filename):
    """Load timing baseline data"""
    try:
        data = np.load(filename)
        print(f"Loaded timing baseline from {filename}")
        return data
    except FileNotFoundError:
        print(f"Timing baseline '{filename}' not found - will create new baseline")
        return None

def benchmark_function(func, *args, **kwargs):
    """Benchmark a function with multiple runs for statistical accuracy"""
    times = []
    result = None
    
    # Run multiple times for accurate timing
    for i in range(5):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return result, {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'times': times
    }

def run_current_code_with_timing(baseline_data):
    """Run the current abundance matching code with detailed timing measurements"""
    
    # Extract parameters from baseline
    scatter = float(baseline_data['meta_scatter'])
    alpha = float(baseline_data['meta_alpha'])
    box_size = float(baseline_data['meta_box_size'])
    vpeak_values = baseline_data['vpeak_values']
    lf_mags = baseline_data['lf_mags']
    lf_phi = baseline_data['lf_phi']
    
    print(f"\nRunning current code with detailed timing measurements...")
    print(f"  Scatter: {scatter}, Alpha: {alpha}, Box size: {box_size}")
    
    # Use the SAME random seed as baseline for fair comparison
    np.random.seed(int(baseline_data['meta_random_seed']))
    
    # Reconstruct the test setup
    lf = np.column_stack([lf_mags, lf_phi])
    
    results = {}
    timing_results = {}
    
    print(f"\n" + "="*60)
    print("DETAILED TIMING MEASUREMENTS")
    print("="*60)
    
    # Benchmark 1: calc_number_densities
    print("⏱️  Benchmarking calc_number_densities()...")
    halo_proxy = vpeak_values * (vpeak_values/vpeak_values)**alpha
    nd_halos, calc_nd_timing = benchmark_function(calc_number_densities, halo_proxy, box_size)
    timing_results['calc_number_densities'] = calc_nd_timing
    print(f"    Time: {calc_nd_timing['mean']:.6f} ± {calc_nd_timing['std']:.6f} seconds")
    
    results['nd_halos'] = nd_halos
    
    # Benchmark 2: AbundanceFunction initialization
    print("⏱️  Benchmarking AbundanceFunction.__init__()...")
    af_orig, init_timing = benchmark_function(AbundanceFunction, lf[:,0], lf[:,1], (-27, -5))
    timing_results['af_init'] = init_timing
    print(f"    Time: {init_timing['mean']:.6f} ± {init_timing['std']:.6f} seconds")
    
    # Benchmark 3: Deconvolution (original parameters)
    print("⏱️  Benchmarking af.deconvolute() [original params]...")
    def deconv_orig():
        af = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
        return af.deconvolute(scatter*LF_SCATTER_MULT, 20)
    
    remainder_orig, deconv_orig_timing = benchmark_function(deconv_orig)
    timing_results['deconvolute_orig'] = deconv_orig_timing
    print(f"    Time: {deconv_orig_timing['mean']:.6f} ± {deconv_orig_timing['std']:.6f} seconds")
    
    # Benchmark 4: Deconvolution (optimized parameters)  
    print("⏱️  Benchmarking af.deconvolute() [optimized params]...")
    def deconv_opt():
        af = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
        return af.deconvolute(scatter*LF_SCATTER_MULT, repeat=10, sm_step=0.01)
    
    remainder_opt, deconv_opt_timing = benchmark_function(deconv_opt)
    timing_results['deconvolute_opt'] = deconv_opt_timing
    print(f"    Time: {deconv_opt_timing['mean']:.6f} ± {deconv_opt_timing['std']:.6f} seconds")
    print(f"    Speedup vs original: {deconv_orig_timing['mean']/deconv_opt_timing['mean']:.2f}x")
    
    # Benchmark 5: Matching (original)
    print("⏱️  Benchmarking af.match() [original]...")
    af_orig = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
    af_orig.deconvolute(scatter*LF_SCATTER_MULT, 20)
    
    def match_orig():
        np.random.seed(int(baseline_data['meta_random_seed']))  # Reset for consistency
        return af_orig.match(nd_halos, scatter*LF_SCATTER_MULT)
    
    catalog_orig, match_orig_timing = benchmark_function(match_orig)
    timing_results['match_orig'] = match_orig_timing
    print(f"    Time: {match_orig_timing['mean']:.6f} ± {match_orig_timing['std']:.6f} seconds")
    
    results['catalog_orig'] = catalog_orig
    
    # Benchmark 6: Matching (optimized)
    print("⏱️  Benchmarking af.match() [optimized]...")
    af_opt = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
    af_opt.deconvolute(scatter*LF_SCATTER_MULT, repeat=10, sm_step=0.01)
    
    def match_opt():
        np.random.seed(int(baseline_data['meta_random_seed']))  # Reset for consistency
        return af_opt.match(nd_halos, scatter*LF_SCATTER_MULT)
    
    catalog_opt, match_opt_timing = benchmark_function(match_opt)
    timing_results['match_opt'] = match_opt_timing
    print(f"    Time: {match_opt_timing['mean']:.6f} ± {match_opt_timing['std']:.6f} seconds")
    
    results['catalog_opt'] = catalog_opt
    
    # Benchmark 7: Deterministic matching
    print("⏱️  Benchmarking af.match() [deterministic]...")
    af_det = AbundanceFunction(lf[:,0], lf[:,1], (-27, -5))
    af_det.deconvolute(scatter*LF_SCATTER_MULT, 20)
    
    def match_det():
        return af_det.match(nd_halos, scatter*LF_SCATTER_MULT, do_add_scatter=False)
    
    catalog_det, match_det_timing = benchmark_function(match_det)
    timing_results['match_det'] = match_det_timing
    print(f"    Time: {match_det_timing['mean']:.6f} ± {match_det_timing['std']:.6f} seconds")
    
    results['catalog_det_orig'] = catalog_det
    
    # Calculate total pipeline times
    total_orig_time = (calc_nd_timing['mean'] + init_timing['mean'] + 
                      deconv_orig_timing['mean'] + match_orig_timing['mean'])
    total_opt_time = (calc_nd_timing['mean'] + init_timing['mean'] + 
                     deconv_opt_timing['mean'] + match_opt_timing['mean'])
    
    timing_results['total_orig_pipeline'] = {
        'mean': total_orig_time, 'components': [
            'calc_number_densities', 'af_init', 'deconvolute_orig', 'match_orig'
        ]
    }
    timing_results['total_opt_pipeline'] = {
        'mean': total_opt_time, 'components': [
            'calc_number_densities', 'af_init', 'deconvolute_opt', 'match_opt'  
        ]
    }
    
    print(f"\n" + "="*60)
    print("PIPELINE TIMING SUMMARY")
    print("="*60)
    print(f"Original pipeline total: {total_orig_time:.6f} seconds")
    print(f"Optimized pipeline total: {total_opt_time:.6f} seconds")
    print(f"Parameter optimization speedup: {total_orig_time/total_opt_time:.2f}x")
    
    # Component breakdown
    print(f"\nComponent breakdown (% of original pipeline):")
    print(f"  calc_number_densities: {calc_nd_timing['mean']:.6f}s ({calc_nd_timing['mean']/total_orig_time*100:.1f}%)")
    print(f"  af_init:              {init_timing['mean']:.6f}s ({init_timing['mean']/total_orig_time*100:.1f}%)")
    print(f"  deconvolute_orig:     {deconv_orig_timing['mean']:.6f}s ({deconv_orig_timing['mean']/total_orig_time*100:.1f}%)")
    print(f"  match_orig:           {match_orig_timing['mean']:.6f}s ({match_orig_timing['mean']/total_orig_time*100:.1f}%)")
    
    return results, timing_results

def compare_timing_against_baseline(current_timing, baseline_timing):
    """Compare current timing results against baseline"""
    
    print(f"\n" + "="*70)
    print("PERFORMANCE COMPARISON AGAINST BASELINE")
    print("="*70)
    
    if baseline_timing is None:
        print("⚠️  No timing baseline found - current run will serve as baseline")
        return None
    
    improvements = {}
    
    # Compare individual components
    for component in ['calc_number_densities', 'af_init', 'deconvolute_orig', 'match_orig']:
        if component in baseline_timing and component in current_timing:
            baseline_time = float(baseline_timing[component]['mean'])
            current_time = current_timing[component]['mean']
            speedup = baseline_time / current_time
            
            status = "🚀" if speedup > 1.05 else "⚡" if speedup > 0.95 else "🐌"
            
            print(f"{status} {component}:")
            print(f"    Baseline: {baseline_time:.6f}s")
            print(f"    Current:  {current_time:.6f}s")
            print(f"    Speedup:  {speedup:.2f}x")
            
            improvements[component] = {
                'baseline': baseline_time,
                'current': current_time,
                'speedup': speedup
            }
    
    # Compare total pipeline
    if 'total_orig_pipeline' in baseline_timing and 'total_orig_pipeline' in current_timing:
        baseline_total = float(baseline_timing['total_orig_pipeline']['mean'])
        current_total = current_timing['total_orig_pipeline']['mean']
        total_speedup = baseline_total / current_total
        
        print(f"\n🎯 OVERALL PIPELINE PERFORMANCE:")
        print(f"    Baseline total: {baseline_total:.6f}s")
        print(f"    Current total:  {current_total:.6f}s")
        print(f"    Overall speedup: {total_speedup:.2f}x")
        
        improvements['total_pipeline'] = {
            'baseline': baseline_total,
            'current': current_total,
            'speedup': total_speedup
        }
        
        if total_speedup > 1.1:
            print(f"🎉 EXCELLENT: {total_speedup:.1f}x algorithmic improvement!")
        elif total_speedup > 1.05:
            print(f"✅ GOOD: {total_speedup:.1f}x algorithmic improvement")
        elif total_speedup > 0.95:
            print(f"➡️  NEUTRAL: No significant performance change")
        else:
            print(f"⚠️  SLOWER: {1/total_speedup:.1f}x performance regression")
    
    return improvements

def save_timing_baseline(timing_results, filename):
    """Save current timing results as baseline"""
    print(f"\nSaving timing baseline to {filename}...")
    
    # Convert timing results to numpy arrays for npz format
    save_data = {}
    for component, timing in timing_results.items():
        if isinstance(timing, dict):
            for key, value in timing.items():
                if isinstance(value, (list, np.ndarray)):
                    save_data[f'{component}_{key}'] = np.array(value)
                else:
                    save_data[f'{component}_{key}'] = np.array(value)
        else:
            save_data[component] = np.array(timing)
    
    # Add metadata
    save_data['baseline_created'] = np.array(time.time())
    save_data['n_benchmark_runs'] = np.array(5)
    
    np.savez_compressed(filename, **save_data)
    print(f"Timing baseline saved successfully!")

def compare_results(baseline_data, current_results, detailed=False):
    """Compare current results against baseline (original accuracy check)"""
    
    print(f"\n" + "="*70)
    print("ACCURACY VALIDATION RESULTS")
    print("="*70)
    
    # Check array shapes
    arrays_to_check = ['catalog_orig', 'catalog_opt', 'catalog_det_orig']
    all_passed = True
    
    for array_name in arrays_to_check:
        baseline_array = baseline_data[array_name]
        current_array = current_results[array_name]
        
        if baseline_array.shape != current_array.shape:
            print(f"❌ SHAPE MISMATCH: {array_name}")
            print(f"   Baseline: {baseline_array.shape}, Current: {current_array.shape}")
            all_passed = False
            continue
            
        # Calculate differences
        abs_diff = np.abs(baseline_array - current_array)
        rel_diff = abs_diff / np.abs(baseline_array)
        
        max_abs_diff = abs_diff.max()
        max_rel_diff = rel_diff.max()
        mean_abs_diff = abs_diff.mean()
        rms_diff = np.sqrt(np.mean(abs_diff**2))
        
        # Define tolerance levels
        abs_tolerance = 1e-10  # Very strict for numerical consistency
        rel_tolerance = 1e-8   # Very strict relative tolerance
        
        passed = (max_abs_diff <= abs_tolerance) and (max_rel_diff <= rel_tolerance)
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"{status} {array_name}:")
        print(f"   Max absolute difference: {max_abs_diff:.2e}")
        print(f"   Max relative difference: {max_rel_diff:.2e}")
        print(f"   Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"   RMS difference: {rms_diff:.2e}")
        
        if detailed:
            print(f"   Baseline range: [{baseline_array.min():.6f}, {baseline_array.max():.6f}]")
            print(f"   Current range:  [{current_array.min():.6f}, {current_array.max():.6f}]")
            
            # Statistical test
            from scipy import stats
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_array, current_array)
            print(f"   KS test: statistic={ks_stat:.2e}, p-value={ks_pvalue:.2e}")
        
        if not passed:
            all_passed = False
            print(f"   ⚠️  Differences exceed tolerance!")
            print(f"      Tolerance: abs={abs_tolerance:.0e}, rel={rel_tolerance:.0e}")
            
            # Show where the largest differences occur
            worst_idx = np.argmax(abs_diff)
            print(f"      Worst case at index {worst_idx}:")
            print(f"        Baseline: {baseline_array[worst_idx]:.10f}")
            print(f"        Current:  {current_array[worst_idx]:.10f}")
            print(f"        Difference: {abs_diff[worst_idx]:.2e}")
    
    print(f"\n" + "-"*70)
    if all_passed:
        print("🎉 ACCURACY VALIDATION PASSED: All arrays match baseline within tolerance!")
        print("   Your algorithmic optimizations preserve numerical accuracy.")
    else:
        print("⚠️  ACCURACY VALIDATION FAILED: Some arrays differ from baseline!")
        print("   Check your algorithmic changes for unintended side effects.")
    
    return all_passed

def plot_comparison(baseline_data, current_results):
    """Create plots comparing baseline vs current results"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    arrays_to_plot = [
        ('catalog_orig', 'Original Parameters'),
        ('catalog_opt', 'Optimized Parameters'), 
        ('catalog_det_orig', 'Deterministic (No Scatter)'),
    ]
    
    for i, (array_name, title) in enumerate(arrays_to_plot):
        if i >= 3:  # Only plot first 3
            break
            
        row, col = divmod(i, 2)
        ax = axes[row, col]
        
        baseline = baseline_data[array_name]
        current = current_results[array_name]
        
        # Scatter plot
        ax.scatter(baseline, current, alpha=0.6, s=1)
        ax.plot([baseline.min(), baseline.max()], 
                [baseline.min(), baseline.max()], 'r--', 
                label='Perfect agreement')
        
        ax.set_xlabel('Baseline Catalog')
        ax.set_ylabel('Current Catalog')
        ax.set_title(f'{title}\nComparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add difference statistics to plot
        diff = np.abs(baseline - current)
        ax.text(0.05, 0.95, f'Max diff: {diff.max():.2e}\nMean diff: {diff.mean():.2e}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Difference histogram
    ax = axes[1, 1]
    for array_name, label in [('catalog_orig', 'Original'), ('catalog_opt', 'Optimized')]:
        baseline = baseline_data[array_name]
        current = current_results[array_name]
        diff = np.abs(baseline - current)
        ax.hist(diff, bins=50, alpha=0.6, label=label, density=True)
    
    ax.set_xlabel('Absolute Difference')
    ax.set_ylabel('Density')
    ax.set_title('Difference Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Validate algorithmic optimizations with performance tracking')
    parser.add_argument('--baseline', default='baseline_catalogs.npz',
                       help='Baseline reference file (default: baseline_catalogs.npz)')
    parser.add_argument('--timing-baseline', default=None,
                       help='Timing baseline file (default: auto-detect)')
    parser.add_argument('--save-timing', type=str, default=None,
                       help='Save current timing results as baseline (e.g., timing_baseline.npz)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed comparison statistics')
    parser.add_argument('--plot', action='store_true',
                       help='Show comparison plots')
    parser.add_argument('--tolerance', type=float, default=1e-10,
                       help='Absolute tolerance for validation (default: 1e-10)')
    
    args = parser.parse_args()
    
    # Auto-detect timing baseline if not specified
    if args.timing_baseline is None:
        args.timing_baseline = args.baseline.replace('.npz', '_timing.npz')
    
    # Load baseline data
    baseline_data = load_baseline(args.baseline)
    if baseline_data is None:
        return 1
    
    # Load timing baseline
    timing_baseline = load_timing_baseline(args.timing_baseline)
    
    # Run current code with timing
    current_results, current_timing = run_current_code_with_timing(baseline_data)
    
    # Compare accuracy against baseline
    accuracy_passed = compare_results(baseline_data, current_results, args.detailed)
    
    # Compare performance against timing baseline
    performance_improvements = compare_timing_against_baseline(current_timing, timing_baseline)
    
    # Save timing baseline if requested
    if args.save_timing:
        save_timing_baseline(current_timing, args.save_timing)
    
    # Optional plotting
    if args.plot:
        plot_comparison(baseline_data, current_results)
    
    # Final summary
    print(f"\n" + "="*70)
    print("FINAL VALIDATION SUMMARY")
    print("="*70)
    
    accuracy_status = "✅ PASSED" if accuracy_passed else "❌ FAILED"
    print(f"Accuracy validation: {accuracy_status}")
    
    if performance_improvements:
        total_speedup = performance_improvements.get('total_pipeline', {}).get('speedup', 1.0)
        if total_speedup > 1.05:
            perf_status = f"🚀 IMPROVED ({total_speedup:.2f}x faster)"
        elif total_speedup > 0.95:
            perf_status = "➡️  NEUTRAL (no significant change)"
        else:
            perf_status = f"🐌 SLOWER ({1/total_speedup:.2f}x)"
        print(f"Performance validation: {perf_status}")
    else:
        print(f"Performance validation: 📊 BASELINE ESTABLISHED")
    
    # Return appropriate exit code
    return 0 if accuracy_passed else 1

if __name__ == '__main__':
    exit(main()) 