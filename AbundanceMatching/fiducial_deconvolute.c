#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define GAUSS_CACHE_SIZE 4097
#define GAUSS_CACHE_CENTER 2048 /* (4097-1)/2 */
#define GAUSS_CACHE_MAX 8
#define GAUSS_CACHE_INV_STEP 256 /* 2048/8 */

/* Pre-computed constants for better performance */
#define SQRT_2PI 2.5066282746310005
#define INV_SQRT_2PI 0.3989422804014327
#define LOG10_E 0.43429448190325182
#define INV_LOG10 2.302585092994046

/* Inline gaussian cache lookup for better performance */
static inline double gaussian_cache(double dm, double scatter, const double * g_cache)
{
  double dm_scaled = dm/scatter*GAUSS_CACHE_INV_STEP + GAUSS_CACHE_CENTER;
  int b = (int)dm_scaled;
  
  /* Early return for out of bounds */
  if (b < 0 || b >= GAUSS_CACHE_SIZE-1) return 0.0;
  
  double frac = dm_scaled - b;
  return (g_cache[b] + frac*(g_cache[b+1]-g_cache[b]))/scatter;
}

void convolved_fit(double * af_key, double * af_val, int num_af_points, 
    double * smm, double * mf, int MASS_BINS, double scatter, 
    int repeat, double sm_step)
{
  if (!repeat || !scatter) return;

  int i, j, k, jstart, jend;
  double sm, fid_sm; 
  double nh, new_nh, lnh, lnnh, mnh, sum;

  /* Allocate memory with better alignment for cache performance */
  double * new_smm = (double *) calloc(MASS_BINS, sizeof(double));
  double * smm_conv = (double *) malloc(MASS_BINS * sizeof(double));
  double * g_cache = (double *) malloc(GAUSS_CACHE_SIZE * sizeof(double));

  /* Pre-compute Gaussian cache */
  for (i = 0; i < GAUSS_CACHE_SIZE; i++) {
    sm = ((double)i)/GAUSS_CACHE_INV_STEP - GAUSS_CACHE_MAX;
    g_cache[i] = exp(-0.5*sm*sm) * INV_SQRT_2PI;
  }

  /* Find active range once */
  jstart = 0;
  jend = MASS_BINS;
  for (i = 1; i < MASS_BINS; i++) {
    if (smm[i] && !smm[i-1]) jstart = i;
    if (!smm[i] && smm[i-1]) {
      jend = i;
      break;
    }
  }
  
  /* Pre-compute frequently used values */
  const double af_key_last = af_key[num_af_points-1];
  const double af_val_last = af_val[num_af_points-1];
  const double af_key_diff_last = af_key_last - af_key[num_af_points-2];

  for (k = 0; k < repeat; ++k) {
    /* Reset new_smm efficiently */
    memset(new_smm + jstart, 0, (jend - jstart) * sizeof(double));
    
    fid_sm = af_key_last + 1.0;
    nh = pow(10.0, af_val_last) * af_key_diff_last;
    
    mnh = 0.0;
    while (mnh < nh && fid_sm >= 0.0) {
      sum = 0.0;
      
      /* Unroll loop by 4 for better performance */
      j = jstart;
      int jend_4 = jstart + ((jend - jstart) & ~3);
      
      /* Process 4 elements at a time */
      for (; j < jend_4; j += 4) {
        double gc0 = gaussian_cache(fid_sm - smm[j], scatter, g_cache);
        double gc1 = gaussian_cache(fid_sm - smm[j+1], scatter, g_cache);
        double gc2 = gaussian_cache(fid_sm - smm[j+2], scatter, g_cache);
        double gc3 = gaussian_cache(fid_sm - smm[j+3], scatter, g_cache);
        
        smm_conv[j] = gc0;
        smm_conv[j+1] = gc1;
        smm_conv[j+2] = gc2;
        smm_conv[j+3] = gc3;
        
        sum += gc0 * mf[j] + gc1 * mf[j+1] + gc2 * mf[j+2] + gc3 * mf[j+3];
      }
      
      /* Handle remainder */
      for (; j < jend; j++) {
        smm_conv[j] = gaussian_cache(fid_sm - smm[j], scatter, g_cache);
        sum += smm_conv[j] * mf[j];
      }
      
      mnh += sum * sm_step;
      if (!isfinite(mnh)) mnh = 0.0;
      
      /* Update new_smm */
      const double update_factor = sm_step * af_key_last;
      for (j = jstart; j < jend; j++) {
        new_smm[j] += smm_conv[j] * update_factor;
      }
      
      fid_sm -= sm_step;
    }
    
    lnh = log10(nh);
    for (i = num_af_points - 1; i > 0; i--) {
      /* Pre-compute values for this iteration */
      const double af_val_i = af_val[i];
      const double af_val_prev = af_val[i-1];
      const double af_key_i = af_key[i];
      const double af_key_prev = af_key[i-1];
      
      const double pow10_diff = pow(10.0, af_val_prev) - pow(10.0, af_val_i);
      const double af_key_diff = af_key_i - af_key_prev;
      const double af_val_diff = af_val_prev - af_val_i;
      
      new_nh = nh + pow10_diff * af_key_diff / (INV_LOG10 * af_val_diff);
      lnnh = log10(new_nh);
      
      const double inv_lnh_diff = 1.0 / (lnh - lnnh);
      
      while (mnh < new_nh && fid_sm >= 0.0) {
        sum = 0.0;
        
        /* Unrolled loop again */
        j = jstart;
        int jend_4 = jstart + ((jend - jstart) & ~3);
        
        for (; j < jend_4; j += 4) {
          double gc0 = gaussian_cache(fid_sm - smm[j], scatter, g_cache);
          double gc1 = gaussian_cache(fid_sm - smm[j+1], scatter, g_cache);
          double gc2 = gaussian_cache(fid_sm - smm[j+2], scatter, g_cache);
          double gc3 = gaussian_cache(fid_sm - smm[j+3], scatter, g_cache);
          
          smm_conv[j] = gc0;
          smm_conv[j+1] = gc1;
          smm_conv[j+2] = gc2;
          smm_conv[j+3] = gc3;
          
          sum += gc0 * mf[j] + gc1 * mf[j+1] + gc2 * mf[j+2] + gc3 * mf[j+3];
        }
        
        for (; j < jend; j++) {
          smm_conv[j] = gaussian_cache(fid_sm - smm[j], scatter, g_cache);
          sum += smm_conv[j] * mf[j];
        }
        
        mnh += sum * sm_step;
        
        /* Compute sm inline */
        sm = af_key_prev + (log10(mnh) - lnnh) * inv_lnh_diff * af_key_diff;
        
        const double update_factor = sm * sm_step;
        for (j = jstart; j < jend; j++) {
          new_smm[j] += smm_conv[j] * update_factor;
        }
        
        fid_sm -= sm_step;
      }
      
      nh = new_nh;
      lnh = lnnh;
    }

    /* Copy results back efficiently */
    memcpy(smm + jstart, new_smm + jstart, (jend - jstart) * sizeof(double));
  }

  free(new_smm);
  free(smm_conv);
  free(g_cache);
}