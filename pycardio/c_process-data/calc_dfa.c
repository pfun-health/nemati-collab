/* Detrended Fluctuation Analysis */
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_fit.h>

void calc_dfa(float x_signal[], float y_detrended[],
	       uint32_t length,
	       float F[],
	       uint32_t min_N, uint32_t max_N){

  uint32_t N, i, si, yi, fi;
  double b, m, cov00, cov01, cov11, sumsq;
  float resid, Y, fluct;
  
  N = min_N; // smallest window
  i = 0;
  fi = 0;
  fluct = 0;
  while (N <= max_N){ // for each N
    
    fluct = 0; // fluctuation for the current N
    double x[N]; // The current window
    double y[N];

    for(i=N-min_N; i < length-N; i+=N){ // Within each window of the time series

      /* find the residual between the y_signal and 
	 LLSQ fit for each window (of size N) */
      
      for (si=i; si < N+i; si++) { // Get a window of size N from x and y
	x[si-i] = (double) x_signal[si];
	y[si-i] = (double) y_detrended[si];
      }
    
      gsl_fit_linear(x, 1, y, 1, N, // Linear regression fit (Y = b + m*x), find b and m.
		     &b, &m, &cov00, &cov01,
		     &cov11, &sumsq);

      // Sum the square residuals bt Y,y
      for (yi=0; yi < N; yi++) {
	Y = (float) (b + m*x[yi]); // The least squares fit for each time
	fluct += (float) powf((y[yi] - Y), 2.0); // The residual
      }
      fluct = fluct / (float) N;
      
      i += N; // the next window begins at i+N
    }
    fi = N - min_N;
    F[fi] = sqrtf(fluct); // Store the fluctuation for this N
    
    N++; // N increments by 1 until the max_N is reached
  }
}
