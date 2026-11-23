/* 
Robert Capps May 2018; adapted from CMA-ES
(Hansen 2006)
N. Hansen: The CMA Evolution Strategy: A Comparing Review, StudFuzz 192, 75–102 (2006)*/

#include "cmaes.h"

int calc_lambda(int N){ return 4+floor(3*log(N)); }

int calc_mu(int lambda){ return floor(lambda / 2.0); }

double calc_weights(double *weights, int mu){
  int i;
  double w, w_sum;
  w_sum = 0;
  for (i=0; i < mu; i++){
    w = log(mu+1)-log(i+1);
    weights[i] = w;
    
#ifdef DEBUG
    printf("0 weights[%d] = %f\n",i,w);
#endif
    
    w_sum = w_sum + w;
  }

#ifdef DEBUG
  printf("\n\nw_sum = %f\n\n",w_sum);
#endif
  
  for (i=0; i < mu; i++){
    weights[i] /= w_sum;
    
#ifdef DEBUG
    printf("1 weights[%d] = %f\n",i,weights[i]);
#endif
    
  }
  return w_sum;
}

double calc_mueff(double *weights, double w_sum, int mu){
  double numer, denom;
  numer = pow(w_sum, 2);
  denom = 0;
  int i;
  for(i=0; i < mu; i++) { denom += pow(weights[i], 2); }
  return numer/denom;
}

void setupSelectionParams(InputArgs *a,
			  SelectionParams *s){
  int lambda, mu;
  lambda = calc_lambda(a->N);
  mu = calc_mu(lambda);
  
#ifdef DEBUG
  printf("\nlambda = %d\n",lambda);
  printf("mu = %d\n",mu);
#endif
  
  double weights[mu];
  double w_sum, mueff;
  w_sum = calc_weights(weights, mu);
  mueff = calc_mueff(weights, w_sum, mu);
  
  s->lambda = lambda;
  s->mu = mu;
  s->weights = weights;
  s->mueff = mueff;
}

void setupAdaptationParams(InputArgs *a,
			   SelectionParams *s,
			   AdaptationParams *adp){

  double cc, cs, mucov, ccov, damps;
  cc = 4.0 / (a->N + 4.0);
  cs = (s->mueff + 2.0) / (a->N + s->mueff + 3.0);
  mucov = s->mueff;
  
  double mueff = s->mueff;
  int N = a->N;
  
  ccov = (1.0 / mucov) * 2.0/pow(N + 1.4, 2)
    + (1.0 - 1.0/mucov) * ( ( 2.0*mueff-1 ) / ( pow(N+2, 2) + 2.0*mueff ) );
  
  damps = 1.0 + 2.0*fmax(0.0, sqrt((mueff-1)/(N+1))-1) + cs;

  adp->cc = cc;
  adp->cs = cs;
  adp->mucov = mucov;
  adp->ccov = ccov;
  adp->damps = damps;
}


void setup(int N,
	    double *xmean, double sigma,
	    double stopfitness, int stopeval){
  
  // Input args to struct
  InputArgs a = { N, xmean, sigma,
		  stopfitness, stopeval };

  // Configure selection params
  InputArgs *a_p = &a;
  SelectionParams *s = malloc(sizeof(SelectionParams));
  setupSelectionParams(a_p, s);

  // Configure adaptation params
  AdaptationParams *adp = malloc(sizeof(AdaptationParams));
  setupAdaptationParams(a_p, s, adp);

  // Initialize dynamic (internal) strategy params and constants
  pc = (double *) calloc(N, sizeof(double));
  ps = (double *) calloc(N, sizeof(double));

  double **B = eye(N);
  double **D = eye(N);
  double **BD = (double **) calloc(N, sizeof(double*));

  matmul2D(B, D, BD, N);
  double **BD_t = transpose(BD, N);

  double **C = (double **) calloc(N, sizeof(double*));
  matmul2D(BD, BD_t, C, N);
  
  double eigeneval = 0.0;
  double chiN = pow(N, 0.5) * ( 1 - ( 1 / (4*N) ) + ( 1 / (21*pow(N, 2)) ) );
}


void _mutate(double *xmean, double sigma, double **arz,
	     int k, int N, double **arx){

  /* Compute the mutation (Eq. 29 from Hansen 2006) and assign to arx */

  for (int i=0; i < N; i++){
    double resi = 0;
    for (int j=0; j < N; j++){
      resi += sigma * BD[i][j] * arz[i][k];
    }
    arx[i][k] = xmean[i] + resi;
  }
  
}
