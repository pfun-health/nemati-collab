/* 
Robert Capps May 2018; adapted from CMA-ES
(Hansen 2006)
N. Hansen: The CMA Evolution Strategy: A Comparing Review, StudFuzz 192, 75–102 (2006)*/

#include "cmaes.h"

void find_means(int N, int mu,
		double **arx, double **arz,
		double *weights,
		struct FitnessTable *ft){

  double xsum = 0.0;
  double zsum = 0.0;
  
  for (int i=0; i < N; i++){
    
    for (int j=0; j < mu; j++){
      
      // find the mean of the top *mu* mutants in arx (recombination)
      xsum = xsum + arx[i][ft[j].ix]*weights[j];
      
      // find the mean of the top *mu* random samples in arz ( == sigmaˆ-1*Dˆ-1*B’*(xmean-xold) )
      zsum = zsum + arz[i][ft[j].ix]*weights[j];
      
    }
    
    xmean_p[i] = xsum;
    zmean_p[i] = zsum;
    xsum = 0.0;
    zsum = 0.0;
    
  }

#ifdef DEBUG2
  printf("\n\nxmean (updated):  ");
  for (int i=0; i<N; i++){
    printf("%f  ",xmean_p[i]);
  }
  printf("\n\nzmean (updated):  ");
  for (int i=0; i<N; i++){
    printf("%f  ",zmean_p[i]);
  }
  printf("\n");
#endif
  
}

void update_path(double *path,
		 const char path_id,
		 int N,
		 double counteval){

  /* Update the paths: ps, pc
     as in Eq. 31, 33 */

  double mueff = s->mueff;
  double lambda = (double) s->lambda;
  double cc;
  double cs;

  if (path_id == "s"){ // path = ps
    cs = adp->cs;
  }
  else if (path_id == "c"){ // path = pc
    cc = adp->cc;
    
    double ps_norm = 0.0;
    for(int i=0; i < N; i++){ // euclidean norm of ps
      ps_norm += pow(ps[i], 2.0);
    }
    
    double hsig = (double) (chiN*ps_norm) /
      sqrt( 1.0 - pow( (1.0 - cs), 2.0*counteval/lambda ) ) < (1.5+(1.0/(N-0.5)));

#ifdef DEBUG2
    printf("\n\nps_norm: %f\n",ps_norm);
    printf("\nhsig: %f\n",hsig);
#endif
    
  }

  int i,j;
  double psum = 0.0;

  for(i=0; i < N; i++){

    if (path_id == "s") { path[i] = (1.0 - cs)*path[i]; }
    else if (path_id == "c") { path[i] = (1.0 - cc)*path[i]; }

    for(j=0; j < N; j++){
      
      if (path_id == "s")
	psum = psum + sqrt( cs*(2.0 - cs)*mueff ) * ( B[i][j]*zmean_p[i] );
      else if (path_id == "c")
	psum = psum + hsig * sqrt( cc*(2.0 - cc)*mueff ) * ( BD[i][j]*zmean_p[i] );
      
    }

    path[i] = path[i] + psum;
    psum = 0.0;
    
  }

#ifdef DEBUG2
  printf("\n\np%s:  ",path_id);
  for (i=0; i<N; i++){
    printf("%f  ",path[i]);
  }
#endif

}

void update_cov(struct FitnessTable *ft,
		int N,
		int mu){

  /* Update the covariance matrix C */

  double ccov = adp->ccov;
  double mucov = adp->mucov;
  double *weights = s->weights;

  // product of pc, pc'
  double pc_dot = 0;
  for(int pi=0; pi < N; pi++){
    pc_dot += pc[pi]*pc[pi];
  }

  // save the top mu samples (using ft)
  double **arz_top = (double **) calloc(N, sizeof(double *));
  for(int azi=0; azi < N; azi++){
    arz_top[azi] = (double *) calloc(M, sizeof(double));
    for(int azj=0; azj < mu; azj++){
      arz_top[azi][azj] = arz[azi][ft[azj].ix];
    }
  }
  
  // B * D * arz(:,arindex(1:mu)) (N x mu)
  double **BDarz = (double **) calloc(N, sizeof(double *));
  matmul2D_ns(arz_top, BD, BDarz, N, mu);
  
  // Create diag(weights) (mu x mu)
  double **dweights = (double **) calloc(mu, sizeof(double *));
  for(int wi=0; wi < mu; wi++){
    dweights[wi] = (double *) calloc(mu, sizeof(double));
    for(int wj=0; wj < mu; wj++){
      dweights[wi][wj] = 0.0;
      if(wi==wj)
	dweights[wi][wj] = weights[wi];
    }
  }
  
  // BDarz * diag(weights) (N x mu)
  double **BDarz_dweights = (double **) calloc(mu, sizeof(double *));
  matmul2D_ns(BDarz, dweights, BDarz_dweights, N, mu);

  // Transpose of nonsquare matrix BDarz (mu x N)
  double **BDarz_t = transpose_ns(BDarz, N, mu);

  // BDarz_dweights * BDarz_t [ (N x mu) (mu x N) ] = (N x N)
  double **rank_mu_prod = (double **) calloc(N, sizeof(double *));
  matmul2d_nonsq(BDarz_dweights, BDarz_t, rank_mu_prod, N, mu, N, mu);
  
  int i,j;
  
  for(i=0; i<N; i++){
    
    for(j=0; j<N; j++){

      C[i][j] = (1.0 - ccov)*C[i][j] + ccov * (1/mucov) * // old matrix
	( pc_dot + (1.0 - hsig) * cc*(2.0-cc) * C[i][j] ) + // rank one update
	ccov * ( 1.0 - (1.0/mucov) ) * rank_mu_prod[i][j]; // rank mu update
      
    }
    
  }
  
  free(arz_top);
  free(BDarz);
  free(dweights);
  free(BDarz_dweights);
  free(BDarz_t);
  free(rank_mu_prod);
  
}

void metropolis_criteria(struct FitnessTable *ft,
			 struct Result *rt,
			 int mu, int N){

  /* Apply the metropolis hastings criteria to the current samples
     
     Store the accepted sample information in the Result structure.
  */

  if (rt->nr_accepted == 0){ // if this is the first iteration, accept the top mu
    for(int j=0; j < mu; j++){
      rt->costs[];
      for(int i=0; i< N; i++){
      }
    }
  }
  
  double p0, p1, alpha, U;
  for (int j=0; j < mu; j++){
    p0 = rt->costs[rt->nr_accepted]
    p1 = exp(-0.5*ft[j].fitness);
    U = (double) rand() / (double) RAND_MAX;
    for (int i=0; i < N; i++){
      
    }
  }
      /* arx[i][ft[j].ix]; */
  
  
}

void generation_loop(PyObject *self, struct Result *result){

  srand((int)time(NULL)); // seed random number generator
  
  int N = a_p->N; // Dimensions of sample vect
  int lambda = s->lambda; // Number of offspring

  double **arz = (double **) calloc(N, sizeof(double *)); // random mutation vect
  double **arx = (double **) calloc(N, sizeof(double *)); // sample vect
  for (int i=0; i < N; i++){
    arz[i] = (double *) calloc(lambda, sizeof(double));
    arx[i] = (double *) calloc(lambda, sizeof(double));
  }

  double *evals = (double *) calloc(N, sizeof(double)); // eigenvalues
  double **evecs = (double **) calloc(N, sizeof(double *)); // eigenvectors
  for (int i=0; i < N; i++)
    evecs[i] = (double *) calloc(N, sizeof(double));
  
  double *randn = (double *) calloc(N, sizeof(double)); // random normal vect

  struct FitnessTable ft[lambda]; // fitness table
  
  double sigma = a_p->sigma; // step size

  double cs = adp->cs;
  double cc = adp->cc;
  double damps = adp->damps;
  
  int counteval = 0; // count total evaluations
  eigeneval = 0; // evals since eigendecomposition
  
  int stopeval = a_p->stopeval; // number of evals after which, we stop
  double stopfitness = a_p->stopfitness; // fitness tolerance

  int mu = s->mu; // choose this many for weighted xmean
  double mueff = s->mueff;
  double *weights = s->weights; // weights for top mu mean

  double cost, ps_norm;

  double zmean[N]; // weighted result of top mu samples
  zmean_p = zmean;

  // Allocate result members
  result->nr_accepted = 0;
  double *result->x = (double *) calloc(N, sizeof(double));
  double *result->costs = (double *) calloc(mu, sizeof(double));
  double **result->samples = (double **) calloc(N, sizeof(double*));
  double **result->C = (double **) calloc(N, sizeof(double*));
  for(int ri=0; ri < mu; ri++)
    result->samples[ri] = (double *) calloc(mu, sizeof(double));
  for(int rj=0; rj < N; rj++)
    result->C[rj] = (double *) calloc(N, sizeof(double));
  
  // Generation loop
  while (counteval < stopeval) {

      // Generate and evaluate [lambda] offspring
      for (int k=0; k < lambda; k++){

	normal(randn, N); // generate normally distributed vector of size N
	for (int i=0; i < N; i++){
	  arz[i][k] = randn[i]; // assign arz
	  _mutate(xmean_p, sigma, arz, k, N, arx); // perform and assign mutation to arx
	}
	
	cost = c_generateCost(self); // evaluate objective for the offspring

	ft[k].ix = k; // store fitness and original indices
	ft[k].fitness = cost;
	
	counteval++; // increment total evals
      }
      
      // sort fitness table in place (minimization)
      qsort(ft, lambda, sizeof(struct FitnessTable), compare_fitness);

      // find the means of the top mu samples
      find_means(N, mu, arx, arz, weights, ft);
      
      // Cumulation : update the evolution paths
      update_path(ps, "s", N, (double) counteval);
      update_path(pc, "c", N, (double) counteval);      

      // Update covariance matrix C (Eq. 34)
      update_cov(ft, N, mu);

      // Adapt step-size sigma
      ps_norm = 0.0;
      for(int i=0; i < N; i++){ // euclidean norm of ps
	       ps_norm += pow(ps[i], 2.0);
      }
      sigma = sigma * exp((cs/damps)*(ps_norm/chiN - 1));

      // Update B and D from C (eigendecomposition)
      if ( counteval - eigeneval > ((lambda/ccov) / ((double ) N / 10.0)) ){
	eigeneval = counteval;
	double **C_triu = upperTriangle(C, N, 0);
	double **C_triu1 = upperTriangle(C, N, 1);
	double **C_triu1_t = transpose(C_trui1, N); // transpose C_triu1
	for(int i=0; i < N; i++){
	  for(int j=0; j < N; j++){ // make sure C is symmetric
	    C[i][j] = C_triu[i][j] + C_triu1_t; 
	  }
	}
	eigenspace(C, evals, evecs, N); // eigendecomposition
	for (int i=0; i < N; i++){
	  for (int j=0; j < N; j++){ // update B and D
	    B[i][j] = evecs[i][j];
	    D[i][j] = 0.0;
	    if(i == j){
	      D[i][j] = sqrt(evals[i]);
	    }
	  }
	}
      }

      // if fitness is good enough, break!
      if ( ft[0].fitness <= stopfitness ) {
	break;
      }

      // escape flat fitness
      if (ft[0].fitness == ft[(int) ceil(0.7 * (float) lambda)].fitness){
	sigma = sigma * exp( 0.2 + ( cs/damps) );
      }

      printf("current best sample : [ ");
      for (int i=0; i < N; i++){
	printf("  %.2f", arx[i][ft[0].ix]);
      }
      printf(" ]\n");
      
      printf("cost: %.2f\n\n",ft[0].fitness);
      
			  
    }
  
  // free intermediate arrays
  free(randn); // free intermediate random vect
  free(C_triu); // free upper triangle of C
  free(C_triu1); // free offset triangle of C
  free(C_triu1_t); // free transpose of C_triu1
  free(evals);
  free(evecs);
  
}

int main(){
  double xmean[] = {1., 2., 3.};
  xmean_p = xmean;
  int N = 3;
  double sigma = 1.0;
  double stopfitness = 1e-7;
  int stopeval = 10;

  setup(N, xmean_p, sigma, stopfitness, stopeval);

  freeGlobalMem(N);

  return 0;
}
