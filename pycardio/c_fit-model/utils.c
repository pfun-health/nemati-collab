#include "cmaes.h"

int compare_fitness(const void *p1, const void *p2){ // least->greatest
  return (int) floor((*(struct FitnessTable*)p1).fitness - (*(struct FitnessTable*)p2).fitness);
}


double ** transpose_ns(double **mat, int N, int M){

  /* Transpose non-square matrix 
     
     mat : NxM
     N   : nr of rows of mat (input)
     M   : nr of cols of mat (input)
     
     returns

     T   : MxN
     
  */
  
  double **T = (double **) calloc(M, sizeof(double*));

  for (int i=0; i < M; i++)
    T[i] = (double *) calloc(N, sizeof(double));

  for( int i=0; i < N; i++){
    
    for (int j=0; j < M; j++){

      T[j][i] = mat[i][j];
      
    }
    
  }

  return T;
  
}

double ** transpose(double **mat, int N)
{
  /* Transpose square matrix */

  double **T = (double **) calloc(N, sizeof(double*));
  
  for( int i=0; i < N; i++)
    T[i] = (double *) calloc(N, sizeof(double));
  
  for( int i=0; i < N; i++){
    for( int j=0; j < N; j++) {
      T[j][i] = mat[i][j];
    }
  }
  
#ifdef DEBUG
  printf("Tranpose of the matrix:\n");
  for( int i=0; i < N; i++){
    for( int j=0; j < N; j++){
      printf("%f  ", T[i][j]);
      if(j==N-1)
	printf("\n\n");
    }
  }
#endif
  
  return T;
}

double ** upperTriangle(double **mat, int N, int k){
  /* Return a 2x2 matrix with the same upper triangle
     as the given matrix (mat).
     
     The input (mat) must be square, and the output will have
     the same dimensions as the input.

     The input N is the number of rows of the input matrix (mat).

     The input k is a shift factor (Matlab triu).
   */

  double **R = (double **) calloc(N, sizeof(double*));
  
  for( int i=0; i < N; i++)
    R[i] = (double *) calloc(N, sizeof(double));

  for(int i = 0; i<N; i++){
    for(int j=0; j<N; j++){
      R[i][j] = 0.0;
      if( j >= i+k ){ // if column ix >= row ix + shift
	R[i][j] = mat[i][j];
      }
    }
  }

#ifdef DEBUG
  printf("Upper triangular matrix is: \n");
  for(int i=0; i < N; i++){
    for(int j=0; j<N; j++){
      printf("%f  ",R[i][j]);
    }
    printf("\n");
  }
#endif
  
}

void eigenspace(double **mat, double *evals, double **evecs, int N){
  /* Find the eigendecomposition of a square matrix */
  
  gsl_matrix_view m = gsl_matrix_view_array(mat, N, N);
  gsl_vector *eval_v = gsl_vector_alloc(N);
  gsl_matrix *evec_v = gsl_matrix_alloc(N, N);
  
  gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(N);
  
  gsl_eigen_symmv (&m.matrix, eval_v, evec_v, w);

  gsl_eigen_symmv_free (w);

  gsl_eigen_symmv_sort (eval_v, evec_v, 
                        GSL_EIGEN_SORT_ABS_ASC);

  for(int i=0; i < N; i++){
    
    evals[i] = gsl_vector_get(eval_v, i);
    
    for(int j=0; j < N; j++){

      evecs[i][j] = gsl_matrix_get(evec_v, i, j);
      
    }
  }
  
}

double ** eye(int N){
  /* Creates a square NxN identity matrix */
  
  double **matrix = (double **) calloc(N, sizeof(double*));

  for ( int row=0; row < N; row++) {
    matrix[row] = (double *) calloc(N, sizeof(double));
    matrix[row][row] = 1.0;
  }
  
#ifdef DEBUG
  printf("Matrix is: \n");
  for ( int row = 0;row < N;row++) {
    for ( int column = 0;column < N;column++) {
      printf("%f ", matrix[row][column]);
    }
    printf("\n");
  }
#endif

  return matrix;

}

void matmul1D2D(double *M1, double **M2, double *result, int N){
  /* in-place matrix multiplication for a 1D and 2D square matrix */
  int i,j;

  double sum = 0.0;

  for(i=0; i<N; i++){

    for(j=0; j<N; j++){

      sum = sum + M1[i]*M2[i][j];
      
    }
    
    result[i] = sum;
    sum = 0.0;
    
  }
  
}

void matmul2D_ns(double **M12, double **M22,
		 double **R,
		 int N, int M){

  /* in-place matrix multiplication for 2D matrices (one is square)
     M12 (NxM)
     M22 (NxN)
     R   (NxM)
  */

  int i,j,k;

  for (i=0; i<N; i++){
    
    R[i] = (double *) calloc(M, sizeof(double));

    for (j=0; j<M; j++){
      
      R[i][j] = 0;

      for(k = 0; k<N; k++){
	
	R[i][j] += M12[i][j]*M22[i][k];
	
      }
     
    }
  }

#ifdef DEBUG
  printf("\nResult of Matrix Multiplication (ns):\n");
  for (i=0; i<N; i++){
    for (j=0; j<M; j++){
      printf("%f  ", R[i][j]);
    }
    printf("\n");
  }
#endif

}

void matmul2D_nonsq(double **M1, double **M2, double **R,
		    int N, int M, int P, int Q){

  /* 2d matrix multiplication where neither matrix is assumed
     to be square.
     
     M1 : (N x M)
     M2 : (P x Q)
     R  : (N x Q)
  */

  int i,j,k;

  if (M != P) {
    printf("M != P. Cannot multiply these matrices.\n");
    return;
  }

  for (i=0; i<N; i++){
    
    R[i] = (double *) calloc(Q, sizeof(double));

    for (j=0; j<Q; j++){

      R[i][j] = 0.0;

      for(k = 0; k<P; k++){
	
	R[i][j] += M1[i][k]*M2[k][j];
	
      }
     
    }
  }

#ifdef DEBUG
  printf("\nResult of Matrix Multiplication (nonsq):\n");
  for (i=0; i<N; i++){
    for (j=0; j<Q; j++){
      printf("%f  ", R[i][j]);
    }
    printf("\n");
  }
#endif
  
  
}


void matmul2D(double **B, double **D, double **R, int N){
  /* in-place matrix multiplication for 2D square matrices */

  int i,j,k;

  for (i=0; i<N; i++){
    R[i] = (double *) calloc(N, sizeof(double));

    for (j=0; j<N; j++){
      R[i][j] = 0;

      for(k = 0; k<N; k++){
	R[i][j] += B[i][k]*D[k][j];
      }
      
    }
  }

#ifdef DEBUG
  printf("Result of Matrix Multiplication:\n");
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      printf("%f  ", R[i][j]);
    }
    printf("\n");
  }
#endif
}

void normal(double * randn, int N){
  /* Generate a normally distributed vector of size N,
     using the box-muller transform */
  double U1, U2;
  for( int i=0; i < N-1; i+=2){
    U1 = (double) rand() / (double) RAND_MAX;
    U2 = (double) rand() / (double) RAND_MAX;
    randn[i] = sqrt(-2.0*log(U1))*cos(2.0*M_PI*U2);
    randn[i+1] = sqrt(-2.0*log(U1))*sin(2.0*M_PI*U2);
  }
}


void freeGlobalMem(int N) {
  
  /* Free the global pointers (cmaes.h) */

  printf("\n\nFreeing global pointers (CMAES)...\n");

  free(arx);
  free(arz);
  free(pc);
  free(ps);
  free(s);
  free(adp);
  free(B);
  free(D);
  free(BD);
  free(BD_t);

  printf("\nFreed all global pointers (CMAES).\n\n");
  
}
