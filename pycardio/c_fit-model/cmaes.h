/* Robert Capps May 2018; adapted from CMA-ES (Hansen 2006) */
#ifndef CMAES_H
#define CMAES_H

#include <Python.h>
#include <assert.h>
#include <math.h>
#include <gsl/gsl_eigen.h>

typedef struct { // user-defined parameters (need to be set)
  int N; // number of objective variables/problem dimension
  double *xmean; // objective variables initial point
  double sigma; // coordinate-wise standard deviation (step size)
  double stopfitness; // stop after stopeval number of function evals
  int stopeval; // stop if fitness < stopfitness (minimization)
} InputArgs;

typedef struct { // strategy selection parameters
  int lambda; // population size, offspring number
  int mu; // number of parents/points for recombination
  double *weights; // muXone array for weighted recombination
  double mueff; // variance vs effective size of mu
} SelectionParams;

typedef struct { // strategy adaptation parameters
  double cc; // time constant for accumulation (covariance matrix)
  double cs; // time const for accumulation (sigma control)
  double mucov; // size of mu used for calculating sigma control
  double ccov; // learning rate for covariance matrix
  double damps; // damping for sigma
} AdaptationParams;

/* struct container for objective fitness information */
struct FitnessTable {
  int ix;
  double fitness;
};

/* struct container for final optimization results */
struct Result {
  double *x; // the optimal sample
  double **C; // the final covariance estimate
  double **samples; // accepted samples
  double *costs; // fitness of each accepted sample
  int nr_accepted; // number of accepted samples
};


// global variables
InputArgs *a_p;
SelectionParams *s;
AdaptationParams *adp;
double *pc, *ps; // evolution paths for C and sigma
double **arx, **arz; // the current offspring and mutants, respectively
double **B; // the coordinate system definition matrix
double **D; // diagonal matrix, scaling definition
double **C; // covariance matrix
double **BD; // matrix multiplication of B, D
double **BD_t; // BD transposed
double *xmean_p; // Pointer to xmean vector
double *zmean_p; // Pointer to zmean vector
int eigeneval; // B and D updated at counteval == 0
double chiN; // expectation of ||N(0,I)|| == norm(randn(N,1))
double hsig;

// configuration/computation functions
int calc_lambda(int); // calculate population size
int calc_mu(int); // calculate number of points for recombination
double calc_weights(double *, int);
double calc_mueff(double *, double, int);
void setupSelectionParams(InputArgs *, SelectionParams *);
void setupAdaptationParams(InputArgs *, SelectionParams *, AdaptationParams *);
void setup(int, double *, double, double, int);
void _mutate(double *, double, double **, int, int, double **);

// matrix/util functions
void freeGlobalMem(int);
double ** transpose(double **, int);
double ** transpose_ns(double **, int, int);
double ** upperTriangle(double **, int, int);
void eigenspace(double **, double *, double **, int);
double ** eye(int);
void matmul1D2D(double *, double **, double *, int);
void matmul2D(double **, double **, double **, int);
void matmul2D_ns(double **, double **, double **, int, int);
void matmul2D_nonsq(double **, double **, double **, int, int, int, int);
void normal(double *, int);
int compare_fitness(const void *, const void *);

// Python extension functions
double c_generateCost(PyObject *);

#endif
