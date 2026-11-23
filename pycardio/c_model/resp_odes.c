#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>

#include "parameters.h"

/* william barnett; March 2018; whbdupree@gmail.com */
/* Robert Ahlroth Capps; April 2018; rocapp@gmail.com */

// random number -0.5, 0.5
double rand_scale(){
  int r = rand();
  double rr = ((double) r / RAND_MAX);
  return 0.5 - rr;
}

// biophysics functions
double ninf(double v, const struct parameters *p) { return( 1/(1+exp((v-p->v12n)/p->kn)) ); }
double mpinf(double v, const struct parameters *p) { return( 1/(1+exp((v-p->v12mp)/p->kmp)) ); }
double hpinf(double v, const struct parameters *p) { return( 1/(1+exp((v-p->v12hp)/p->khp)) ); }
double tauinf(double v, const struct parameters *p) { return( p->tauhp/cosh((v-p->v12hp)/(p->khp)) ); }

double vlow = -50;
double vhigh = -20;
double f(double v) {
  if (v < vlow) { return( 0 ); }
  else if (v > vhigh) { return( 1 ); }
  else { return( (v-vlow)/(vhigh-vlow) ); }
}

double beat(double phi) { return( pow( (1+sin(phi*2*M_PI))/2, 20 )); }

void noisy_step(const double y[],
		struct parameters *p,
		double h,
		double tn,
		double t1){
  double rs = p->rs1*rand_scale();
  double eiAd;

  eiAd = (f(y[1])-p->eiAd)/tad_ei; // early-i iAD activation
  eiAd += rs;

  /* printf("\n\nrand %f dyeiAd %f \n\n",rs,eiAd); */

  while (tn < t1){
    p->eiAd += eiAd * h;
    tn += h;
  }
  
}

int rhs (double t,
	 const double y[],
	 double dy[],
	 struct parameters *p) {

  (void)(t);

  //pre-i/i
  double n0 = ninf(y[0], p);
  dy[0]= -p->gKdr*n0*n0*n0*n0*(y[0]-p->EK)-p->gL*(y[0]-p->EL)-p->gNaP*mpinf(y[0], p)*y[5]*(y[0]-p->ENa);
  dy[0]+= -p->gE*p->d1*(y[0]-p->Ee)-p->gI*(p->b31*f(y[2])+p->b41*f(y[3]))*(y[0]-p->Ei);
  dy[0]/= p->C;

  //early-i
  double n1 = ninf(y[1], p);
  dy[1]= -p->gKdr*n1*n1*n1*n1*(y[1]-p->EK)-p->gL*(y[1]-p->EL)-p->gAD*p->eiAd*(y[1]-p->EK);
  dy[1]+= -p->gE*(p->d2+p->a12*f(y[0]))*(y[1]-p->Ee)-p->gI*(p->b32*f(y[2])+p->b42*f(y[3])+p->hb2*y[8]+p->st2)*(y[1]-p->Ei);
  dy[1]/= p->C;

  //aug-e
  double n2 = ninf(y[2], p);
  dy[2]= -p->gKdr*n2*n2*n2*n2*(y[2]-p->EK)-p->gL*(y[2]-p->EL)-p->gAD*y[6]*(y[2]-p->EK);
  dy[2]+= -p->gE*(p->d3)*(y[2]-p->Ee)-p->gI*(p->b23*f(y[1])+p->b43*f(y[3]))*(y[2]-p->Ei);
  dy[2]/= p->C;

  //post-i
  double n3 = ninf(y[3], p);
  dy[3]= -p->gKdr*n3*n3*n3*n3*(y[3]-p->EK)-p->gL*(y[3]-p->EL)-p->gAD*y[7]*(y[3]-p->EK);
  dy[3]+= -p->gE*(p->d4+p->hb*y[8]+p->st4+p->baro*y[10])*(y[3]-p->Ee)-p->gI*(p->b24*f(y[1])+p->b34*f(y[2]))*(y[3]-p->Ei);
  dy[3]/= p->C;

  //ramp-i
  double n4 = ninf(y[4], p);
  dy[4]= -p->gKdr*n4*n4*n4*n4*(y[4]-p->EK)-p->gL*(y[4]-p->EL);
  dy[4]+= -p->gE*(p->d5+p->a15*f(y[0]))*(y[4]-p->Ee)-p->gI*(p->b25*f(y[1])+p->b35*f(y[2])+p->b45*f(y[3]))*(y[4]-p->Ei);
  dy[4]/= p->C;

  //pre-i/i nap inactivation
  dy[5]= (hpinf(y[0], p)-y[5]) / tauinf(y[0], p);

  // aug-e iAD activation
  dy[6]= (f(y[2])-y[6])/p->tad;

  // post-i iAD activation
  dy[7]= (f(y[3])-y[7])/p->tad;

  dy[8]= ( f(y[4]) - y[8] ) / p->taul; // tidal volume

  // heart rate
  dy[9] = p->hr + p->rsa1*f(y[3]) + p->rsa2*f(y[4]) - p->db*y[11];

  // blood pressure
  dy[10]= ( beat(y[9]) + p->sym*f(y[4]) -p->sb*y[11] - (y[10] - p->po) ) / p->taup;

  // smoothed blood pressure
  dy[11]= (y[10]-y[11])/p->taupp;

  return GSL_SUCCESS;
}

void integrate (int ln, int nd, double *uu, double te, struct parameters * p)
{

  /* printf("\n\n rs1 %f \n\n",p->rs1); */
  
  srand(time(0));
  
  double y[] = { -60, -60, -60, -60, -30, .3, 0, 0, 0, 0, .1, .1 };

  p->eiAd = 0.0; // slow variable init
  
  double dy[nd];
  
  double t = 0.0;
  
  double h = te / (double) ln;

  gsl_odeiv2_system sys;
  sys.function = rhs;
  sys.jacobian = NULL;
  sys.dimension = nd;
  sys.params = p;

  gsl_odeiv2_driver *d =
    gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rk2,
				   1e-3, 1e-5, 1e-7);
  
  int j,l,s;
  for(l=0; l < ln; l++){

    double ti = (double) l;
    double tn = t;
    s = gsl_odeiv2_driver_apply (d, &t, ti, y);
    
    if (s != GSL_SUCCESS)
      {
	printf ("\nerror: driver returned %d\n", s);
	break;
      }

    // integrate noisy variables
    noisy_step(y, p, 1e-3, tn, ti);

    
    int ix_t = l*(nd+1); // index offset
    
    // store firing rates
    for(j=0; j<5; j++){
      uu[ix_t+j] = f(y[j]);
    }

    // inactivation/activation
    uu[ix_t+5] = y[5];
    uu[ix_t+6] = p->eiAd;

    for(j=7;j<nd+1;j++) {
      uu[ix_t+j]=y[j-1]; // everything else
    }

  }

    gsl_odeiv2_driver_free (d);  
  
}
