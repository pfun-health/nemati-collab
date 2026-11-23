#include <stdlib.h>
#include <stdio.h>

#define THRESH 4.0 // We keep RRIs less than or equal to this threshold

void calc_rsa(double time[],
	      int beat_ix[], int nr_beats,
	      int resp_pks[], int nr_resp,
	      double rri_phase[][3]){
  int ti, tj;
  for(ti=0; ti < nr_beats-1; ti++){
    double beat_time0 = time[beat_ix[ti]];
    double beat_time1 = time[beat_ix[ti+1]];
    double rri = beat_time1 - beat_time0;
    if ( rri <= THRESH ){
      double middle = (beat_time1 + beat_time0) / 2;
      for(tj=0; tj < nr_resp-1; tj++){
	  double resp0 = time[resp_pks[tj]];
	  double resp1 = time[resp_pks[tj+1]];
	  if ((middle < resp1) && (middle >= resp0)){
	    double cycle_time = resp1 - resp0;
	    rri_phase[ti][0] = (middle-resp0)/cycle_time;
	    rri_phase[ti][1] = rri;
	    rri_phase[ti][2] = middle;
	    break;
	  } else {
	    rri_phase[ti][0] = -999;
	    rri_phase[ti][1] = -999;
	    rri_phase[ti][2] = -999;
	  }
      }
    } else {
      rri_phase[ti][0] = -999;
      rri_phase[ti][1] = -999;
      rri_phase[ti][2] = -999;
    }
  }
}

int main() {
  printf("test.");
}
