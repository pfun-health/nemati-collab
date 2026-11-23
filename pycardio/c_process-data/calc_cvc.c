#include <stdlib.h>
#include <stdio.h>

#define THRESH 10.0 // We keep resp cycles of size less than or equal to this threshold (seconds)

void calc_cvc(double time[], int beat_ix[], int nr_beats, int resp_pks[], int nr_resp, double beat_latency[]){
  int ti, tj;
  for(ti=0; ti < nr_beats; ti++){
    double beat_time = time[beat_ix[ti]];
    for(tj=0; tj < nr_resp-1; tj++){
      double resp0 = time[resp_pks[tj]];
      double resp1 = time[resp_pks[tj+1]];
      if ((beat_time <= resp1) && (beat_time > resp0) && (resp1 - resp0 <= THRESH)){
	beat_latency[ti] = (beat_time - resp0) / (resp1 - resp0); // latency normalized to the respiratory phase
	break;
      } else {
	beat_latency[ti] = -999.0; // Flag as not used
      }
    }
  }
}

int main() {
  printf("test.");
}
