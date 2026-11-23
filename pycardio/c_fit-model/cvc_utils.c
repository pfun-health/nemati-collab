#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void pairwise_dists(double x1[], int N1, double x2[], int N2,
                    struct DistTable dists){
    int i,j,k;
    k = 0;
    for(i=0; i<N1; i++){
        for(j=0; j<N2; j++){
            dists[k].ix = k
            dists[k].ix1 = i
            dists[k].ix2 = j
            dists[k].dist = pow(x1[i]-x2[j], 2);
            k++;
        }
    }
}

struct DistTable {
    int ix;
    int ix1;
    int ix2;
    double dist;
};

int compare_dists(const void *p1, const void *p2){ // sorted least->greatest
  return (int) floor( (*(struct DistTable*)p2).dist - (*(struct DistTable*)p1).dist);
}

void downsample(double x1[], int N1,
                double x2[], int N2,
                int N3, double xout[N3][4]){

    struct DistTable dt[N3];

    pairwise_dists(x1, N1, x2, N2, dt); // find pairwise dists store in DT

    qsort(dt, N3, sizeof(struct DistTable), compare_dists); // sort least->greatest

    int shift;
    shift = N2 > N1 ? N1 : N2;
    int Nk = N3 - shift;
    int ik = 0;
    while (ik < N3) {
        xout[ik][0] = (double) dt[Nk+ik].ix;
        xout[ik][1] = (double) dt[Nk+ik].ix1;
        xout[ik][2] = (double) dt[Nk+ik].ix2;
        xout[ik][3] = (double) dt[Nk+ik].dist;

        ik++;
    }

}