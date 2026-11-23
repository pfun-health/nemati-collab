#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct DistTable {
    int ix1;
    int ix2;
    double dist;
};

void pairwise_dists(double x1[], int N1, double x2[], int N2, DistTable dists[]);
int compare_dists(const void *p1, const void *p2);
void downsample(double x1[], int N1,
                double x2[], int N2,
                int N3, double xout[N3][3]);


void pairwise_dists(double x1[], int N1, double x2[], int N2,
                    DistTable dists[]){
    int i,j,k;
    k = 0;
    double dist;
    for(i=0; i<N1; i++){
        for(j=0; j<N2; j++){
            dists[k].ix1 = i;
            dists[k].ix2 = j;
            dist = pow(x1[i]-x2[j], 2.0);
            dists[k].dist = dist;
// #ifdef DEBUG
//             printf("%d : dist: %.3f\n",k,dist);
// #endif
            k++;
        }
    }
}


int compare_dists(const void *p1, const void *p2){ // sorted least->greatest
  return (int) floor( (*(struct DistTable*)p1).dist - (*(struct DistTable*)p2).dist);
}

void downsample(double x1[], int N1,
                double x2[], int N2,
                int N3, double xout[N3][3]){

    struct DistTable dt[N3];

    pairwise_dists(x1, N1, x2, N2, dt); // find pairwise dists store in DT

    qsort((void*)dt, N3, sizeof(struct DistTable), compare_dists); // sort least->greatest

    int ik;
    for(ik=0; ik<N3; ik++){
        xout[ik][0] = (double) dt[ik].ix1;
        xout[ik][1] = (double) dt[ik].ix2;
        xout[ik][2] = (double) dt[ik].dist;

#ifdef DEBUG
        printf("(x1, x2): [ %.2f , %.2f ] ; dist : %.2f\n",
            x1[(int)xout[ik][0]],x2[(int)xout[ik][1]],xout[ik][2]);
#endif

    }

}

int main(void){
    double x1[] = {1., 2., 3., 4., 5.};
    double x2[] = {0., 2., 6., 7., 150., 199.};
    int N1 = 5;
    int N2 = 6;
    int N3 = (int) N1*N2;
    double xout[N3][3];
    downsample(x1, N1, x2, N2, N3, xout);
}
