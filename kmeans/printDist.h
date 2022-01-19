#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void printDist(int psize, int dsize, int csize, int modsize, double *centers, double *points){
    printf("The original Distance\n");
    double distt[psize * csize];
    for (int i = 0; i != psize * csize; ++i){
        distt[i] = 0;
    }
    // centers
    for (int i = 0; i != csize; ++i){
        // Dimensions
        for (int j = 0; j != dsize; ++j){
            // # of points
            for (int l = 0; l != psize; l++){
                double temp = centers[i*dsize+j] - points[l*dsize+j];
                distt[i*psize+l] += temp*temp; 
            }
        }
    }
    for (int i = 0; i != psize * csize; ++i){
        printf("%lf, ", distt[i]);
        if ((i + 1) % modsize == 0){
             printf("\n");
        }
        if ((i + 1) % psize == 0){
             printf("\n");
        }
    }
    printf("\n");
}