#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "immintrin.h"
#include "pack.h"
#include "kernel.h"
#include <string.h>
#include "rdtsc.h"

//#define DEBUG 1
//#define KERNEL 1

#define KERNEL_P_SIZE 8
#define KERNEL_D_SIZE 4
#define KERNEL_C_SIZE 4
int psize; // number of points
int dsize; // number of dimensions
int k;     // number of centers
int iterations;

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("Wrong arguments!");
    }

    iterations = atoi(argv[1]);
    k = atoi(argv[2]);
    dsize = atoi(argv[3]);
    psize = atoi(argv[4]);

    double *points;
    double *centers;
    double *packed_points;
    double *packed_centers;
    double *new_centers;
    int *new_centers_size;
    //double *final_labels;

    posix_memalign((void **)&points, 64, dsize * psize * sizeof(double));
    posix_memalign((void **)&centers, 64, k * dsize * sizeof(double));
    posix_memalign((void **)&packed_points, 64, dsize * psize * sizeof(double));
    posix_memalign((void **)&packed_centers, 64, dsize * psize * sizeof(double));
    posix_memalign((void **)&new_centers, 64, k * dsize * sizeof(double));
    posix_memalign((void **)&new_centers_size, 64, k * sizeof(int));
    //posix_memalign((void **)&final_labels, 64, psize * sizeof(double));

    // initialize points and centers
    read_points_and_centers(points, centers, dsize, psize, k);

    // pack the data into panel order
    pack_points(points, packed_points, dsize, psize);

#ifdef DEBUG
    printf("Pack!\n");
    for (int i = 0; i != dsize * psize; ++i)
    {
        printf("%lf, ", packed_points[i]);
        if ((i + 1) % psize == 0)
        {
            printf("\n");
        }
    }
    printf("\n");

    printDist(psize, dsize, k, KERNEL_P_SIZE, centers,points); 

#endif
    memcpy(new_centers, centers, k * dsize * sizeof(double));

    tsc_counter t0, t1;
    long long sum1 = 0;

    int t_num = 1;

// Benchmark the inidividual kernels
// Close it in a thorough run
#ifdef KERNEL
    long long distance_kernel_sum = 0;
    long long distance_kernel_cmp = 0;
    long long distance_kernel_cluster = 0;
    tsc_counter t_dis_0, t_dis_1;
    tsc_counter t_cmp_0, t_cmp_1;
    tsc_counter t_cluster_0, t_cluster_1;
#endif

    RDTSC(t0);
    // kmeans algorithm
    for (int i = 0; i < iterations; i += 1)
    {
        // initialize new centers to 0
        pack_centers(new_centers, packed_centers, dsize, k);
        memset(new_centers, 0, k * dsize * sizeof(double));
        memset(new_centers_size, 0, k * sizeof(int));
            
        for (int p = 0; p < psize; p += KERNEL_P_SIZE)
        {
            // set the proper initialize value
            // 8 registers reserved for intermediate distance result
            __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
            double tmp_cluster_arr_g1[4] = {0.0, 0.0, 0.0, 0.0};
            double tmp_min_dist_arr_g1[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
            double tmp_cluster_arr_g2[4] = {0.0, 0.0, 0.0, 0.0};
            double tmp_min_dist_arr_g2[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
            for (int c = 0; c < k; c += KERNEL_C_SIZE) // 4 centers computed in each kernel
            {
                // 8 registers to store intermediate distances
                ymm0 = _mm256_setzero_pd();
                ymm1 = _mm256_setzero_pd();
                ymm2 = _mm256_setzero_pd();
                ymm3 = _mm256_setzero_pd();
                ymm4 = _mm256_setzero_pd();
                ymm5 = _mm256_setzero_pd();
                ymm6 = _mm256_setzero_pd();
                ymm7 = _mm256_setzero_pd();
                for (int d = 0; d < dsize; d += KERNEL_D_SIZE) // 4 dimension in each kernel
                {
                    // apply the kernel on 8 * 4 points matrix and 4 * 4 on centers matrix
                    #ifdef KERNEL
                    RDTSC(t_dis_0);
                    #endif

                    kernel(KERNEL_D_SIZE, KERNEL_P_SIZE, &ymm0, &ymm1, &ymm2, &ymm3, &ymm4, &ymm5, &ymm6, &ymm7,
                        packed_points + p * dsize + d * KERNEL_P_SIZE, packed_centers + c * dsize + d * KERNEL_C_SIZE);

                    #ifdef KERNEL
                    RDTSC(t_dis_1);
                    distance_kernel_sum += COUNTER_DIFF(t_dis_1, t_dis_0, CYCLES);
                    #endif
                }
                #ifdef KERNEL
                RDTSC(t_cmp_0);
                #endif
                findCluster(&ymm0, &ymm1, &ymm2, &ymm3, &ymm4, &ymm5, &ymm6, &ymm7,
                            tmp_cluster_arr_g1, tmp_min_dist_arr_g1, tmp_cluster_arr_g2, tmp_min_dist_arr_g2, c * 1.0);
                #ifdef KERNEL
                RDTSC(t_cmp_1);
                distance_kernel_cmp += COUNTER_DIFF(t_cmp_1, t_cmp_0, CYCLES);
                #endif
            }

            #ifdef KERNEL
            RDTSC(t_cluster_0);
            #endif

            updateCluster(tmp_cluster_arr_g1, tmp_cluster_arr_g2, points + p * dsize, new_centers, psize, dsize, new_centers_size);

            #ifdef KERNEL
            RDTSC(t_cluster_1);
            distance_kernel_cluster += COUNTER_DIFF(t_cluster_1, t_cluster_0, CYCLES);
            #endif
        }
        
        computeNewCluster(new_centers, new_centers_size, dsize, k);
    }
    
    RDTSC(t1);

    sum1 += (COUNTER_DIFF(t1, t0, CYCLES));

    // print new centers
    for (int m = 0; m < dsize * k; m++) {
        printf("%lf ", new_centers[m]);
        if (m % dsize == dsize - 1) {
            printf("\n");
        }
    }

    printf("Execution cycles: %lld, times: %lf seconds, FLOPS/cycle: %lf\n",
        sum1, (double) sum1 / (double) PROCESSOR_FREQ, 
        (double) (iterations * (psize * k * dsize * 3 + 2 * psize * k + psize * dsize)) / (double) sum1);
    
    #ifdef KERNEL
    printf("FLOPS/cycle of distance kernel: %lf\n",
        (double) (iterations * (psize * k * dsize * 3)) / (double) distance_kernel_sum);
    printf("FLOPS/cycle of compare kernel: %lf\n",
        (double) (iterations * (2 * psize * k)) / (double) distance_kernel_cmp);
    printf("FLOPS/cycle of cluster kernel: %lf\n",
        (double) (iterations * (psize * dsize)) / (double) distance_kernel_cluster);
    #endif

#ifdef DEBUG
    // final result of labels
    for (int i = 0; i < psize; i++)
    {
        printf("%f ", final_labels[i]);
    }
    printf("\n");
#endif
}
