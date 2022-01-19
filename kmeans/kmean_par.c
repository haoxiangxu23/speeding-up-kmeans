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

int runs = 1000;

int psize; // number of points
int dsize; // number of dimensions
int k;     // number of centers
int iterations;
int t_num;

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        printf("Wrong arguments!");
    }

    iterations = atoi(argv[1]);
    k = atoi(argv[2]);
    dsize = atoi(argv[3]);
    psize = atoi(argv[4]);
    t_num = atoi(argv[5]);

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

    memcpy(new_centers, centers, k * dsize * sizeof(double));

    unsigned long long t0, t1;
    unsigned long long sum1 = 0;
    
    // kmeans algorithm
    for (int r = 0; r < runs; ++r) {
        t0 = rdtsc();
        for (int i = 0; i < iterations; ++i)
        {
            // initialize new centers to 0
            pack_centers(new_centers, packed_centers, dsize, k);
            memset(new_centers, 0, k * dsize * sizeof(double));
            memset(new_centers_size, 0, k * sizeof(int));
            int p_loop = psize / KERNEL_P_SIZE;

            #pragma omp parallel for num_threads(t_num)
            for (int p_i = 0; p_i < p_loop; ++p_i)
            {
                // set the proper initialize value
                // 8 registers reserved for intermediate distance result
                int p = p_i * KERNEL_P_SIZE;
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
                        kernel(KERNEL_D_SIZE, KERNEL_P_SIZE, &ymm0, &ymm1, &ymm2, &ymm3, &ymm4, &ymm5, &ymm6, &ymm7,
                            packed_points + p * dsize + d * KERNEL_P_SIZE, packed_centers + c * dsize + d * KERNEL_C_SIZE);
                    }
                    findCluster(&ymm0, &ymm1, &ymm2, &ymm3, &ymm4, &ymm5, &ymm6, &ymm7,
                             tmp_cluster_arr_g1, tmp_min_dist_arr_g1, tmp_cluster_arr_g2, tmp_min_dist_arr_g2, c * 1.0);
                }
                // update the cluster
                #pragma omp critical
                {
                    updateCluster(tmp_cluster_arr_g1, tmp_cluster_arr_g2, points + p * dsize, new_centers, psize, dsize, new_centers_size);
                }
            }
            // Compute the new centers
            computeNewCluster(new_centers, new_centers_size, dsize, k);
        }
        t1 = rdtsc();
        sum1 += (t1 - t0);
    }

    // print new centers
    for (int m = 0; m < dsize * k; m++) {
        printf("%lf ", new_centers[m]);
        if (m % dsize == dsize - 1) {
            printf("\n");
        }
    }

    printf("Execution cycles: %lld, times: %lf seconds, FLOPS/cycle: %lf\n",
        sum1, (double) sum1 / (double) PROCESSOR_FREQ, 
        (double) (iterations * (psize * k * dsize * 3 + 2 * psize * k + psize * dsize)) / ((double) (sum1 / runs)));
}
