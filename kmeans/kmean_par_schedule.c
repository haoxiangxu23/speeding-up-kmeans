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

#define MAX_CENTER_NUM 64
#define MAX_LOCAL_CLUSTER_SIZE 32768
#define MAX_THREAD 40


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

    if (k > MAX_CENTER_NUM) {
        printf("Number of centers should be less than %d\n", MAX_CENTER_NUM);
        return 0;
    }

    if (k * dsize > MAX_LOCAL_CLUSTER_SIZE) {
        printf("Total size of centers matrix should be less than %d\n", MAX_LOCAL_CLUSTER_SIZE);
        return 0;
    }

    if (t_num > MAX_THREAD) {
        printf("Total number of threads should be less than %d\n", MAX_THREAD);
        return 0;
    }

    double *points;
    double *centers;
    double *packed_points;
    double *packed_centers;
    double *new_centers;
    int *new_centers_size;
    double *thread_local_centers;
    int *thread_local_sizes;

    posix_memalign((void **)&points, 64, dsize * psize * sizeof(double));
    posix_memalign((void **)&centers, 64, k * dsize * sizeof(double));
    posix_memalign((void **)&packed_points, 64, dsize * psize * sizeof(double));
    posix_memalign((void **)&packed_centers, 64, dsize * psize * sizeof(double));
    posix_memalign((void **)&new_centers, 64, k * dsize * sizeof(double));
    posix_memalign((void **)&new_centers_size, 64, k * sizeof(int));
    posix_memalign((void **)&thread_local_centers, 64, t_num * k * dsize * sizeof(double));
    posix_memalign((void **)&thread_local_sizes, 64, t_num * k * sizeof(int));

    // initialize points and centers
    read_points_and_centers(points, centers, dsize, psize, k);

    // pack the data into panel order
    pack_points(points, packed_points, dsize, psize);

    memcpy(new_centers, centers, k * dsize * sizeof(double));

    tsc_counter t0, t1;
    long long sum1 = 0;

    #ifdef CORRECTNESS_CHECK
    runs = 1;
    #endif

    RDTSC(t0);
    // kmeans algorithm
    for (int r = 0; r < runs; ++r) {
        for (int i = 0; i < iterations; ++i)
        {
            // initialize new centers to 0
            pack_centers(new_centers, packed_centers, dsize, k);
            memset(new_centers, 0, k * dsize * sizeof(double));
            memset(new_centers_size, 0, k * sizeof(int));
            memset(thread_local_centers, 0, t_num * k * dsize * sizeof(double));
            memset(thread_local_sizes, 0, t_num * k * sizeof(int));
            int p_loop = psize / KERNEL_P_SIZE;

            #pragma omp parallel for num_threads(t_num) schedule(static, p_loop / t_num)
            for (int p_i = 0; p_i < p_loop; ++p_i)
            {   
                // local copy of centers
                int tid = omp_get_thread_num();
                
                double* new_centers_local = &thread_local_centers[tid * k * dsize];
                int* new_centers_size_local = &thread_local_sizes[tid * k];
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
                // each thread only update its own local version to avoid cache ping pong
                updateCluster(tmp_cluster_arr_g1, tmp_cluster_arr_g2, points + p * dsize,
                    new_centers_local, psize, dsize, new_centers_size_local);
            }

            // merge the result
            // The loop can be reversed and paralleled with OpenMP, but the center size is too small
            // to make use of the parallelism
            for (int t_i = 0; t_i < t_num; ++t_i) 
            {
                for (int c_i = 0; c_i < k; ++c_i)
                {
                    for (int d_i = 0; d_i < dsize; ++d_i)
                    {
                        new_centers[c_i * dsize + d_i] += thread_local_centers[t_i * k * dsize + c_i * dsize + d_i];
                    }
                    new_centers_size[c_i] += thread_local_sizes[t_i * k + c_i];
                }
            }
            // Compute the new centers
            computeNewCluster(new_centers, new_centers_size, dsize, k);
        }
    }
    RDTSC(t1);

    sum1 += (COUNTER_DIFF(t1, t0, CYCLES));

    // print new centers
    #ifdef CORRECTNESS_CHECK
    for (int m = 0; m < dsize * k; m++) {
        printf("%lf ", new_centers[m]);
        if (m % dsize == dsize - 1) {
            printf("\n");
        }
    }
    #endif

    printf("Execution cycles: %lld, times: %lf seconds, FLOPS/cycle: %lf\n",
        sum1, (double) sum1 / (double) PROCESSOR_FREQ, 
        (double) (iterations * ((psize * k * dsize * 3 + 2 * psize * k + psize * dsize) / ((double) (sum1 / runs)))));
}
