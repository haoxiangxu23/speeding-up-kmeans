#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define KERNEL_P_SIZE 8
#define KERNEL_C_SIZE 4

void transpose(double *points, double *packed_points, int d, int p) {
    for (int i = 0; i < d; i += 4) {
        for (int j = 0; j < p; j += 4) {
            __m256d a1a2a3a4 = _mm256_load_pd(points + j * d + i); 
            __m256d b1b2b3b4 = _mm256_load_pd(points + (j + 1) * d + i);
            __m256d c1c2c3c4 = _mm256_load_pd(points + (j + 2) * d + i);
            __m256d d1d2d3d4 = _mm256_load_pd(points + (j + 3) * d + i);

            __m256d a1b1a3b3 = _mm256_shuffle_pd(a1a2a3a4, b1b2b3b4, 0 | (0 << 1) | (0 << 2) | (0 << 3));
            __m256d a2b2a4b4 = _mm256_shuffle_pd(a1a2a3a4, b1b2b3b4, 1 | (1 << 1) | (1 << 2) | (1 << 3));
            __m256d c1d1c3d3 = _mm256_shuffle_pd(c1c2c3c4, d1d2d3d4, 0 | (0 << 1) | (0 << 2) | (0 << 3));
            __m256d c2d2c4d4 = _mm256_shuffle_pd(c1c2c3c4, d1d2d3d4, 1 | (1 << 1) | (1 << 2) | (1 << 3));
            __m256d ref = _mm256_permute2f128_pd(a1b1a3b3, c1d1c3d3, 0 | (2 << 4));
            _mm256_store_pd(packed_points + i * p + j, ref); 
            ref = _mm256_permute2f128_pd(a2b2a4b4, c2d2c4d4, 0 | (2 << 4));
            _mm256_store_pd(packed_points + (i + 1) * p + j, ref);
            ref = _mm256_permute2f128_pd(a1b1a3b3, c1d1c3d3, 1 | (3 << 4));
            _mm256_store_pd(packed_points + (i + 2) * p + j, ref);
            ref = _mm256_permute2f128_pd(a2b2a4b4, c2d2c4d4, 1 | (3 << 4));
            _mm256_store_pd(packed_points + (i + 3) * p + j, ref);
        }
    }
}

void pack_points(double *points, double * packed_points, int d, int p) {
    int offset = 0;
    for (int i = 0; i < p; i += KERNEL_P_SIZE) {
        transpose(points + offset, packed_points + offset, d, KERNEL_P_SIZE);
        offset += KERNEL_P_SIZE * d;
    }
}

void pack_centers(double *centers, double *packed_centers, int d, int k) {
    int offset = 0;
    for (int i = 0; i < k; i += KERNEL_C_SIZE) {
        transpose(centers + offset, packed_centers + offset, d, KERNEL_C_SIZE);
        offset += KERNEL_C_SIZE * d;
    }
}

// read points and centers in column order
void read_points_and_centers(double *points, double *centers, int num_d, int num_p, int num_c) {
    // read points
    for (int i = 0; i < num_d * num_p; ++i) scanf("%lf", &points[i]);
    // read centers
    for (int i = 0; i < num_d * num_c; ++i) scanf("%lf", &centers[i]);
}