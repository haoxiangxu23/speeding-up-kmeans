#include <immintrin.h>
#include <omp.h>
// p for points (column majored)
// i_d for intermediate_distance
// c for centers (column majored)
// kernel size k * n = 4 * 8
// k = 4, n = 8
void kernel(
    int k,
    int n,
    __m256d *ymm0, __m256d *ymm1, __m256d *ymm2, __m256d *ymm3,
    __m256d *ymm4, __m256d *ymm5, __m256d *ymm6, __m256d *ymm7,
    double *restrict p,
    double *restrict c)
{
    __m256d ymm8, ymm9, ymm10, ymm11, ymm12;

    /** dimension 0 **/
    ymm8 = _mm256_load_pd(p);
    ymm9 = _mm256_load_pd(p + k);
    // for center c in 4 centers:
    //column majored c11
    ymm10 = _mm256_broadcast_sd(&c[0]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm0 = _mm256_fmadd_pd(ymm11, ymm11, *ymm0);
    *ymm4 = _mm256_fmadd_pd(ymm12, ymm12, *ymm4);
    // column majored c12
    ymm10 = _mm256_broadcast_sd(&c[1]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm1 = _mm256_fmadd_pd(ymm11, ymm11, *ymm1);
    *ymm5 = _mm256_fmadd_pd(ymm12, ymm12, *ymm5);
    //column majored c13
    ymm10 = _mm256_broadcast_sd(&c[2]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm2 = _mm256_fmadd_pd(ymm11, ymm11, *ymm2);
    *ymm6 = _mm256_fmadd_pd(ymm12, ymm12, *ymm6);
    //column majored c14
    ymm10 = _mm256_broadcast_sd(&c[3]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm3 = _mm256_fmadd_pd(ymm11, ymm11, *ymm3);
    *ymm7 = _mm256_fmadd_pd(ymm12, ymm12, *ymm7);

    /** dimension 1 **/
    ymm8 = _mm256_load_pd(p + n);
    ymm9 = _mm256_load_pd(p + n + k);
    // for center c in 4 centers:
    //column majored c11
    ymm10 = _mm256_broadcast_sd(&c[k]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm0 = _mm256_fmadd_pd(ymm11, ymm11, *ymm0);
    *ymm4 = _mm256_fmadd_pd(ymm12, ymm12, *ymm4);
    // column majored c12
    ymm10 = _mm256_broadcast_sd(&c[k + 1]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm1 = _mm256_fmadd_pd(ymm11, ymm11, *ymm1);
    *ymm5 = _mm256_fmadd_pd(ymm12, ymm12, *ymm5);
    //column majored c13
    ymm10 = _mm256_broadcast_sd(&c[k + 2]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm2 = _mm256_fmadd_pd(ymm11, ymm11, *ymm2);
    *ymm6 = _mm256_fmadd_pd(ymm12, ymm12, *ymm6);
    //column majored c14
    ymm10 = _mm256_broadcast_sd(&c[k + 3]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm3 = _mm256_fmadd_pd(ymm11, ymm11, *ymm3);
    *ymm7 = _mm256_fmadd_pd(ymm12, ymm12, *ymm7);

    /** dimension 2 **/
    ymm8 = _mm256_load_pd(p + n * 2);
    ymm9 = _mm256_load_pd(p + n * 2 + k);
    // for center c in 4 centers:
    //column majored c11
    ymm10 = _mm256_broadcast_sd(&c[k * 2]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm0 = _mm256_fmadd_pd(ymm11, ymm11, *ymm0);
    *ymm4 = _mm256_fmadd_pd(ymm12, ymm12, *ymm4);
    // column majored c12
    ymm10 = _mm256_broadcast_sd(&c[k * 2 + 1]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm1 = _mm256_fmadd_pd(ymm11, ymm11, *ymm1);
    *ymm5 = _mm256_fmadd_pd(ymm12, ymm12, *ymm5);
    //column majored c13
    ymm10 = _mm256_broadcast_sd(&c[k * 2 + 2]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm2 = _mm256_fmadd_pd(ymm11, ymm11, *ymm2);
    *ymm6 = _mm256_fmadd_pd(ymm12, ymm12, *ymm6);
    //column majored c14
    ymm10 = _mm256_broadcast_sd(&c[k * 2 + 3]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm3 = _mm256_fmadd_pd(ymm11, ymm11, *ymm3);
    *ymm7 = _mm256_fmadd_pd(ymm12, ymm12, *ymm7);

    /** dimension 3 **/
    ymm8 = _mm256_load_pd(p + n * 3);
    ymm9 = _mm256_load_pd(p + n * 3 + k);
    // for center c in 4 centers:
    //column majored c11
    ymm10 = _mm256_broadcast_sd(&c[k * 3]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm0 = _mm256_fmadd_pd(ymm11, ymm11, *ymm0);
    *ymm4 = _mm256_fmadd_pd(ymm12, ymm12, *ymm4);
    // column majored c12
    ymm10 = _mm256_broadcast_sd(&c[k * 3 + 1]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm1 = _mm256_fmadd_pd(ymm11, ymm11, *ymm1);
    *ymm5 = _mm256_fmadd_pd(ymm12, ymm12, *ymm5);
    //column majored c13
    ymm10 = _mm256_broadcast_sd(&c[k * 3 + 2]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm2 = _mm256_fmadd_pd(ymm11, ymm11, *ymm2);
    *ymm6 = _mm256_fmadd_pd(ymm12, ymm12, *ymm6);
    //column majored c14
    ymm10 = _mm256_broadcast_sd(&c[k * 3 + 3]);
    ymm11 = _mm256_sub_pd(ymm8, ymm10);
    ymm12 = _mm256_sub_pd(ymm9, ymm10);
    *ymm3 = _mm256_fmadd_pd(ymm11, ymm11, *ymm3);
    *ymm7 = _mm256_fmadd_pd(ymm12, ymm12, *ymm7);
}

// given 2 groups of 4 points, update the corresponding min distances and cluster index
void findCluster(__m256d *dist_p0_c0, __m256d *dist_p0_c1, __m256d *dist_p0_c2, __m256d *dist_p0_c3,
                   __m256d *dist_p1_c0, __m256d *dist_p1_c1, __m256d *dist_p1_c2, __m256d *dist_p1_c3,
                   double *tmp_cluster_arr_p1, double *tmp_min_dist_arr_p1,
                   double *tmp_cluster_arr_p2, double *tmp_min_dist_arr_p2, double start_cluster_index)
{
  // load and initialization
  __m256d incr = _mm256_set_pd(1.0, 1.0, 1.0, 1.0);
  __m256d cur_index = _mm256_set_pd(start_cluster_index, start_cluster_index, start_cluster_index, start_cluster_index);
  __m256d tmp_cluster = _mm256_load_pd(tmp_cluster_arr_p1);
  __m256d tmp_min_dist = _mm256_load_pd(tmp_min_dist_arr_p1);

  // update the clutser index and min distance
  __m256d mask = _mm256_cmp_pd(*dist_p0_c0, tmp_min_dist, _CMP_GT_OQ);
  tmp_cluster = _mm256_blendv_pd(cur_index, tmp_cluster, mask);
  tmp_min_dist = _mm256_blendv_pd(*dist_p0_c0, tmp_min_dist, mask);
  cur_index = _mm256_add_pd(cur_index, incr);

  mask = _mm256_cmp_pd(*dist_p0_c1, tmp_min_dist, _CMP_GT_OQ);
  tmp_cluster = _mm256_blendv_pd(cur_index, tmp_cluster, mask);
  tmp_min_dist = _mm256_blendv_pd(*dist_p0_c1, tmp_min_dist, mask);
  cur_index = _mm256_add_pd(cur_index, incr);

  mask = _mm256_cmp_pd(*dist_p0_c2, tmp_min_dist, _CMP_GT_OQ);
  tmp_cluster = _mm256_blendv_pd(cur_index, tmp_cluster, mask);
  tmp_min_dist = _mm256_blendv_pd(*dist_p0_c2, tmp_min_dist, mask);
  cur_index = _mm256_add_pd(cur_index, incr);

  mask = _mm256_cmp_pd(*dist_p0_c3, tmp_min_dist, _CMP_GT_OQ);
  tmp_cluster = _mm256_blendv_pd(cur_index, tmp_cluster, mask);
  tmp_min_dist = _mm256_blendv_pd(*dist_p0_c3, tmp_min_dist, mask);
  cur_index = _mm256_add_pd(cur_index, incr);

  // store the tmp result
  _mm256_store_pd(tmp_cluster_arr_p1, tmp_cluster);
  _mm256_store_pd(tmp_min_dist_arr_p1, tmp_min_dist);

  // second group
  cur_index = _mm256_set_pd(start_cluster_index, start_cluster_index, start_cluster_index, start_cluster_index);
  tmp_cluster = _mm256_load_pd(tmp_cluster_arr_p2);
  tmp_min_dist = _mm256_load_pd(tmp_min_dist_arr_p2);

  // update the clutser index and min distance
  mask = _mm256_cmp_pd(*dist_p1_c0, tmp_min_dist, _CMP_GT_OQ);
  tmp_cluster = _mm256_blendv_pd(cur_index, tmp_cluster, mask);
  tmp_min_dist = _mm256_blendv_pd(*dist_p1_c0, tmp_min_dist, mask);
  cur_index = _mm256_add_pd(cur_index, incr);

  mask = _mm256_cmp_pd(*dist_p1_c1, tmp_min_dist, _CMP_GT_OQ);
  tmp_cluster = _mm256_blendv_pd(cur_index, tmp_cluster, mask);
  tmp_min_dist = _mm256_blendv_pd(*dist_p1_c1, tmp_min_dist, mask);
  cur_index = _mm256_add_pd(cur_index, incr);

  mask = _mm256_cmp_pd(*dist_p1_c2, tmp_min_dist, _CMP_GT_OQ);
  tmp_cluster = _mm256_blendv_pd(cur_index, tmp_cluster, mask);
  tmp_min_dist = _mm256_blendv_pd(*dist_p1_c2, tmp_min_dist, mask);
  cur_index = _mm256_add_pd(cur_index, incr);

  mask = _mm256_cmp_pd(*dist_p1_c3, tmp_min_dist, _CMP_GT_OQ);
  tmp_cluster = _mm256_blendv_pd(cur_index, tmp_cluster, mask);
  tmp_min_dist = _mm256_blendv_pd(*dist_p1_c3, tmp_min_dist, mask);
  cur_index = _mm256_add_pd(cur_index, incr);

  // store the tmp result
  _mm256_store_pd(tmp_cluster_arr_p2, tmp_cluster);
  _mm256_store_pd(tmp_min_dist_arr_p2, tmp_min_dist);
}

void updateCluster(double *tmp_cluster_arr_g1, double *tmp_cluster_arr_g2, double *points, 
                    double *new_centers, int p, int d, int *cluster_size) {
  // #pragma omp parallel for num_threads(4)
  for (int i = 0; i < 4; i++) {
    int index1 = tmp_cluster_arr_g1[i];
    int index2 = tmp_cluster_arr_g2[i];
    __m256d vector_c_1;
    __m256d vector_p_1;
    __m256d vector_c_2;
    __m256d vector_p_2;
    for (int j = 0; j < d; j += 4) {
      // group 1
      vector_c_1 = _mm256_load_pd(new_centers + index1 * d + j);
      vector_p_1 = _mm256_load_pd(points + i * d + j);
      vector_c_1 = _mm256_add_pd(vector_c_1, vector_p_1);
      _mm256_store_pd(new_centers + index1 * d + j, vector_c_1);
      // group 2
      vector_c_2 = _mm256_load_pd(new_centers + index2 * d + j);
      vector_p_2 = _mm256_load_pd(points + (i + 4) * d + j);
      vector_c_2 = _mm256_add_pd(vector_c_2, vector_p_2);
      _mm256_store_pd(new_centers + index2 * d + j, vector_c_2);
    }
    cluster_size[index1]++;
    cluster_size[index2]++;
  }
}

void computeNewCluster(double *new_centers, int *new_centers_size, int dsize, int k) {
  double size;
  __m256d vector_size;
  __m256d vector_c;
  for (int i = 0; i < k; i++) {
      if (new_centers_size[i] == 0) {
          continue;
      }
      size = (double) (1.0 / new_centers_size[i]);
      vector_size = _mm256_broadcast_sd(&size);
      for (int j = 0; j < dsize; j+= 4) {
          vector_c = _mm256_load_pd(new_centers + i * dsize + j);
          vector_c = _mm256_mul_pd(vector_c, vector_size);
          _mm256_store_pd(new_centers + i * dsize + j, vector_c);
      }
  }
}
