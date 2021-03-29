#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cuda_utils.h"
#include "utils.h"


// input: group_points(B,S,N,C), group_num(B,S), tmp(B,S,N)
// ouput: fps_idx(B,S,N)
// b,s: parallel

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__device__ void furthest_point_sampling_kernel(
    int b, int n, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
  if (m <= 0) return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      if (mag <= 1e-3) continue;

      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 512) {
      if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }
}


__global__ void fps_withmask_forward_kernel(int B, int N, int C, int S, const int K,
                                            const float* group_points,
                                            const int* group_num,
                                            float* tmps,
                                            int* fps_idx) {

    int batch_idx = blockIdx.x; //per batch per block
    group_points += batch_idx * S * N * C; // addr
    tmps += batch_idx * N * S; // addr
    group_num += batch_idx * S; // addr
    fps_idx += batch_idx * S * N; // addr

    int index = threadIdx.x;
    int stride = blockDim.x;
    /***
    // ----- parallel loop for S ---------
    for (int s = index; s < S; s += stride) {

        n = group_num[s];
        dataset = group_points + s * N * C;
        temp = tmps + n * S;
        idxs = fps_idx + s * N;

        if (n > K) {
          __shared__ float dists[512];
          __shared__ int dists_i[512];

          int tid = threadIdx.x;
          const int stride = block_size;

          int old = 0;
          if (threadIdx.x == 0) idxs[0] = old;

          __syncthreads();
          for (int j = 1; j < K; j++) {
            int besti = 0;
            float best = -1;
            float x1 = dataset[old * 3 + 0];
            float y1 = dataset[old * 3 + 1];
            float z1 = dataset[old * 3 + 2];
            for (int k = tid; k < n; k += stride) {
              float x2, y2, z2;
              x2 = dataset[k * 3 + 0];
              y2 = dataset[k * 3 + 1];
              z2 = dataset[k * 3 + 2];
              float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
              if (mag <= 1e-3) continue;

              float d =
                  (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

              float d2 = min(d, temp[k]);
              temp[k] = d2;
              besti = d2 > best ? k : besti;
              best = d2 > best ? d2 : best;
            }
            dists[tid] = best;
            dists_i[tid] = besti;
            __syncthreads();

            if (block_size >= 512) {
              if (tid < 256) {
                __update(dists, dists_i, tid, tid + 256);
              }
              __syncthreads();
            }
            if (block_size >= 256) {
              if (tid < 128) {
                __update(dists, dists_i, tid, tid + 128);
              }
              __syncthreads();
            }
            if (block_size >= 128) {
              if (tid < 64) {
                __update(dists, dists_i, tid, tid + 64);
              }
              __syncthreads();
            }
            if (block_size >= 64) {
              if (tid < 32) {
                __update(dists, dists_i, tid, tid + 32);
              }
              __syncthreads();
            }
            if (block_size >= 32) {
              if (tid < 16) {
                __update(dists, dists_i, tid, tid + 16);
              }
              __syncthreads();
            }
            if (block_size >= 16) {
              if (tid < 8) {
                __update(dists, dists_i, tid, tid + 8);
              }
              __syncthreads();
            }
            if (block_size >= 8) {
              if (tid < 4) {
                __update(dists, dists_i, tid, tid + 4);
              }
              __syncthreads();
            }
            if (block_size >= 4) {
              if (tid < 2) {
                __update(dists, dists_i, tid, tid + 2);
              }
              __syncthreads();
            }
            if (block_size >= 2) {
              if (tid < 1) {
                __update(dists, dists_i, tid, tid + 1);
              }
              __syncthreads();
            }

            old = dists_i[0];
            if (tid == 0) idxs[j] = old;
          }

    }

    ***/

}

void fps_withmask_forward_kernel_wrapper(int B, int N, int C, int S, int K,
                                               const at::Tensor& group_points,
                                               const at::Tensor& group_num,
                                               at::Tensor fps_idx) {

    CHECK_CONTIGUOUS(group_points);
    CHECK_CONTIGUOUS(group_num);
    CHECK_CONTIGUOUS(fps_idx);

    at::Tensor tmp =
      torch::full({B, S, N}, 1e10,
                  at::device(group_points.device()).dtype(at::ScalarType::Float));

    const float* group_points_data = group_points.data_ptr<float>();
    const int* group_num_data = group_num.data_ptr<int>();
    float* tmp_data = tmp.data_ptr<float>();
    int* fps_idx_data = fps_idx.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fps_withmask_forward_kernel<<<B, opt_n_threads(S*N), 0, stream>>>(
        B, N, C, S, K, group_points_data, group_num_data, tmp_data, fps_idx_data);

    CUDA_CHECK_ERRORS();

}


