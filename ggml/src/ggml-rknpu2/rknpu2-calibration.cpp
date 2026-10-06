#include "rknpu2-calibration.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <omp.h>

namespace rknpu2_calibration {

// --- Hadamard Transform Implementations ---

// Iterative Fast Walsh-Hadamard Transform (in-place)
static void fwht_iterative(float* data, int size) {
    for (int h = 1; h < size; h <<= 1) {
        for (int i = 0; i < size; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                float x = data[j];
                float y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
    }
}

int next_power_of_two(int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

void hadamard_transform(float* dst, const float* src, int K, int padded_size) {
    // If no padding is needed, copy and perform in-place.
    if (K == padded_size) {
        memcpy(dst, src, K * sizeof(float));
        fwht_iterative(dst, K);
        return;
    }

    // Using a thread-local buffer to avoid repeated heap allocations.
    thread_local static std::vector<float> padded_data;

    // Resizing the buffer only if the current one is too small.
    if (padded_data.size() < (size_t)padded_size) {
        padded_data.resize(padded_size);
    }

    // Copying source data and zero-fill the rest (padding).
    memcpy(padded_data.data(), src, K * sizeof(float));
    if (padded_size > K) {
        memset(padded_data.data() + K, 0, (padded_size - K) * sizeof(float));
    }

    // Applying the transform to our temporary buffer
    fwht_iterative(padded_data.data(), padded_size);

    // Copying the result to the destination buffer
    memcpy(dst, padded_data.data(), padded_size * sizeof(float));
}

// --- Low-Rank Approximation Implementations ---

// Function for upper-bounding rank-1 decomposition of scale matrix
void lowrank_decomposition(const float * matrix, int K_seg, int N_seg, float qmax, float * u_out, float * v_out) {
    std::fill(v_out, v_out + K_seg, 0.0f);

    // Calculating the 'u' vector
    #pragma omp parallel for
    for (int i = 0; i < N_seg; ++i) {
        const float * row_fp32 = matrix + (size_t)i * K_seg;
        float amax = 0.0f;
        for (int j = 0; j < K_seg; ++j) {
            amax = std::max(amax, std::abs(row_fp32[j]));
        }
        u_out[i] = std::max(amax / qmax, 1e-7f);
    }

    // Calculating the 'v' vector
    #pragma omp parallel
    {
        std::vector<float> v_local(K_seg, 0.0f);

        #pragma omp for schedule(static)
        for (int i = 0; i < N_seg; ++i) {
            const float * row_fp32 = matrix + (size_t)i * K_seg;
            const float u_i_inv = 1.0f / u_out[i];

            for (int j = 0; j < K_seg; ++j) {
                float s_ideal_ji = std::max(std::abs(row_fp32[j]) / qmax, 1e-7f);
                float val = s_ideal_ji * u_i_inv;
                if (val > v_local[j]) {
                    v_local[j] = val;
                }
            }
        }

        #pragma omp critical
        {
            for (int j = 0; j < K_seg; ++j) {
                if (v_local[j] > v_out[j]) {
                    v_out[j] = v_local[j];
                }
            }
        }
    }
}

} // namespace rknpu2_calibration