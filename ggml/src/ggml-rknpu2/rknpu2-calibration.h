#pragma once

#include <vector>
#include <cstddef>

/**
 * @brief Provides algorithms for tensor calibration and transformation.
 *
 * This namespace contains functions for finding optimal quantization parameters (amax)
 * using various statistical methods, as well as mathematical transformations like
 * the Hadamard transform used in advanced quantization schemes.
 */
namespace rknpu2_calibration {

// --- Hadamard Transform Implementations ---

/**
 * @brief Applies a Fast Walsh-Hadamard Transform.
 *
 * This version is optimized for scenarios where padding is always expected. It takes separate
 * source and destination buffers and avoids extra allocations by using a thread-local buffer.
 *
 * @param dst Pointer to the destination buffer. Must be of size `padded_size`.
 * @param src Pointer to the source data buffer of size `K`.
 * @param K The original number of elements in the source data.
 * @param padded_size The target size for the transform (must be a power of two >= K). The full result is written to dst.
 */
void hadamard_transform(float* dst, const float* src, int K, int padded_size);

/**
 * @brief Calculates the next power of two for a given integer.
 * @param n The input integer.
 * @return The smallest power of two that is greater than or equal to n.
 */
int next_power_of_two(int n);

// --- Low-Rank Approximation Implementations ---

/**
 * @brief Computes a strictly upper-bounding Rank-1 decomposition (u * v^T >= S)
 * of a scale matrix in closed form.
 *
 * @param matrix Row-major weight matrix of shape (N_seg x K_seg).
 * @param K_seg Number of columns (K dimension).
 * @param N_seg Number of rows (N dimension).
 * @param qmax Maximum quantization value.
 * @param u_out Output-axis factors (size N_seg).
 * @param v_out Input-axis factors (size K_seg).
 */
void lowrank_decomposition(const float * matrix, int K_seg, int N_seg, float qmax, float * u_out, float * v_out);

} // namespace rknpu2_calibration